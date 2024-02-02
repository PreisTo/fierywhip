#!/usr/bin/env python3
from threeML.analysis_results import BayesianResults, load_analysis_results
import logging
import numpy as np
import healpy as hp
import os
from scipy import special


class MorgothHealpix:
    """
    small class to create healpix maps for the ra-dec probabilites
    """

    def __init__(self, analysis_result: BayesianResults, **kwargs):
        """
        Creates the Healpix Map from an threeML BayesianResults instance

            :param analysis_result: Bayesian Analysis Results
        :type analysis_result: threeML.analysis_results.BayesianResults
        """
        self._result = analysis_result
        logging.info(
            "We are assuming the first two dimensions of the samples "
            + "correspond to Ra and Dec respectively"
        )
        self._nside = kwargs.get("nside", 512)
        self._result_path = kwargs.get(
            "result_path",
            os.path.join(os.environ.get("GBM_TRIGGER_DATA_DIR"), "healpix.fits"),
        )
        self._hp_map = kwargs.get("hp_map", None)
        if self._hp_map is None and self._result is not None:
            self.create_healpix_map()

    def create_healpix_map(self):
        npix = hp.nside2npix(self._nside)
        healpix_map = np.zeros(npix)

        ras = self._result.samples[0]
        # have to wrap the ras to -180-180
        ras[ras > 180] -= 360
        ras[ras <= -180] += 360

        # should not be necessary for decs but lets do it nevertheless
        decs = self._result.samples[1]
        decs[decs > 90] -= 180
        decs[decs <= -90] += 180

        prob = self._result.log_probability
        prob = np.exp(prob)
        # numpy can apparently work better with bigger numbers
        # will take care of that later
        prob = prob / np.min(prob)

        # getting the corresponding pixel_ids for our ra-dec sample pairs
        ids = hp.ang2pix(self._nside, ras, decs, lonlat=True)

        # create the numpy array and set the probabilites
        hp_map = np.zeros(hp.nside2npix(self._nside))
        hp_map[ids] = prob
        # normalize this shit
        hp_map = hp_map / np.sum(hp_map)

        self._hp_map = hp_map
        logging.info("Healpix Map was created successfully")

    def save_healpix_map(self, path=None):
        """
        Saves the healpix as a .fits file

        :param path: path + filename where the healpix is stored, defaults to
            $GBM_TRIGGER_DATA_DIR/healpix.fits
        :type path: pathlike
        """
        if path is None:
            logging.info(f"No path passed to function, will use {self._result_path}")
        else:
            self._result_path = path

        hp.write_map(self._result_path, self._hp_map)

    def probability_circle(self, pos, angle):
        """
        Returns the probability contained in a circle around a given position
        :param pos: Ra Dec Position in deg
        :type pos: list, tuple
        :param angle: angle in degree
        :type angle: float

        :return: probability inside the circle
        :rtype: float
        """
        pos = ra_dec_wrapper(pos)
        vec = radec2cartesian(pos)
        ids = hp.query_disc(self._nside, vec, np.deg2rad(angle))

        return np.sum(self._hp_map[ids])

    def sigma_radius(self, pos, sigma,step_size = 0.1):
        """
        Returns radius needed to contain Nr of sigma
        May take some time if step_size too small

        :param pos: position of the center in ra/dec
        :type pos: list/tuple
        :param sigma: nr of sigma
        :type sigma: int or list
        :param step_size: step size for which the radius is increased,
            defaults to 0.1
        :type step_size: float

        :return: radius or list of radii
        """
        pos = ra_dec_wrapper(pos)
        probs = []
        radii = []
        p = 0
        r = 0
        while p < np.sum(self._hp_map):
            p = self.probability_circle(pos, r)
            probs.append(p)
            radii.append(r)
            r += step_size
        probs = np.array(probs)
        radii = np.array(radii)
        if type(sigma) == list:
            return_radius = []
            for s in sigma:
                percentage = special.erf(s)
                s_radius = radii[np.argwhere(probs >= percentage)[0, 0]]
                return_radius.append(s_radius)
            return return_radius
        else:
            percentage = special.erf(sigma)
            return radii[np.argwhere(probs >= percentage)[0, 0]]

    @property
    def healpix_map(self):
        return self._hp_map

    @classmethod
    def from_result_fits_file(cls, result_file: str, **kwargs):
        ar = load_analysis_results(result_file)
        return cls(analysis_result=ar, **kwargs)

    @classmethod
    def from_healpix_file(cls, healpix_file):
        hp_map = hp.read_map(healpix_file)
        return cls(analysis_result=None, hp_map=hp_map)


def radec2cartesian(pos):
    pos = np.deg2rad(pos)
    vec = [
        np.cos(pos[0]) * np.cos(pos[1]),
        np.sin(pos[0]) * np.cos(pos[1]),
        np.sin(pos[1]),
    ]
    return vec

def ra_dec_wrapper(pos):
    ras = pos[0]
    decs = pos[1]
    if type(ras) == float:
        if ras >180:
            ras -= 360
        elif ras <= -180:
            ras += 360
        if decs > 90:
            decs-= 180
        elif decs <= -90:
            decs += 180
    else:
        ras = np.array(ras)
        decs = np.array(decs)
        ras[ras>180] -= 360
        ras[ras<=-180] += 360
        decs[decs>90] -= 180
        decs[decs<-90] += 180
    return ras,decs

