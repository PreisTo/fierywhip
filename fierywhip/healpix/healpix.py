#!/usr/bin/env python3
from threeML.analysis_results import BayesianResults, load_analysis_results
import logging
import numpy as np
import healpy as hp
import os


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

    def _create_healpix_map(self):
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
        """
        vec = radec2cartesian(pos)
        ids = hp.query_disc(self._nside, vec, np.deg2rad(angle))

        return np.sum(self._hp_map[ids])

    @property
    def healpix_map(self):
        return self._hp_map

    @classmethod
    def from_result_fits_file(cls, result_file: str, **kwargs):
        ar = load_analysis_results(result_file)
        return cls(analysis_result=ar, **kwargs)


def radec2cartesian(pos):
    pos = np.deg2rad(pos)
    vec = [
        np.cos(pos[0]) * np.cos(pos[1]),
        np.sin(pos[0]) * np.cos(pos[1]),
        np.sin(pos[1]),
    ]
    return vec
