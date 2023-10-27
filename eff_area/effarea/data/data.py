#!/usr/bin/python

import pkg_resources
from astropy.coordinates import SkyCoord
import astropy.time as time
import astropy.units as u
import pandas as pd
from datetime import datetime, timedelta
from gbmgeometry.utils.gbm_time import GBMTime
import os
from gbmbkgpy.io.downloading import download_trigdata_file
from urllib.error import URLError
import yaml
from mpi4py import MPI
from effarea.detectors.detectors import DetectorSelection, DetectorSelectionError

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


class GRBList:
    def __init__(self):
        self._grbs = []
        self._load_swift_bursts()

    def _load_swift_bursts(
        self,
        swift_list=pkg_resources.resource_filename("effarea", "data/Fermi_Swift.lis"),
    ):
        table = pd.read_csv(swift_list, sep=" ", index_col=False, header=None)
        # TODO Remove before flight
        for j, i in table.iloc[0:100].iterrows():
            name = str(i.loc[0])
            ra = str(i.loc[5])
            dec = str(i.loc[6])
            ra_dec_units = (u.hourangle, u.deg)
            if len(name) < 9:
                name = f"GRB0{name}"
            else:
                name = f"GRB{name}"
            try:
                if not self._check_already_run(name):
                    grb = GRB(name, ra, dec, ra_dec_units)
                    self._grbs.append(grb)
            except GRBInitError:
                pass

    def _check_already_run(
        self,
        name,
        path=os.path.join(os.environ.get("GBMDATA"), "localizing/results.yml"),
    ):
        if rank == 0:
            if os.path.exists(path):
                with open(path, "r") as f:
                    already_run_dict = yaml.safe_load(f)
                if name in already_run_dict.keys():
                    ret = True
                else:
                    ret = False
            else:
                ret = False
        ret = comm.bcast(ret, root=0)
        return ret

    @property
    def grbs(self):
        return self._grbs


class GRB:
    def __init__(self, name, ra, dec, ra_dec_units=None, grb_time=None):
        self._name = name
        self._ra_icrs = ra
        self._dec_icrs = dec
        if grb_time is not None:
            assert (
                type(grb_time) is datetime
            ), "Wrong grb_time type, needs to be datetime"
            self._time = grb_time
        else:
            grb = self._name.strip("GRB")
            year = grb[:2]
            month = grb[2:4]
            day = grb[4:6]
            frac = grb[6:]
            tot_seconds = 24 * 3600 / 1000
            self._time = datetime(int(year), int(month), int(day)) + timedelta(
                seconds=tot_seconds * int(frac)
            )
        if ra_dec_units is None:
            units = (u.deg, u.deg)
        else:
            units = ra_dec_units
        self._position = SkyCoord(ra=ra, dec=dec, unit=units, frame="icrs")
        self._get_trigdat_path()
        try:
            self._get_detector_selection()
        except DetectorSelectionError:
            raise GRBInitError

    @property
    def position(self):
        return self._position

    def name(self):
        return self._name

    @property
    def time(self):
        return self._time

    @property
    def time_astropy(self):
        astro_time = time.Time(self._time, format="datetime", scale="utc")
        return astro_time

    @property
    def time_gbm(self):
        gbm_time = GBMTime(self.time_astropy)
        return gbm_time

    @property
    def trigdat(self):
        return self._trigdat

    @property
    def detector_selection(self):
        return self._detector_selection

    def _get_trigdat_path(self):
        trigdat_path = os.path.join(
            os.environ.get("GBMDATA"),
            "trigdat/",
            f"20{self._name.strip('GRB')[:2]}",
            f"glg_trigdat_all_bn{self._name.strip('GRB')}_v00.fit",
        )
        if not os.path.exists(trigdat_path):
            try:
                download_trigdata_file(f"bn{self._name.strip('GRB')}")
            except (TypeError, URLError):
                raise GRBInitError
        if os.path.exists(trigdat_path):
            self._trigdat = trigdat_path
        else:
            raise GRBInitError

    def _get_detector_selection(self, max_sep=60, max_sep_normalizing=20):
        self._detector_selection = DetectorSelection(
            self, max_sep=60, max_sep_normalizing=max_sep_normalizing
        )


class GRBInitError(Exception):
    pass
