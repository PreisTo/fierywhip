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


class GRBList:
    def __init__(self):
        self._grbs = []
        self._load_swift_bursts()

    def _load_swift_bursts(
        self,
        swift_list=pkg_resources.resource_filename("effarea", "data/Fermi_Swift.lis"),
    ):
        table = pd.read_csv(swift_list, sep=" ", index_col=False, header=None)

        for j, i in table.iterrows():
            name = str(i.loc[0])
            ra = str(i.loc[5])
            dec = str(i.loc[6])
            ra_dec_units = (u.hourangle, u.deg)
            if len(name) < 9:
                name = f"GRB0{name}"
            else:
                name = f"GRB{name}"
            try:
                grb = GRB(name, ra, dec, ra_dec_units)
                self._grbs.append(grb)
            except GRBInitError:
                pass

    def _check_already_run(
        self, path=os.path.join(os.environ.get("GBMDATA"), "localizing/results.yml")
    ):
        pass

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

    def _get_trigdat_path(self):
        path = os.path.join(
            os.environ.get("GBMDATA"),
            "trigdat",
            str(self._time.year),
            f"glg_trigdat_all_bn{self._name.strip('GRB')}_v00.fit",
        )
        if not os.path.exists(path):
            download_trigdata_file(f"bn{self._name.strip('GRB')}")
        self._trigdat = path
