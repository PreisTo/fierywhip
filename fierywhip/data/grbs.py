#!/usr/bin/python

import pkg_resources
from astropy.coordinates import SkyCoord
import astropy.time as time
import astropy.units as u
import pandas as pd
from datetime import datetime, timedelta
from gbmgeometry.utils.gbm_time import GBMTime
import os
from fierywhip.io.downloading import download_tte_file, download_cspec_file
from gbmbkgpy.io.downloading import download_trigdata_file
from urllib.error import URLError
import yaml
from mpi4py import MPI
from fierywhip.detectors.detectors import DetectorSelection, DetectorSelectionError
from fierywhip.normalizations.normalization_matrix import NormalizationMatrix
from morgoth.auto_loc.time_selection import TimeSelectionBB
from fierywhip.config.configuration import fierywhip_config
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

lu = [
    "n0",
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
    "n7",
    "n8",
    "n9",
    "na",
    "nb",
]
month_lu = {
    "JAN": "01",
    "FEB": "02",
    "MAR": "03",
    "APR": "04",
    "MAY": "05",
    "JUN": "06",
    "JUL": "07",
    "AUG": "08",
    "SEP": "09",
    "OCT": "10",
    "NOV": "11",
    "DEC": "12",
}


class GRBList:
    """
    Class to load GRB positions and times from all different sources
    """

    def __init__(self, check_finished=True):
        self._check_finished = check_finished
        self._grbs = []
        namess, rass, decss = self._load_swift_bursts()
        namesi, rasi, decsi, typesi = self._load_ipn_bursts()
        names_all = namesi
        ras_all = rasi
        decs_all = decsi
        types_all = list(typesi)
        for i, n in enumerate(namess):
            if n not in names_all:
                names_all.append(n)
                ras_all.append(rass[i])
                decs_all.append(decss[i])
                types_all.append("swift")
        self._table = pd.DataFrame(
            {"name": names_all, "ra": ras_all, "dec": decs_all, "type": types_all},
            index=None,
        )
        self._table.sort_values(by="name", inplace=True)
        self._create_grb_objects()

    def _create_grb_objects(self):
        for index, row in self._table.iloc[:10].iterrows():
            if not self._check_already_run(row["name"]):
                try:
                    grb = GRB(row["name"], row["ra"], row["dec"])
                    self._grbs.append(grb)
                except GRBInitError:
                    pass

    def _load_swift_bursts(
        self,
        swift_list=pkg_resources.resource_filename("fierywhip", "data/Fermi_Swift.lis"),
    ):
        """
        Loads Fermi-Swift burst provided in package resources
        """

        names, ras, decs = [], [], []
        if fierywhip_config.swift:
            self._swift_table = pd.read_csv(
                swift_list, sep=" ", index_col=False, header=None
            )
            for j, i in self._swift_table.iterrows():
                name = str(i.loc[0])
                ra = str(i.loc[5])
                dec = str(i.loc[6])
                ra_dec_units = (u.hourangle, u.deg)
                if len(name) < 9:
                    name = f"GRB0{name}"
                else:
                    name = f"GRB{name}"
                names.append(name)
                coord = SkyCoord(ra=ra, dec=dec, unit=ra_dec_units)
                ras.append(round(coord.ra.deg, 3))
                decs.append(round(coord.dec.deg, 3))
            print("Done loading Swift List")
        return names, ras, decs

    def _load_ipn_bursts(
        self, table_path=pkg_resources.resource_filename("fierywhip", "data/ipn.csv")
    ):
        names = []
        ras = []
        decs = []
        types = []
        if fierywhip_config.ipn:
            self._ipn_table = pd.read_csv(table_path, sep=" ", index_col=False)
            for i, b in self._ipn_table.iterrows():
                names.append(f"GRB{str(b['name']).strip('bn')}")
                ras.append(float(b["ra"]))
                decs.append(float(b["dec"]))
                types.append("ipn")
            print("Done loading IPN list")
        return names, ras, decs, types

    def _check_already_run(
        self,
        name,
        path=os.path.join(os.environ.get("GBMDATA"), "localizing/results.yml"),
    ):
        """
        Check if GRB is already in result file, if so skip it
        :param name: grb name as string
        :param path: path like to result file
        """
        if self._check_finished:
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
            else:
                ret = None
            ret = comm.bcast(ret, root=0)
            return ret
        else:
            return False

    @property
    def grbs(self):
        """
        :returns: list with grb objects
        """
        return self._grbs

    @property
    def table(self):
        return self._table


class GRB:
    def __init__(self, name, ra, dec, ra_dec_units=None, grb_time=None):
        """
        :param name: name of grb - needs to be like GRB231223001
        :param ra: ra of grb
        :param dec: dec of grb
        :param ra_dec_units: units of ra and dec as list like - astropy.units
        :param grb_time: optional datetime object of grb_time
        :param normalizing_matrix: normalizing matrix used to setting the effective area corrections
        """
        self._name = name
        self._active_time = None
        self._bkg_time = None
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
        self.download_files()

    @property
    def position(self):
        """
        :returns: SkyCoord of GRB
        """
        return self._position

    @property
    def name(self):
        """
        :returns: str of grb name like GRB231223001
        """
        return self._name

    @property
    def time(self):
        """
        :returns: datetime of grb time
        """
        return self._time

    @property
    def time_astropy(self):
        """
        :returns: astropy.time.Time object of grb time
        """
        astro_time = time.Time(self._time, format="datetime", scale="utc")
        return astro_time

    @property
    def time_gbm(self):
        """
        :returns: gbmgeometry.utils.gbm_time.GBMtime object of grb time
        """
        gbm_time = GBMTime(self.time_astropy)
        return gbm_time

    @property
    def trigdat(self):
        """
        :returns: path to trigdat file
        """
        return self._trigdat

    @property
    def detector_selection(self):
        """
        :returns: DetectorSelection object for this grb
        """
        return self._detector_selection

    @property
    def active_time(self):
        """
        :returns: string with start/stop time of trigger
        """
        return self._active_time

    @property
    def bkg_time(self):
        """
        :returns: list with bkg neg and bkg pos start/stop time
        """
        return self._bkg_time

    def download_files(self):
        """
        Downloading TTE and CSPEC files from FTP
        """
        print("Downloading TTE and CSPEC files")
        self.tte_files = {}
        self.cspec_files = {}
        for d in self.detector_selection.good_dets:
            self.tte_files[d] = download_tte_file(self._name, d)
            self.cspec_files[d] = download_cspec_file(self._name, d)

    def _get_trigdat_path(self):
        """
        sets path to trigdat file
        """
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

    def _get_detector_selection(
        self,
        max_sep=fierywhip_config.max_sep,
        max_sep_normalizing=fierywhip_config.max_sep_norm,
    ):
        """
        :param max_sep: max separation of center from det in deg
        :param max_sep_normalization: max sep of center for det used as normalization
        """
        self._detector_selection = DetectorSelection(
            self, max_sep=max_sep, max_sep_normalizing=max_sep_normalizing
        )
        print(self._detector_selection.good_dets)

    def run_timeselection(self):
        """
        Timeselection for GRB using morogth auto_loc timeselection
        """
        if self._active_time is None and self._bkg_time is None:
            tsbb = TimeSelectionBB(self._name, self._trigdat, fine=True)
            self._active_time = tsbb.active_time
            self._bkg_time = [tsbb.background_time_neg, tsbb.background_time_pos]

    def _get_effective_area_correction(self, nm):
        print(type(nm))
        self._normalizing_matrix = nm
        norm_det = self._detector_selection.normalizing_det
        good_dets = self._detector_selection.good_dets
        norm_id = lu.index(norm_det)
        row = self._normalizing_matrix[norm_id]
        eff_area_dict = {}
        for gd in good_dets:
            if gd != norm_det and gd not in ("b0", "b1"):
                i = lu.index(gd)
                eff_area_dict[gd] = row[i]
            else:
                eff_area_dict[gd] = 1

        self._effective_area_dict = eff_area_dict

    def effective_area_correction(self, det):
        return self._effective_area_dict[det]


class GRBInitError(Exception):
    """
    Error in Initializing the GRB Object
    """

    def __init__(self, message=None):
        if message is None:
            message = "Failed to initialize GRB Object"
        super().__init__(message)
