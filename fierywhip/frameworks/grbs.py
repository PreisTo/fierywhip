#!/usr/bin/python

import pkg_resources
from astropy.coordinates import SkyCoord
import astropy.time as time
import astropy.units as u
import pandas as pd
from datetime import datetime, timedelta
from gbmgeometry.utils.gbm_time import GBMTime
from gbmgeometry.position_interpolator import PositionInterpolator
from gbmgeometry.gbm_frame import GBMFrame
import os
from fierywhip.io.downloading import download_tte_file, download_cspec_file
from gbmbkgpy.io.downloading import download_trigdata_file
from urllib.error import URLError, HTTPError
from urllib.request import urlopen
import yaml
from mpi4py import MPI
from fierywhip.detectors.detector_selection import (
    DetectorSelection,
    DetectorSelectionError,
)
from fierywhip.utils.detector_utils import detector_list, nai_list
from fierywhip.timeselection.timeselection import TimeSelectionNew
from fierywhip.timeselection.split_active_time import time_splitter
from fierywhip.config.configuration import fierywhip_config
from fierywhip.frameworks.eac import EffectiveAreaNormalization
import numpy as np
from threeML.utils.progress_bar import trange
import logging

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

lu = detector_list()
lu_nai = nai_list()

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

    def __init__(
        self,
        check_finished=fierywhip_config.config.grb_list.check_finished,
        run_det_sel=fierywhip_config.config.grb_list.run_det_sel,
        testing=fierywhip_config.config.grb_list.testing,
        reverse=fierywhip_config.config.grb_list.reverse,
        **kwargs,
    ):
        """
        :param check_finished:  looks up the localizing/results.yml if GRB is
                                already in there
        :param run_det_sel: pass through argument for GRB class - run detector
                            selection
        :param testing: run the whole dataset
        :type testing:  bool or int: if bool then first 20 entries will be used
                        if int this will define number of entries
        """
        self._check_finished = check_finished
        self._run_det_sel = run_det_sel
        self._grbs = []
        self._testing = testing
        if rank == 0:
            namess, rass, decss = self._load_swift_bursts()
            namesf, rasf, decsf = self._load_others(skip=namess)
            names_all = namess + namesf
            ras_all = rass + rasf
            decs_all = decss + decsf
            types_all = ["swift"] * len(decss) + ["other"] * len(decsf)
            self._table = pd.DataFrame(
                {"name": names_all, "ra": ras_all, "dec": decs_all, "type": types_all},
                index=None,
            )

            self._table.sort_values(by="name", inplace=True)
            if reverse:
                self._table.sort_values(by="name", ascending=False, inplace=True)
            self._table.reset_index(inplace=True)
        else:
            self._table = None

        self._table = comm.bcast(self._table, root=0)
        if fierywhip_config.config.grb_list.create_objects:
            self._create_grb_objects()

    def _create_grb_objects(self):
        grbs_temp = []
        if rank == 0:
            if isinstance(self._testing, bool):
                if self._testing:
                    stop_last = 20
                else:
                    stop_last = len(self._table)

            elif isinstance(self._testing, int):
                stop_last = self._testing
            size_per_rank = stop_last // size
        else:
            size_per_rank = None
            stop_last = None
        size_per_rank = comm.bcast(size_per_rank, root=0)
        stop_last = comm.bcast(stop_last, root=0)
        start = size_per_rank * rank
        stop = size_per_rank * (rank + 1)
        if rank == size - 1:
            stop = stop_last
        for index, row in self._table.iloc[start:stop].iterrows():
            if not self._check_already_run(row["name"]):
                try:
                    logging.debug(f"Creating Object for {row['name']}")
                    grb = GRB(
                        name=row["name"],
                        ra=row["ra"],
                        dec=row["dec"],
                        run_det_sel=self._run_det_sel,
                    )
                    grbs_temp.append(grb)
                except GRBInitError:
                    pass
        comm.Barrier()
        grbs = comm.gather(grbs_temp, root=0)
        if rank == 0:
            for g in grbs:
                self._grbs.extend(g)
        else:
            self._grbs = None
        self._grbs = comm.bcast(self._grbs, root=0)

    def _load_swift_bursts(
        self,
        swift_list=pkg_resources.resource_filename("fierywhip", "data/Fermi_Swift.lis"),
    ):
        """
        Loads Fermi-Swift burst provided in package resources
        """
        names, ras, decs = [], [], []
        if fierywhip_config.config.swift:
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
            logging.info("Done loading Swift List")
        return names, ras, decs

    def _load_others(
        self,
        full_list=pkg_resources.resource_filename(
            "fierywhip", "data/full_jcg_list.csv"
        ),
        skip=[],
    ):
        names, ras, decs = [], [], []
        if fierywhip_config.config.full_list:
            self._full_list = pd.read_csv(full_list, index_col=0)
            for n in self._full_list["name"]:
                if f"GRB{n.strip('bn')}" not in skip:
                    row = self._full_list[self._full_list["name"] == n]
                    names.append(f"GRB{n.strip('bn')}")
                    ra = row["ra"][0]
                    dec = row["dec"][0]
                    coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
                    ras.append(round(coord.ra.deg, 3))
                    decs.append(round(coord.dec.deg, 3))

        return names, ras, decs

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
    def __init__(self, **kwargs):
        """
        :param name: name of grb - needs to be like GRB231223001
        :param ra: ra of grb
        :param dec: dec of grb
        :param ra_dec_units: units of ra and dec as list like - astropy.units
        :param grb_time: optional datetime object of grb_time
        """
        self._name = kwargs.get("name")
        self._active_time = kwargs.get("active_time", None)
        self._bkg_time = kwargs.get("bkg_time", None)
        self._ra_icrs = kwargs.get("ra", None)
        self._dec_icrs = kwargs.get("dec", None)
        grb_time = kwargs.get("grb_time", None)
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
        ra_dec_units = kwargs.get("ra_dec_units", (u.deg, u.deg))

        if self._ra_icrs is None and self._dec_icrs is None:
            swift = pd.read_csv(
                pkg_resources.resource_filename("fierywhip", "data/Fermi_Swift.lis"),
                sep=" ",
                index_col=False,
                header=None,
            )
            logging.info(f"This is the stripped GRB name {self._name.strip('GRB')}")
            for j, i in swift.iterrows():
                name = str(i.loc[0])
                if len(name) == 8:
                    name = "0" + name
                if name == self._name.strip("GRB") or name == self._name.strip("bn"):
                    logging.info(f"Found a match!")
                    self._ra_icrs = str(i.loc[5])
                    self._dec_icrs = str(i.loc[6])
                    ra_dec_units = (u.hourangle, u.deg)
                    self._position = SkyCoord(
                        ra=self._ra_icrs,
                        dec=self._dec_icrs,
                        unit=ra_dec_units,
                        frame="icrs",
                    )
                    break
        else:
            self._position = SkyCoord(
                ra=self._ra_icrs, dec=self._dec_icrs, unit=ra_dec_units, frame="icrs"
            )
        self._ra_icrs = float(self._position.ra.deg)
        self._dec_icrs = float(self._position.dec.deg)
        self._get_trigdat_path()
        self._set_effective_area_correction(
            kwargs.get("custom_effective_area_dict", None)
        )
        run_det_sel = kwargs.get("run_det_sel", True)
        self._detector_selection = None
        if run_det_sel:
            try:
                self._get_detector_selection()
            except DetectorSelectionError:
                raise GRBInitError
            self.download_files()

    @property
    def position(self) -> SkyCoord:
        """
        :returns: SkyCoord of GRB
        """
        return self._position

    @property
    def grb_gbm_position(self) -> SkyCoord:
        """
        :returns: SkyCoord of GRB
        """
        return self._grb_gbm_position

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
        if not hasattr(self, "_detector_selection"):
            self._get_detector_selection()
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

    @property
    def long_grb(self):
        return self._long_grb

    def is_long_grb(self, is_it: bool):
        self._long_grb = is_it

    def download_files(self, dets="good_dets"):
        """
        Downloading TTE and CSPEC files from FTP
        """
        logging.info("Downloading TTE and CSPEC files")
        self.tte_files = {}
        self.cspec_files = {}
        if dets == "good_dets":
            for d in self.detector_selection.good_dets:
                self.tte_files[d] = download_tte_file(self._name, d)
                self.cspec_files[d] = download_cspec_file(self._name, d)
        elif dets == "all":
            for d in lu:
                self.tte_files[d] = download_tte_file(self._name, d)
                self.cspec_files[d] = download_cspec_file(self._name, d)

        elif isinstance(dets, list):
            for d in dets:
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
                download_trigdata_file(f"bn{self._name.strip('GRB')}".strip("\n"))
            except (TypeError, URLError):
                raise GRBInitError
        if os.path.exists(trigdat_path):
            self._trigdat = trigdat_path
        else:
            raise GRBInitError

    def _get_detector_selection(self, **kwargs):
        """
        :param max_sep: max separation of center from det in deg
        :param max_sep_normalization: max sep of center for det used as normalization
        """
        self._detector_selection = DetectorSelection(self, **kwargs)
        logging.debug(self._detector_selection.good_dets)

    def run_timeselection(self, **kwargs):
        """
        Timeselection for GRB using morogth auto_loc timeselection
        """
        if self._active_time is None:
            tsbb = TimeSelectionNew(
                name=self._name, trigdat_file=self._trigdat, fine=True, **kwargs
            )
            self._active_time = tsbb.active_time
            self._bkg_time = [tsbb.background_time_neg, tsbb.background_time_pos]

        start, stop = time_splitter(self._active_time)
        if stop - start > 10:
            self._long_grb = True
        else:
            self._long_grb = False

        pi = PositionInterpolator.from_trigdat(self._trigdat)
        self._grb_gbm_position = self._position.transform_to(
            GBMFrame(**pi.quaternion_dict(start))
        )

    def save_timeselection(self, path=None):
        if self._active_time is None:
            self.run_timeselection()
        if path is None:
            if not os.path.exists(
                os.path.join(os.environ.get("GBM_TRIGGER_DATA_DIR"), self.name)
            ):
                os.makedirs(
                    os.path.join(os.environ.get("GBM_TRIGGER_DATA_DIR"), self.name)
                )
            path = os.path.join(
                os.environ.get("GBM_TRIGGER_DATA_DIR"), self.name, "timeselection.yml"
            )

        bkg_neg_start, bkg_neg_stop = time_splitter(self._bkg_time[0])
        bkg_pos_start, bkg_pos_stop = time_splitter(self._bkg_time[1])
        active_time_start, active_time_stop = time_splitter(self._active_time)
        output_dict = {
            "active_time": {
                "start": active_time_start,
                "stop": active_time_stop,
            },
            "background_time": {
                "before": {"start": bkg_neg_start, "stop": bkg_neg_stop},
                "after": {"start": bkg_pos_start, "stop": bkg_pos_stop},
            },
            "max_time": bkg_pos_stop,
            "poly_order": -1,
            "fine": True,
        }
        with open(path, "w+") as f:
            yaml.safe_dump(output_dict, f)
        logging.info(f"Saved TS into path {path}")
        self.timeselection_path = path

    def timeselection_from_yaml(self, path):
        with open(path, "r") as f:
            ts = yaml.safe_load(f)
        at = ts["active_time"]
        self._active_time = f"{at['start']}-{at['stop']}"
        bkg_times = []
        bkg = ts["background_time"]
        for x in ["before", "after"]:
            bkg_times.append(f"{bkg[x]['start']}-{bkg[x]['stop']}")
        self._bkg_time = bkg_times

    def _set_effective_area_correction(self, eff_area_dict):
        """setter function for the effective area dict"""
        self._effective_area_dict = eff_area_dict
        if self._effective_area_dict is not None:
            self._eff_area = EffectiveAreaNormalization(self._effective_area_dict)
            logging.info(f"Set effective area to {self._effective_area_dict}")

    @property
    def effective_area(self):
        return self._eff_area

    def save_grb(self, path):
        export_dict = {}
        export_dict["name"] = str(self._name)
        export_dict["gbm_time"] = float(self.time_gbm.met)
        export_dict["ra"] = float(self._ra_icrs)
        export_dict["dec"] = float(self._dec_icrs)

        if self._effective_area_dict is not None:
            logging.debug("exported eff area")
            export_dict["eff_area_dict"] = self._effective_area_dict

        if self._active_time is not None:
            export_dict["active_time"] = str(self._active_time)
        if self._bkg_time is not None:
            export_dict["bkg_time"] = self._bkg_time
        export_dict["trigdat"] = str(self._trigdat)

        with open(path, "w+") as f:
            yaml.safe_dump(export_dict, f)
        return path

    @classmethod
    def grb_from_file(cls, path):
        with open(path, "r") as f:
            import_dict = yaml.safe_load(f)

        name = import_dict["name"]
        gbm_time = import_dict["gbm_time"]
        ra = import_dict["ra"]
        dec = import_dict["dec"]
        active_time = import_dict["active_time"]
        bkg_time = import_dict["bkg_time"]
        grb_time = GBMTime.from_MET(float(gbm_time)).time.datetime
        trigdat = import_dict["trigdat"]

        if "eff_area_dict" in import_dict.keys():
            logging.debug("import eff area dict")
            custom_effective_area_dict = import_dict["eff_area_dict"]
            out = cls(
                name=name,
                grb_time=grb_time,
                ra=ra,
                dec=dec,
                active_time=active_time,
                bkg_time=bkg_time,
                custom_effective_area_dict=custom_effective_area_dict,
                trigdat=trigdat,
            )
        else:
            out = cls(
                name=name,
                grb_time=grb_time,
                ra=ra,
                dec=dec,
                active_time=active_time,
                bkg_time=bkg_time,
                trigdat=trigdat,
            )

        return out


class GRBInitError(Exception):
    """
    Error in Initializing the GRB Object
    """

    def __init__(self, message=None):
        if message is None:
            message = "Failed to initialize GRB Object"
        super().__init__(message)
