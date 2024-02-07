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
from urllib.error import URLError, HTTPError
from urllib.request import urlopen
import yaml
from mpi4py import MPI
from fierywhip.detectors.detectors import DetectorSelection, DetectorSelectionError
from fierywhip.normalizations.normalization_matrix import NormalizationMatrix
from fierywhip.timeselection.timeselection import TimeSelectionNew
from morgoth.auto_loc.time_selection import TimeSelectionBB
from fierywhip.timeselection.split_active_time import time_splitter
from fierywhip.config.configuration import fierywhip_config
import numpy as np
from threeML.utils.progress_bar import trange
import logging

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

    def __init__(
        self, check_finished=True, run_det_sel=True, testing=False, reverse=False
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
            # namesi, rasi, decsi, typesi = self._load_ipn_bursts()
            names_all = namess
            ras_all = rass
            decs_all = decss
            types_all = ["swift"] * len(decs_all)
            # for i, n in enumerate(namess):
            #    if n not in names_all:
            #        names_all.append(n)
            #        ras_all.append(rass[i])
            #        decs_all.append(decss[i])
            #        types_all.append("swift")
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
            self._check_already_run = None

        self._table = comm.bcast(self._table, root=0)
        self._check_already_run = comm.bcast(self._check_already_run, root=0)
        if fierywhip_config.grb_list.create_objects:
            self._create_grb_objects()

    def _create_grb_objects(self):
        grbs_temp = []
        if rank == 0:
            if type(self._testing) == bool:
                if self._testing:
                    stop_last = 20
                else:
                    stop_last = len(self._table)

            elif type(self._testing) == int:
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
            logging.info("Done loading Swift List")
        return names, ras, decs

    def _load_ipn_bursts(
        self, table_path=pkg_resources.resource_filename("fierywhip", "data/ipn.csv")
    ):
        names = []
        ras = []
        decs = []
        types = []
        if fierywhip_config.ipn.small:
            self._ipn_table = pd.read_csv(table_path, sep=" ", index_col=False)
            for i, b in self._ipn_table.iterrows():
                names.append(f"GRB{str(b['name']).strip('bn')}")
                ras.append(float(b["ra"]))
                decs.append(float(b["dec"]))
                types.append("ipn")
            logging.info("Done loading IPN list")
        return names, ras, decs, types

    def _load_ipn_arcs(
        self,
        table_path=pkg_resources.resource_filename(
            "fierywhip", "data/ipn_all_data.csv"
        ),
    ):
        table = pd.read_csv(table_path)
        names = []
        total_seconds = 24 * 3600
        year = table["YEAR"].astype(str)
        month = np.array([month_lu[i.strip(" ")[1:-1]] for i in table["MONTH"]])
        day = table["DAY"].astype(str)
        sod = table["SOD"].astype(int)
        sod = round(sod / total_seconds * 1000, 0)
        sod = sod.astype(int)
        for y, m, d, s in zip(year, month, day, sod):
            names.append(
                f"GRB{y[2:]}{str(m).zfill(2)}{str(d).zfill(2)}{str(s).zfill(3)}"
            )
        table["name"] = np.array(names)
        base_url = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/"
        folder_trigdat = "/current/glg_trigdat_all_bn"
        non_exist = []
        if rank == 0:
            check_per_rank = len(table) // size

        else:
            check_per_rank = None
        check_per_rank = comm.bcast(check_per_rank, root=0)
        start = rank * check_per_rank
        stop = (rank + 1) * check_per_rank
        if rank == size - 1:
            stop = len(table)
        versions_to_check = 3
        for i in trange(start, stop, 1):
            name = table.iloc[i]["name"]
            year = table.iloc[i]["YEAR"]
            url = f"{base_url}{year}/bn{name.strip('GRB')}{folder_trigdat}{name.strip('GRB')}_v0"
            exists = False
            for v in range(versions_to_check):
                url_version = f"{url}{v}.fit"
                try:
                    response = urlopen(url_version)
                    exists = True
                    break
                except HTTPError:
                    pass
            if not exists:
                non_exist.append(i)
        logging.debug(f"Rank {rank} finished")
        res = comm.gather(non_exist, root=0)
        if rank == 0:
            drop_list = []
            for r in res:
                drop_list.extend(r)

        else:
            drop_list = None
        drop_list = comm.bcast(drop_list, root=0)
        if rank == 0:
            logging.debug(f"Before removing: {len(table)}")
            table.drop(non_exist, inplace=True)
            logging.debug(f"After removing: {len(table)}")
        else:
            table = None
        table = comm.bcast(table, root=0)
        for el in table.iterrows():
            # get the relevant numbers

            ra1 = el["IPN RA1"]
            ra2 = el["IPN RA2"]

            dec1 = el["IPN DEC1"]
            dec2 = el["IPN DEC2"]

            r1 = el["IPN R1"]
            r2 = el["IPN R2"]

            dr1 = el["IPN DR1"]
            dr2 = el["IPN DR2"]

        names = []
        ras = []
        decs = []
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
            print(f"This is the stripped GRB name {self._name.strip('GRB')}")
            for j, i in swift.iterrows():
                name = str(i.loc[0])
                if len(name) == 8:
                    name = "0" + name
                if name == self._name.strip("GRB"):
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
        run_det_sel = kwargs.get("run_det_sel", True)
        if run_det_sel:
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

    @property
    def long_grb(self):
        return self._long_grb

    def is_long_grb(self, is_it: bool):
        self._long_grb = is_it

    def download_files(self):
        """
        Downloading TTE and CSPEC files from FTP
        """
        logging.info("Downloading TTE and CSPEC files")
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
        if fierywhip_config.timeselection.store_and_reload:
            flag = False
            if self._active_time is None and self._bkg_time is None:
                if os.path.exists(
                    os.path.join(
                        os.environ.get("GBMDATA"), "localizing/timeselections.yml"
                    )
                ):
                    with open(
                        os.path.join(
                            os.environ.get("GBMDATA"), "localizing/timeselections.yml"
                        ),
                        "r",
                    ) as f:
                        ts = yaml.safe_load(f)
                    if self._name in ts.keys():
                        flag = True
                if flag:
                    self._active_time = ts[self._name]["active_time"]
                    self._bkg_time = [
                        ts[self._name]["bkg_neg"],
                        ts[self._name]["bkg_pos"],
                    ]
                else:
                    try:
                        tsbb = TimeSelectionBB(
                            self._name, self._trigdat, fine=True, **kwargs
                        )
                        self._active_time = tsbb.active_time
                        self._bkg_time = [
                            tsbb.background_time_neg,
                            tsbb.background_time_pos,
                        ]
                    except Exception as e:
                        raise GRBInitError(str(e))
                if fierywhip_config.timeselection.save and flag is not True:
                    if os.path.exists(
                        os.path.join(
                            os.environ.get("GBMDATA"), "localizing/timeselections.yml"
                        )
                    ):
                        with open(
                            os.path.join(
                                os.environ.get("GBMDATA"),
                                "localizing/timeselections.yml",
                            ),
                            "r",
                        ) as f:
                            ts = yaml.safe_load(f)
                    else:
                        ts = {}
                    try:
                        os.makedirs(
                            os.path.join(os.environ.get("GBMDATA"), "localizing")
                        )
                    except FileExistsError:
                        pass
                    with open(
                        os.path.join(
                            os.environ.get("GBMDATA"), "localizing/timeselections.yml"
                        ),
                        "w+",
                    ) as f:
                        ts[self._name] = {
                            "active_time": self._active_time,
                            "bkg_neg": self._bkg_time[0],
                            "bkg_pos": self._bkg_time[1],
                        }
                        yaml.safe_dump(ts, f)
        else:
            if self._active_time is None:
                tsbb = TimeSelectionNew(
                    name=self._name, trigdat_file=self._trigdat, fine=True, **kwargs
                )
                self._active_time = tsbb.active_time
                self._bkg_time = [tsbb.background_time_neg, tsbb.background_time_pos]

        at = self._active_time.split("-")
        if len(at) == 2:
            start = float(at[0])
            stop = float(at[-1])
        elif len(at) == 3:
            start = -float(at[1])
            stop = float(at[-1])
        elif len(at) == 4:
            start = -float(at[1])
            stop = -float(at[-1])
        else:
            raise ValueError
        if stop - start > 10:
            self._long_grb = True
        else:
            self._long_grb = False

    def save_timeselection(self, path=None):
        if self._active_time is None:
            self.run_timeselection()    
        if path is None:
            path = os.path.join(
                os.environ.get("GBM_TRIGGER_DATA_DIR"), self.name, "timeselection.yml"
            )

        bkg_neg_start, bkg_neg_stop = time_splitter(self._bkg_time[0])
        bkg_pos_start, bkg_pos_stop = time_splitter(self._bkg_time[1])
        active_time_start,active_time_stop = time_splitter(self._active_time)
        output_dict = {
            "active_time": {
                "start": active_time_start,
                "stop": active_time_stop,
            },
            "background_time": {
                "before": {"start": bkg_neg_start, "stop": bkg_neg_stop},
                "after": {"start": bkg_pos_start, "stop": bkg_pos_stop},
            },
        }
        with open(path, "w+") as f:
            yaml.safe_dump(output_dict, f)
        logging.info(f"Saved TS into path {path}")
        self.timeselection_path = path
    def _get_effective_area_correction(self, nm):
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

    def save_grb(self, path):
        export_dict = {}
        export_dict["name"] = self._name
        export_dict["position"] = {}
        export_dict["position"]["ra"] = float(self._position.ra.deg)
        export_dict["position"]["dec"] = float(self._position.dec.deg)

        if self._active_time is not None:
            export_dict["time_selection"] = {}
            export_dict["time_selection"]["active_time"] = str(self._active_time)
        if self._bkg_time is not None:
            if "time_selection" not in export_dict.keys():
                export_dict["time_selection"] = {}
            export_dict["time_selection"]["background"] = self._bkg_time
        export_dict["trigdat"] = str(self._trigdat)
        export_dict["time"] = self._time.strftime("%y%m%d-%H:%M:%S.%f")

        with open(path, "w+") as f:
            yaml.safe_dump(export_dict, f)

    @classmethod
    def grb_from_file(cls, path):
        with open(path, "r") as f:
            restored = yaml.safe_load(f)
        name = restored["name"]
        grb_time = datetime.strptime(restored["time"], "%y%m%d-%H:%M:%S.%f")
        ra = restored["position"]["ra"]
        dec = restored["position"]["dec"]
        ra_dec_units = (u.deg, u.deg)
        return cls(
            name=name,
            ra=ra,
            dec=dec,
            ra_dec_units=ra_dec_units,
            grb_time=grb_time,
            run_det_sel=False,
        )


class GRBInitError(Exception):
    """
    Error in Initializing the GRB Object
    """

    def __init__(self, message=None):
        if message is None:
            message = "Failed to initialize GRB Object"
        super().__init__(message)
