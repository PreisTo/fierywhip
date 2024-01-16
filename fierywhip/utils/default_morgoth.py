#!/usr/bin/env python3

import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from threeML import update_logging_level

update_logging_level("CRITICAL")
from gbmgeometry.utils.gbm_time import GBMTime
import shutil
import os
from shutil import copyfile
from datetime import datetime
import yaml
import pandas as pd
import morgoth
import pkg_resources
from morgoth.configuration import morgoth_config
from morgoth.utils.result_reader import ResultReader, get_best_fit_with_errors
from morgoth.utils.env import get_env_value
from morgoth.utils import file_utils
from morgoth.utils.download_file import BackgroundDownload
from morgoth.utils.env import get_env_value
from morgoth.utils.trig_reader import TrigReader
from morgoth.trigger import GBMTriggerFile
from morgoth.auto_loc.time_selection import TimeSelectionBB, TimeSelectionKnown
from morgoth.auto_loc.bkg_fit import BkgFittingTrigdat
from morgoth.auto_loc.utils.fit import MultinestFitTrigdat
from fierywhip.config.configuration import fierywhip_config
from fierywhip.frameworks.grbs import GRB
from fierywhip.utils.eff_area_morgoth import MultinestFitTrigdatEffArea
from fierywhip.utils.detector_utils import name_to_id
from fierywhip.detectors.timeselection import TimeSelectionNew
from mpi4py import MPI
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.io.fits as fits
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from morgoth.utils.plot_utils import create_corner_all_plot
from datetime import datetime
import subprocess
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
siez = comm.Get_size()

base_dir = get_env_value("GBM_TRIGGER_DATA_DIR")
result_csv = os.path.join(os.environ.get("GBM_TRIGGER_DATA_DIR"), "morgoth_results.csv")


class RunMorgoth:
    def __init__(self, grb: GRB = None, **kwargs):
        self._grb = grb
        self._trigdat_path = self._grb.trigdat
        self._spectrum = kwargs.get("spectrum", "cpl")
        print(f"Using spectrum {self._spectrum}!")
        start_ts = datetime.now()
        run_ts = self.timeselection()
        stop_ts = datetime.now()
        if not run_ts:
            self._runtime_ts = float((stop_ts - start_ts).total_seconds())
        else:
            self._runtime_ts = np.nan
        self.fit_background()

    def run_fit(self):
        print("Starting Fit")
        start_fit = datetime.now()
        run_fit = self.fit()
        stop_fit = datetime.now()
        if run_fit:
            self._runtime_fit = float((stop_fit - start_fit).total_seconds())
        else:
            self._runtime_fit = np.nan
        print("Starting Analyzing")
        self.analyze()

    def timeselection(self):
        ts_available = False
        if os.path.exists(
            os.path.join(os.environ.get("GBMDATA"), "localizing/timeselections.yml")
        ):
            with open(
                os.path.join(
                    os.environ.get("GBMDATA"), "localizing/timeselections.yml"
                ),
                "r",
            ) as f:
                ts_dict = yaml.safe_load(f)
            if self._grb.name in ts_dict.keys():
                ts_available = True
        ts_available = False
        if ts_available:
            self._tsbb = TimeSelectionKnown(
                active_time=ts_dict[self._grb.name]["active_time"],
                background_time_neg=ts_dict[self._grb.name]["bkg_neg"],
                background_time_pos=ts_dict[self._grb.name]["bkg_pos"],
                max_time=float(ts_dict[self._grb.name]["bkg_pos"].split("-")[-1]),
                fine=True,
            )
            print("Done TimeSelectionKnown")
        else:
            self._tsbb = TimeSelectionNew(
                name=self._grb.name,
                trigdat_file=self._trigdat_path,
                fine=True,
                min_trigger_duration=0.128,
                min_bkg_time=45,
            )
            print("Done TimeSelectionNew")
            if os.path.exists(
                os.path.join(os.environ.get("GBMDATA"), "localizing/timeselections.yml")
            ):
                with open(
                    os.path.join(
                        os.environ.get("GBMDATA"), "localizing/timeselections.yml"
                    ),
                    "r",
                ) as f:
                    temp = yaml.safe_load(f)
            else:
                temp = {}
            temp[self._grb.name] = {}
            temp[self._grb.name]["active_time"] = self._tsbb.active_time
            temp[self._grb.name]["bkg_neg"] = self._tsbb.background_time_neg
            temp[self._grb.name]["bkg_pos"] = self._tsbb.background_time_pos
            if fierywhip_config.timeselection.save:
                with open(
                    os.path.join(
                        os.environ.get("GBMDATA"), "localizing/timeselections.yml"
                    ),
                    "w+",
                ) as f:
                    yaml.safe_dump(temp, f)

        try:
            os.makedirs(os.path.join(base_dir, self._grb.name))
        except FileExistsError:
            pass
        self._ts_yaml = os.path.join(base_dir, self._grb.name, "timeselection.yml")
        self._tsbb.save_yaml(self._ts_yaml)
        return ts_available

    def fit_background(self):
        self._bkg_yaml = os.path.join(base_dir, self._grb.name, "bkg_fit.yml")
        self._bkg = BkgFittingTrigdat(
            grb_name=self._grb.name,
            version="v00",
            trigdat_file=self._trigdat_path,
            time_selection_file_path=self._ts_yaml,
        )
        print("Done BkgFittingTrigdat")
        self._bkg.save_lightcurves(
            os.path.join(base_dir, self._grb.name, "trigdat", "v00", "lc")
        )
        self._bkg.save_bkg_file(
            os.path.join(base_dir, self._grb.name, "trigdat", "v00", "bkg_files")
        )
        self._bkg.save_yaml(self._bkg_yaml)

    def fit(self):
        ncores = str(int(morgoth_config["multinest"]["n_cores"]))
        path_to_python = morgoth_config["multinest"]["path_to_python"]

        fit_script_path = f"{morgoth.__file__[:-12]}/auto_loc/fit_script.py"

        env = os.environ
        if os.path.exists(
            os.path.join(
                base_dir,
                self._grb.name,
                "trigdat",
                "v00",
                "trigdat_v00_loc_results.fits",
            )
        ):
            run_fit = False
        else:
            run_fit = True
            p = subprocess.check_output(
                f"/usr/bin/mpiexec -n {ncores} --bind-to core {path_to_python} {fit_script_path} {self._grb.name} v00 {self._trigdat_path} {self._bkg_yaml} {self._ts_yaml} trigdat",
                shell=True,
                env=env,
                stdin=subprocess.PIPE,
            )
        return run_fit

    def analyze(
        self,
    ):
        version = "v00"
        result_file = os.path.join(
            base_dir,
            self._grb.name,
            "trigdat",
            "v00",
            "trigdat_v00_loc_results.fits",
        )

        base_job = os.path.join(base_dir, self._grb.name, "trigdat", version)
        post_equal_weights_path = os.path.join(
            base_job, "chains", f"trigdat_{version}_post_equal_weights.dat"
        )
        trig_reader = TrigReader(self._trigdat_path)
        i = 0
        flag = True
        while flag and i < 10:
            v = f"v0{i}"
            uri = f"https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/20{self._grb.name[3:5]}/bn{self._grb.name[3:]}/current/glg_trigdat_all_bn{self._grb.name[3:]}_{v}.fit"
            try:
                res = urlopen(uri, timeout=10)
                flag = False
            except (HTTPError, URLError):
                i += 1
                flag = True

        tf = GBMTriggerFile(
            None,
            GBMTime.from_MET(trig_reader._trigtime).utc.replace(" ", "T") + "Z",
            self._grb.name,
            None,
            None,
            None,
            uri,
            None,
            None,
            None,
            None,
        )
        tf_path = os.path.join(base_dir, self._grb.name, "grb_parameters.yml")
        tf.write(tf_path)
        result_reader = ResultReader(
            grb_name=self._grb.name,
            report_type="trigdat",
            version=version,
            trigger_file=tf_path,
            time_selection_file=self._ts_yaml,
            background_file=self._bkg_yaml,
            post_equal_weights_file=post_equal_weights_path,
            result_file=result_file,
            trigdat_file=self._trigdat_path,
        )
        #
        result_path = os.path.join(base_job, f"trigdat_{version}_fit_result.yml")
        result_reader.save_result_yml(result_path)

        create_corner_all_plot(
            post_equal_weights_path,
            model=self._spectrum,
            save_path=os.path.join(base_job, "plots", "all_corner_plot.png"),
        )
        template = [
            "grb",
            "ra",
            "ra_err",
            "dec",
            "dec_err",
            "balrog_1sigma",
            "balrog_2sigma",
            "grb_ra",
            "grb_dec",
            "separation",
            "runtime_fit",
            "runtime_ts",
        ]

        if os.path.exists(result_csv):
            result_df = pd.read_csv(result_csv, index_col=None)
            for k in template:
                if k not in result_df.columns:
                    vals = np.zeros(len(result_df)) + np.nan
                    result_df[k] = vals
        else:
            result_df = pd.DataFrame(columns=template)
        row = [
            self._grb.name,
            result_reader.ra[0],
            result_reader.ra[1],
            result_reader.dec[0],
            result_reader.dec[1],
            result_reader._balrog_one_sig_err_circle,
            result_reader._balrog_two_sig_err_circle,
            self._grb.position.ra.deg,
            self._grb.position.dec.deg,
            SkyCoord(
                ra=result_reader.ra[0],
                dec=result_reader.dec[0],
                unit=(u.deg, u.deg),
                frame="icrs",
            )
            .separation(self._grb.position)
            .deg,
            self._runtime_fit,
            self._runtime_ts,
        ]
        result_df.loc[len(result_df)] = row
        if os.path.exists(result_csv):
            os.rename(
                result_csv,
                os.path.join(os.environ.get("GBM_TRIGGER_DATA_DIR"), "backup.csv"),
            )
        result_df.to_csv(result_csv, index=False)


class RunEffAreaMorgoth(RunMorgoth):
    def __init__(
        self,
        grb: GRB = None,
        use_eff_area: bool = False,
        det_sel_mode: str = "default",
        **kwargs,
    ):
        assert isinstance(
            grb, GRB
        ), "grb needs to be of type fierywhip.frameworks.grbs.GRB"
        self._grb = grb
        self._use_eff_area = use_eff_area
        self._det_sel_mode = det_sel_mode
        super().__init__(grb, **kwargs)
        self.setup_use_dets()

    def setup_use_dets(self):
        if self._det_sel_mode != "default":
            if self._det_sel_mode == "bgo_sides_no_bgo":
                self._grb._get_detector_selection(
                    min_number_nai=6,
                    max_number_nai=6,
                    mode=self._det_sel_mode,
                    bkg_yaml=self._bkg_yaml,
                )
            with open(self._bkg_yaml, "r") as f:
                data = yaml.safe_load(f)
                data["use_dets"] = list(
                    map(name_to_id, self._grb.detector_selection.good_dets)
                )
            with open(self._bkg_yaml, "w") as f:
                yaml.safe_dump(data, f)

    def fit(self):
        ncores = str(int(morgoth_config["multinest"]["n_cores"]))
        path_to_python = morgoth_config["multinest"]["path_to_python"]

        fit_script_path = pkg_resources.resource_filename(
            "fierywhip", "utils/fit_morgoth_eff_area.py"
        )
        grb_obj_path = os.path.join(
            base_dir, self._grb.name, "trigdat", "v00", "grb_object.yml"
        )
        self._grb.save_grb(grb_obj_path)

        env = os.environ
        if os.path.exists(
            os.path.join(
                base_dir,
                self._grb.name,
                "trigdat",
                "v00",
                "trigdat_v00_loc_results.fits",
            )
        ):
            run_fit = False
        else:
            run_fit = True
            p = subprocess.check_output(
                f"/usr/bin/mpiexec -n {ncores} --bind-to core {path_to_python} {fit_script_path} {self._grb.name} v00 {self._trigdat_path} {self._bkg_yaml} {self._ts_yaml} {self._det_sel_mode} {self._use_eff_area} {grb_obj_path} {self._spectrum}",
                shell=True,
                env=env,
                stdin=subprocess.PIPE,
            )
        return run_fit


# TODO fix Multinest Core Issue
