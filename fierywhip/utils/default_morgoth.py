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
from fierywhip.data.grbs import GRB
from mpi4py import MPI
from astropy.coordinates import SkyCoord
import astropy.units as u

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
siez = comm.Get_size()

base_dir = get_env_value("GBM_TRIGGER_DATA_DIR")
result_csv = os.path.join(os.environ.get("GBM_TRIGGER_DATA_DIR"), "morgoth_results.csv")


class RunMorgoth:
    def __init__(self, grb: GRB = None):
        self._grb = grb
        self._trigdat_path = self._grb.trigdat
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
        if ts_available:
            self._tsbb = TimeSelectionKnown(
                active_time=ts_dict[self._grb.name]["active_time"],
                background_time_neg=ts_dict[self._grb.name]["bkg_neg"],
                background_time_pos=ts_dict[self._grb.name]["bkg_pos"],
                max_time=float(ts_dict[self._grb.name]["bkg_pos"].split("-")[-1]),
                fine=True,
            )
        else:
            self._tsbb = TimeSelectionBB(
                grb_name=self._grb.name, trigdat_file=self._trigdat_path, fine=True
            )
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
        self._bkg_yaml = os.path.join(base_dir, self._grb.name, "bkg_fit.yml")
        self._bkg = BkgFittingTrigdat(
            grb_name=self._grb.name,
            version="v00",
            trigdat_file=self._trigdat_path,
            time_selection_file_path=self._ts_yaml,
        )
        self._bkg.save_lightcurves(
            os.path.join(base_dir, self._grb.name, "trigdat", "v00", "lc")
        )
        self._bkg.save_bkg_file(
            os.path.join(base_dir, self._grb.name, "trigdat", "v00", "bkg_files")
        )
        self._bkg.save_yaml(self._bkg_yaml)
        self.fit()
        self.analyze()

    def fit(self):
        ncores = str(int(morgoth_config["multinest"]["n_cores"]))
        path_to_python = morgoth_config["multinest"]["path_to_python"]

        fit_script_path = f"{morgoth.__file__[:-12]}/auto_loc/fit_script.py"

        env = os.environ
        import subprocess

        p = subprocess.check_output(
            f"/usr/bin/mpiexec -n {ncores} --bind-to core {path_to_python} {fit_script_path} {self._grb.name} v00 {self._trigdat_path} {self._bkg_yaml} {self._ts_yaml} trigdat",
            shell=True,
            env=env,
            stdin=subprocess.PIPE,
        )
        # multinest_fit = MultinestFitTrigdat(
        #    self._grb.name, "v00", self._trigdat_path, self._bkg_yaml, self._ts_yaml
        # )
        # multinest_fit.fit()
        # mutlinest_fit.save_fit_result()
        # multinest_fit.create_spectrum_plot()
        # multinest_fit.move_chains_dir()

    def analyze(
        self,
    ):
        result_file = (
            f"{base_dir}/{self._grb.name}/trigdat/v00/trigdat_v00_loc_results.fits",
        )
        with fits.open(result_file) as f:
            values = f["ANALYSIS_RESULTS"].data["VALUE"]
            pos_error = f["ANALYSIS_RESULTS"].data["POSITIVE_ERROR"]
            neg_error = f["ANALYSIS_RESULTS"].data["NEGATIVE_ERROR"]

        self._ra = values[0]
        self._ra_pos_err = pos_error[0]
        self._ra_neg_err = neg_error[0]

        if np.absolute(self._ra_pos_err) > np.absolute(self._ra_neg_err):
            self._ra_err = np.absolute(self._ra_pos_err)
        else:
            self._ra_err = np.absolute(self._ra_neg_err)

        self._dec = values[1]
        self._dec_pos_err = pos_error[1]
        self._dec_neg_err = neg_error[1]

        if np.absolute(self._dec_pos_err) > np.absolute(self._dec_neg_err):
            self._dec_err = np.absolute(self._dec_pos_err)
        else:
            self._dec_err = np.absolute(self._dec_neg_err)

        if self.report_type == "trigdat":
            self._K = values[2]
            self._K_pos_err = pos_error[2]
            self._K_neg_err = neg_error[2]

            if np.absolute(self._K_pos_err) > np.absolute(self._K_neg_err):
                self._K_err = np.absolute(self._K_pos_err)
            else:
                self._K_err = np.absolute(self._K_neg_err)

            self._index = values[3]
            self._index_pos_err = pos_error[3]
            self._index_neg_err = neg_error[3]

            if np.absolute(self._index_pos_err) > np.absolute(self._index_neg_err):
                self._index_err = np.absolute(self._index_pos_err)
            else:
                self._index_err = np.absolute(self._index_neg_err)

            try:
                self._xc = values[4]
                self._xc_pos_err = pos_error[4]
                self._xc_neg_err = neg_error[4]
                if np.absolute(self._xc_pos_err) > np.absolute(self._xc_neg_err):
                    self._xc_err = np.absolute(self._xc_pos_err)
                else:
                    self._xc_err = np.absolute(self._xc_neg_err)
                self._model = "cpl"
            except:
                self._model = "pl"
        base_job = os.path.join(base_dir, self._grb.name, "trigdat", "v00")
        post_equal_weights_path = os.path.join(
            base_job, "chains", f"trigdat_v00_post_equal_weights.dat"
        )
        res = get_best_fit_with_errors(post_equal_weights_path, self._model)
        if os.path.exists(result_csv):
            result_df = pd.read_csv(result_csv)
        else:
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
            ]
            result_df = pd.DataFrame(columns=template)
        row = [
            self._grb.name,
            *res,
            self._grb.position.ra.deg,
            self._grb.position.dec.deg,
            SkyCoord(ra=res[0], dec=res[2], unit=(u.deg, u.deg), frame="icrs")
            .separation(self._grb.position)
            .deg,
        ]
        result_df.loc[len(result_df)] = row
        os.rename(
            result_csv,
            os.path.join(os.environ.get("GBM_TRIGGER_DATA_DIR"), "backup.csv"),
        )
        result_df.to_csv(result_csv)
