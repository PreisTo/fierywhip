#!/usr/bin/env python3
import warnings
from threeML import update_logging_level
from gbmgeometry.utils.gbm_time import GBMTime
import os
from datetime import datetime
import yaml
import pandas as pd
import morgoth
import pkg_resources
from morgoth.configuration import morgoth_config
from fierywhip.utils.result_reader import ResultReader
from morgoth.utils.env import get_env_value
from morgoth.utils.trig_reader import TrigReader
from morgoth.trigger import GBMTriggerFile
from morgoth.auto_loc.bkg_fit import BkgFittingTrigdat
from fierywhip.config.configuration import fierywhip_config
from fierywhip.frameworks.grbs import GRB
from fierywhip.utils.detector_utils import name2id
from fierywhip.timeselection.timeselection import TimeSelectionNew
from fierywhip.timeselection.split_active_time import time_splitter
from mpi4py import MPI
from astropy.coordinates import SkyCoord
import astropy.units as u
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
import subprocess
import numpy as np
import logging

update_logging_level("CRITICAL")
warnings.filterwarnings("ignore")
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
siez = comm.Get_size()

base_dir = get_env_value("GBM_TRIGGER_DATA_DIR")
result_csv = os.path.join(os.environ.get("GBM_TRIGGER_DATA_DIR"), "morgoth_results.csv")


class RunMorgoth:
    """
    Class that runs all relevant Morgoth Tasks
    Not compatible with different Detector selections
    """

    def __init__(self, grb: GRB = None, **kwargs):
        """
        Init the RunMorgoth Object
        :param grb: GRB object
        :type grb: fierywhip.frameworks.grbs.GRB
        :param **kwargs: spectrum, max_trigger_duration
        """
        self._grb = grb
        self._trigdat_path = self._grb.trigdat
        self._spectrum = kwargs.get("spectrum", "cpl")
        self._max_trigger_duration = kwargs.get("max_trigger_duration", 11)
        logging.info(f"Using spectrum {self._spectrum}")
        start_ts = datetime.now()
        self.timeselection()
        stop_ts = datetime.now()
        self._runtime_ts = float((stop_ts - start_ts).total_seconds())
        self.fit_background()

    def run_fit(self):
        """
        Runs the Fit on the RunMorgoth object and analyzes it afterwards
        """
        logging.info("Starting Fit")
        start_fit = datetime.now()
        run_fit = self.fit()
        stop_fit = datetime.now()
        if run_fit:
            self._runtime_fit = float((stop_fit - start_fit).total_seconds())
        else:
            self._runtime_fit = np.nan
        logging.info("Starting Analyzing")
        self.analyze()

    def timeselection(self):
        """
        Runs the Timeselection for the GRB
        """
        self._tsbb = TimeSelectionNew(
            name=self._grb.name,
            trigdat_file=self._trigdat_path,
            fine=True,
            min_trigger_duration=0.064,  # Thats one fine bin
            max_trigger_duration=self._max_trigger_duration,
            min_bkg_time=45,
        )
        logging.info("Done TimeSelectionNew")

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
        start, stop = time_splitter(self._tsbb._active_time)
        if stop - start > 10:
            self._long_grb = True
        else:
            self._long_grb = False
        self._grb._active_time = self._tsbb.active_time
        self._grb._bkg_time = [
            self._tsbb.background_time_neg,
            self._tsbb.background_time_pos,
        ]
        self._grb.is_long_grb(self._long_grb)

    def fit_background(self):
        """
        Fitting the Background using Morgoths BkgFittingTrigdat
        """
        self._bkg_yaml = os.path.join(base_dir, self._grb.name, "bkg_fit.yml")
        self._bkg = BkgFittingTrigdat(
            grb_name=self._grb.name,
            version="v00",
            trigdat_file=self._trigdat_path,
            time_selection_file_path=self._ts_yaml,
        )
        logging.info("Done BkgFittingTrigdat")
        self._bkg.save_lightcurves(
            os.path.join(base_dir, self._grb.name, "trigdat", "v00", "lc")
        )
        self._bkg.save_bkg_file(
            os.path.join(base_dir, self._grb.name, "trigdat", "v00", "bkg_files")
        )
        self._bkg.save_yaml(self._bkg_yaml)

    def fit(self):
        """
        The actual fit (needs the morgoth config)

        :returns: if the fit has actual been run
        """
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
        """
        Analyzes the result of the fit (well just reads the
        result to be honest) and saves them to the result_csv
        """
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
            grb=self._grb,
            post_equal_weights_file=post_equal_weights_path,
            results_file=result_file,
        )
        #
        result_path = os.path.join(base_job, f"trigdat_{version}_fit_result.yml")
        result_reader.save_result_yml(result_path)

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
            result_reader.balrog_1_sigma,
            result_reader.balrog_2_sigma,
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
    """
    Same as RunMorgoth but with some more configurability
    """

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
        """
        Runs the detector selection and saves them to the bkg_fit.yml
        """
        self._grb._get_detector_selection(
            min_number_nai=6,
            max_number_nai=6,
            mode=self._det_sel_mode,
            bkg_yaml=self._bkg_yaml,
        )
        with open(self._bkg_yaml, "r") as f:
            data = yaml.safe_load(f)
            data["use_dets"] = list(
                map(name2id, self._grb.detector_selection.good_dets)
            )
        with open(self._bkg_yaml, "w") as f:
            yaml.safe_dump(data, f)

    def fit(self):
        """
        Runs the fit script
        """
        ncores = str(int(morgoth_config["multinest"]["n_cores"]))
        path_to_python = morgoth_config["multinest"]["path_to_python"]

        fit_script_path = pkg_resources.resource_filename(
            "fierywhip", "model/utils/fit_morgoth_eff_area.py"
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
                f"/usr/bin/mpiexec -n {ncores} --bind-to core {path_to_python} {fit_script_path} {self._grb.name} v00 {self._trigdat_path} {self._bkg_yaml} {self._ts_yaml} {self._det_sel_mode} {self._use_eff_area} {grb_obj_path} {self._spectrum} {str(self._long_grb)}",
                shell=True,
                env=env,
                stdin=subprocess.PIPE,
            )
        return run_fit


# TODO fix Multinest Core Issue
