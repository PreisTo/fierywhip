#!/usr/bin/env python3

from morgoth.auto_loc.utils.fit import MultinestFitTrigdat
from gbm_drm_gen.io.balrog_like import BALROGLike
from gbmgeometry.gbm import GBM
from gbmgeometry.position_interpolator import PositionInterpolator
from threeML.data_list import DataList
from fierywhip.utils.detector_utils import name2id
from fierywhip.frameworks.grbs import GRB
from fierywhip.utils.balrog_like import BALROGLikeMultiple
from fierywhip.timeselection.split_active_time import calculate_active_time_splits
from fierywhip.config.configuration import FierywhipConfig
import yaml
import os
from morgoth.utils.trig_reader import TrigReader
from mpi4py import MPI
import logging
from astromodels.functions import (
    Cutoff_powerlaw,
    Uniform_prior,
    Log_uniform_prior,
    Powerlaw,
    Band,
)
from astromodels.sources.point_source import PointSource
from astromodels.core.model import Model
from threeML.bayesian.bayesian_analysis import BayesianAnalysis
from threeML.io.plotting.post_process_data_plots import display_spectrum_model_counts

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

base_dir = os.environ.get("GBM_TRIGGER_DATA_DIR")


class MultinestFitTrigdatEffArea(MultinestFitTrigdat):
    """
    Adaption of Morgoth's MultinestFitTrigdat Class to deal with
    effective area correction and detector selection
    """

    def __init__(
        self,
        grb: GRB,
        grb_name: str,
        version: str,
        trigdat_file: str,
        bkg_fit_yaml_file: str,
        time_selection_yaml_file: str,
        use_eff_area: bool = False,
        det_sel_mode: str = "default",
        grb_file: str = None,
        config_path: str = None,
        **kwargs,
    ):
        if grb is not None:
            self._grb = grb
        elif grb_file is not None:
            self._grb = GRB.grb_from_file(grb_file)
        else:
            raise ValueError("need to provide either grb object or file to recreate")
        self._grb_name = grb_name
        self._version = version
        self._bkg_fit_yaml_file = bkg_fit_yaml_file
        self._time_selection_yaml_file = time_selection_yaml_file
        self._trigdat_file = trigdat_file
        if config_path is not None:
            self._fierywhip_config = FierywhipConfig.from_yaml(config_path)
        else:
            logging.warning("USING DEFAULT FIERYWHIP CONFIG!!!")
        self._use_eff_area = use_eff_area
        self._grb_name = grb_name
        self._custom_eff_area_area_dict = kwargs.get("custom_eff_area_dict", None)
        if self._use_eff_area and self._custom_eff_area_area_dict is not None:
            self._grb._set_effective_area_correction = self._custom_eff_area_area_dict
        self._spectrum_model = kwargs.get("spectrum", "cpl")
        if self._grb._detector_selection is None:
            logging.info("Detector Selection is None, running it")
            if det_sel_mode != "default":
                logging.debug(f"Using det_sel_mode {det_sel_mode}")
                if det_sel_mode == "max_sig_old":
                    self._grb._get_detector_selection(
                        max_number_nai=5, min_number_nai=5, mode=det_sel_mode
                    )
                    self._normalizing_det = self._grb.detector_selection.normalizing_det
                    self._use_dets = self._grb.detector_selection.good_dets
                    logging.debug(f"\n\n USING DETS {self._use_dets}")
                    if rank == 0:
                        with open(bkg_fit_yaml_file, "r") as f:
                            data = yaml.safe_load(f)
                            self._bkg_fit_files = data["bkg_fit_files"]
                        with open(bkg_fit_yaml_file, "w") as f:
                            data["use_dets"] = list(map(name2id, self._use_dets))
                            yaml.safe_dump(data, f)
                    else:
                        with open(bkg_fit_yaml_file, "r") as f:
                            data1 = yaml.safe_load(f)
                            self._bkg_fit_files = data1["bkg_fit_files"]

                elif det_sel_mode == "max_sig_and_lowest_old":
                    self._grb._get_detector_selection(
                        max_number_nai=6, min_number_nai=6, mode=det_sel_mode
                    )
                    self._normalizing_det = self._grb_.detector_selection.good_dets[0]
                    self._use_dets = self._grb.detector_selection.good_dets
                    logging.debug(f"\n\n USING DETS {self._use_dets}\n\n")
                    if rank == 0:
                        with open(bkg_fit_yaml_file, "r") as f:
                            data = yaml.safe_load(f)
                            self._bkg_fit_files = data["bkg_fit_files"]
                        with open(bkg_fit_yaml_file, "w") as f:
                            data["use_dets"] = list(map(name2id, self._use_dets))
                            yaml.safe_dump(data, f)
                    else:
                        with open(bkg_fit_yaml_file, "r") as f:
                            data1 = yaml.safe_load(f)
                            self._bkg_fit_files = data1["bkg_fit_files"]

                elif det_sel_mode == "max_sig_triplets":
                    self._grb._get_detector_selection(
                        max_number_nai=6, min_number_nai=6, mode=det_sel_mode
                    )
                    self._normalizing_det = self._grb.detector_selection.normalizing_det
                    self._use_dets = self._grb.detector_selection.good_dets
                    logging.debug(f"\n\n USING DETS {self._use_dets}")
                    if rank == 0:
                        with open(bkg_fit_yaml_file, "r") as f:
                            data = yaml.safe_load(f)
                            self._bkg_fit_files = data["bkg_fit_files"]
                        with open(bkg_fit_yaml_file, "w") as f:
                            data["use_dets"] = list(map(name2id, self._use_dets))
                            yaml.safe_dump(data, f)
                    else:
                        with open(bkg_fit_yaml_file, "r") as f:
                            data1 = yaml.safe_load(f)
                            self._bkg_fit_files = data1["bkg_fit_files"]

                elif det_sel_mode == "bgo_sides_no_bgo":
                    logging.debug("Using pre-set detectors from bkg yaml file")
                    self._grb._get_detector_selection(
                        max_number_nai=6,
                        min_number_nai=6,
                        mode=det_sel_mode,
                        bkg_yaml=bkg_fit_yaml_file,
                    )
                    with open(bkg_fit_yaml_file, "r") as f:
                        data = yaml.safe_load(f)
                        self._bkg_fit_files = data["bkg_fit_files"]
                    self._normalizing_det = self._grb.detector_selection.normalizing_det
                    self._use_dets = self._grb.detector_selection.good_dets
                elif det_sel_mode == "huntsville":
                    self._grb._get_detector_selection(
                        max_number_nai=6, min_number_nai=6, mode=det_sel_mode
                    )
                    self._normalizing_det = self._grb.detector_selection.normalizing_det
                    self._use_dets = self._grb.detector_selection.good_dets
                    logging.info("Set Dets according to huntsville simulation")
                    if rank == 0:
                        with open(bkg_fit_yaml_file, "r") as f:
                            data = yaml.safe_load(f)
                            self._bkg_fit_files = data["bkg_fit_files"]
                        with open(bkg_fit_yaml_file, "w") as f:
                            data["use_dets"] = list(map(name2id, self._use_dets))
                            yaml.safe_dump(data, f)
                    else:
                        with open(bkg_fit_yaml_file, "r") as f:
                            data1 = yaml.safe_load(f)
                            self._bkg_fit_files = data1["bkg_fit_files"]
                elif det_sel_mode == "all":
                    self._grb._get_detector_selection(mode=det_sel_mode)
                    with open(bkg_fit_yaml_file, "r") as f:
                        data = yaml.safe_load(f)
                        self._bkg_fit_files = data["bkg_fit_files"]

                    self._normalizing_det = self._grb.detector_selection.normalizing_det
                    self._use_dets = self._grb.detector_selection.good_dets
                    logging.info("Using all those beautiful scintillation dets")
                else:
                    raise NotImplementedError("det_sel_mode not supported (yet)")
                self.setup_essentials()
            else:
                if self._use_eff_area:
                    logging.error(
                        "Currently doing this is absolutely useless and will likely worsen the results"
                    )
                    super().setup_essentials()
                    # just use the first one as normalizing det
                    self._normalizing_det = self._use_dets[0]
                else:
                    with open(bkg_fit_yaml_file, "r") as f:
                        data = yaml.safe_load(f)
                        self._bkg_fit_files = data["bkg_fit_files"]
                    super().setup_essentials()
        else:
            logging.info("Detector selection was already run")

            self._use_dets = self._grb._detector_selection.good_dets
            self._normalizing_det = self._grb._detector_selection.normalizing_det

            assert isinstance(self._use_dets, list), "use dets is not list"
            if rank == 0:
                with open(bkg_fit_yaml_file, "r") as f:
                    data = yaml.safe_load(f)
                    self._bkg_fit_files = data["bkg_fit_files"]
                with open(bkg_fit_yaml_file, "w") as f:
                    data["use_dets"] = list(map(name2id, self._use_dets))
                    yaml.safe_dump(data, f)

            comm.Barrier()
            with open(bkg_fit_yaml_file, "r") as f:
                data1 = yaml.safe_load(f)
                self._bkg_fit_files = data1["bkg_fit_files"]
            self.setup_essentials()

        super().__init__(
            grb_name,
            version,
            trigdat_file,
            bkg_fit_yaml_file,
            time_selection_yaml_file,
            spectrum=self._spectrum_model,
        )

    def setup_essentials(self):
        with open(self._bkg_fit_yaml_file, "r") as f:
            data = yaml.safe_load(f)
            self._bkg_fit_yaml_file = data["bkg_fit_files"]

        with open(self._time_selection_yaml_file, "r") as f:
            data = yaml.safe_load(f)
            self._active_time = (
                f"{data['active_time']['start']}-{data['active_time']['stop']}"
            )
            self._fine = data["fine"]
        self._define_model(self._spectrum_model)
        self._setup_plugins()

    def _setup_plugins(self):
        """
        Set the plugins using the saved background hdf5 files
        :return:
        """
        success_restore = False
        i = 0
        while not success_restore:
            try:
                trig_reader = TrigReader(
                    self._trigdat_file,
                    fine=self._fine,
                    verbose=False,
                    restore_poly_fit=self._bkg_fit_files,
                )
                success_restore = True
                i = 0
            except Exception:
                import time

                time.sleep(1)
                pass
            i += 1
            if i == 50:
                raise AssertionError(
                    f"Can not restore background fit...\n{self._bkg_fit_files}"
                )

        trig_reader.set_active_time_interval(self._active_time)

        # trig_data = trig_reader.to_plugin(*self._use_dets)
        trig_data = []
        for d in self._use_dets:
            speclike = trig_reader.time_series[d].to_spectrumlike()
            time = 0.5 * (
                trig_reader.time_series[d].tstart + trig_reader.time_series[d].tstop
            )
            balrog_like = BALROGLike.from_spectrumlike(speclike, time=time)
            balrog_like.set_active_measurements("c1-c6")
            if self._use_eff_area:
                balrog_like.fix_eff_area_correction(
                    self._grb.effective_area.get_eac_for_det(d, self._normalizing_det)
                )
            trig_data.append(balrog_like)
        self._data_list = DataList(*trig_data)

    def _define_model(self, spectrum="cpl"):
        """
        Define a Model for the fit
        :param spectrum: Which spectrum type should be used (cpl, band, pl, sbpl or solar_flare)
        """
        # data_list=comm.bcast(data_list, root=0)
        if spectrum == "cpl":
            # we define the spectral model
            cpl = Cutoff_powerlaw()
            cpl.K.max_value = 10**4
            low, high = self._fierywhip_config.config.trigdat.cpl.k_prior_bounds
            cpl.K.prior = Log_uniform_prior(lower_bound=low, upper_bound=high)
            low, high = self._fierywhip_config.config.trigdat.cpl.xc_prior_bounds
            cpl.xc.prior = Log_uniform_prior(lower_bound=low, upper_bound=high)
            if self._fierywhip_config.config.trigdat.cpl.index_prior_bounds is not None:
                low, high = self._fierywhip_config.config.trigdat.cpl.index_prior_bounds
                logging.info(f"Setting index priors bounds to {low} - {high}")
                cpl.index.prior = Uniform_prior(lower_bound=low, upper_bound=high)
            else:
                cpl.index.set_uninformative_prior(Uniform_prior)
                logging.info("Settinf uninformative index prior")
            # we define a point source model using the spectrum we just specified
            if not self._fierywhip_config.config.trigdat.cpl.smart_ra_dec_init:
                logging.info("Setting initial position to 0,0")
                ra, dec = 0.0, 0.0
            else:
                highest_sig_det = self._grb.detector_selection.normalizing_det
                pi = PositionInterpolator.from_trigdat(self._grb.trigdat)
                gbm = GBM(quaternion=pi.quaternion(0), sc_pos=pi.sc_pos(0))
                hsd_center = (
                    gbm.detectors[highest_sig_det].get_center().transform_to("icrs")
                )
                ra = hsd_center.ra.deg
                dec = hsd_center.dec.deg
                logging.info(
                    f"Using smart ra dec: ra {round(ra,2)}\tdec {round(dec,2)}"
                )

            self._model = Model(PointSource("first", ra, dec, spectral_shape=cpl))

        elif spectrum == "band":
            band = Band()
            band.K.prior = Log_uniform_prior(lower_bound=1e-5, upper_bound=1200)
            band.alpha.set_uninformative_prior(Uniform_prior)
            band.xp.prior = Log_uniform_prior(lower_bound=10, upper_bound=1e4)
            band.beta.set_uninformative_prior(Uniform_prior)

            self._model = Model(PointSource("first", 0.0, 0.0, spectral_shape=band))

        elif spectrum == "pl":
            pl = Powerlaw()
            pl.K.max_value = 10**4
            pl.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=10**4)
            pl.index.set_uninformative_prior(Uniform_prior)
            # we define a point source model using the spectrum we just specified
            self._model = Model(PointSource("first", 0.0, 0.0, spectral_shape=pl))

        else:
            raise Exception("Use valid model type: cpl, pl, sbpl, band")

    def fit(self):
        """
        Fit the model to data using multinest
        :return:
        """

        # wrap for ra angle
        wrap = [0] * len(self._model.free_parameters)
        wrap[0] = 1
        self._bayes = BayesianAnalysis(self._model, self._data_list)

        # define temp chain save path
        self._temp_chains_dir = os.path.join(
            base_dir, self._grb_name, f"c_trig_{self._version}"
        )
        chain_path = os.path.join(self._temp_chains_dir, f"trigdat_{self._version}_")

        # Make temp chains folder if it does not exists already
        if not os.path.exists(self._temp_chains_dir):
            os.mkdir(os.path.join(self._temp_chains_dir))

        # use multinest to sample the posterior
        # set main_path+trigger to whatever you want to use

        self._bayes.set_sampler("multinest", share_spectrum=True)
        self._bayes.sampler.setup(
            n_live_points=800, chain_name=chain_path, wrapped_params=wrap, verbose=True
        )
        self._bayes.sample()
        if rank == 0:
            fig = self._bayes.results.corner_plot()
            fig.savefig(os.path.join(base_dir, self._grb_name, "cc_plots.png"))

            df = self._bayes.results.get_data_frame(error_type="hpd")
            df.to_csv(os.path.join(base_dir, self._grb_name, "parameters.csv"))


mapping = {"0": "first", "1": "second", "2": "third", "3": "fourth"}


class MultinestFitTrigdatMultipleSelections(MultinestFitTrigdatEffArea):
    def __init__(
        self,
        grb: GRB,
        grb_name: str,
        version: str,
        trigdat_file: str,
        bkg_fit_yaml_file: str,
        time_selection_yaml_file: str,
        use_eff_area: bool = False,
        det_sel_mode: str = "default",
        grb_file: str = None,
        config_path: str = None,
        **kwargs,
    ):
        super().__init__(
            grb=grb,
            grb_name=grb_name,
            version=version,
            trigdat_file=trigdat_file,
            bkg_fit_yaml_file=bkg_fit_yaml_file,
            time_selection_yaml_file=time_selection_yaml_file,
            use_eff_area=use_eff_area,
            det_sel_mode=det_sel_mode,
            grb_file=grb_file,
            config_path=config_path,
            **kwargs,
        )

    def setup_essentials(self):
        with open(self._bkg_fit_yaml_file, "r") as f:
            data = yaml.safe_load(f)
            self._bkg_fit_yaml_file = data["bkg_fit_files"]

        with open(self._time_selection_yaml_file, "r") as f:
            data = yaml.safe_load(f)
            self._active_time = (
                f"{data['active_time']['start']}-{data['active_time']['stop']}"
            )
            self._fine = data["fine"]

        self._active_times_float = calculate_active_time_splits(
            self._trigdat_file,
            self._active_time,
            self._bkg_fit_files,
            self._use_dets,
            grb=self._grb_name,
            max_nr_responses=1,
        )
        self._define_model(self._spectrum_model)
        self._setup_plugins()

    def _setup_plugins(self):
        """
        Set the plugins using the saved background hdf5 files
        :return:
        """
        success_restore = False
        i = 0
        while not success_restore:
            try:
                trig_reader = TrigReader(
                    self._trigdat_file,
                    fine=self._fine,
                    verbose=False,
                    restore_poly_fit=self._bkg_fit_files,
                )
                success_restore = True
                i = 0
            except Exception:
                import time

                time.sleep(1)
                pass
            i += 1
            if i == 50:
                raise AssertionError(
                    f"Can not restore background fit...\n{self._bkg_fit_files}"
                )
        logging.info(
            f"Duration of Burst is {self._active_times_float[-1]-self._active_times_float[0]}, we will use 2 responses for this"
        )

        trig_data = []
        for x in range(len(self._active_times_float) - 1):
            key = mapping[str(x)]
            trig_reader.set_active_time_interval(
                f"{self._active_times_float[x]}-{self._active_times_float[x+1]}"
            )
            for d in self._use_dets:
                speclike = trig_reader.time_series[d].to_spectrumlike()
                time = 0.5 * (
                    trig_reader.time_series[d].tstart + trig_reader.time_series[d].tstop
                )
                balrog_like = BALROGLikeMultiple.from_spectrumlike(
                    speclike, name=f"{d}_{key}", time=time
                )
                balrog_like.assign_to_source(key)
                balrog_like.set_active_measurements("c1-c6")
                if self._use_eff_area:
                    balrog_like.fix_eff_area_correction(
                        self._grb.effective_area_correction(d)
                    )
                trig_data.append(balrog_like)
            logging.info(f"Added model for {key}")

        self._data_list = DataList(*trig_data)
        # define bayes object with model and data_list
        self._bayes = BayesianAnalysis(self._model, self._data_list)

    def _define_model(self, spectrum="cpl"):
        """
        Define a Model for the fit
        :param spectrum: Which spectrum type should be used (cpl, band, pl, sbpl or solar_flare)
        """
        # data_list=comm.bcast(data_list, root=0)
        if spectrum == "cpl":
            # we define the spectral model
            #
            ps_list = []
            cpl1 = Cutoff_powerlaw()
            cpl1.K.max_value = 10**4
            cpl1.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=10**4)
            cpl1.xc.prior = Log_uniform_prior(lower_bound=10, upper_bound=1e4)
            cpl1.index.set_uninformative_prior(Uniform_prior)
            # we define a point source model using the spectrum we just specified
            ps1 = PointSource("first", ra=0.0, dec=0.0, spectral_shape=cpl1)
            ps_list.append(ps1)
            logging.getLogger().setLevel("INFO")
            logging.info(
                f"These are the splits: {self._active_times_float}, and this the length: {len(self._active_times_float)}"
            )
            if len(self._active_times_float) >= 3:
                cpl2 = Cutoff_powerlaw()
                cpl2.K.max_value = 10**4
                cpl2.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=10**4)
                cpl2.xc.prior = Log_uniform_prior(lower_bound=10, upper_bound=1e4)
                cpl2.index.set_uninformative_prior(Uniform_prior)
                # we define a point source model using the spectrum we just specified
                ps2 = PointSource("second", ra=0.0, dec=0.0, spectral_shape=cpl2)
                ps_list.append(ps2)
                logging.info("Added PS2")
            if len(self._active_times_float) >= 4:
                cpl3 = Cutoff_powerlaw()
                cpl3.K.max_value = 10**4
                cpl3.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=10**4)
                cpl3.xc.prior = Log_uniform_prior(lower_bound=10, upper_bound=1e4)
                cpl3.index.set_uninformative_prior(Uniform_prior)
                # we define a point source model using the spectrum we just specified
                ps3 = PointSource("third", ra=0.0, dec=0.0, spectral_shape=cpl3)
                ps_list.append(ps3)
                logging.info("Added PS3")
            if len(self._active_times_float) >= 5:
                cpl4 = Cutoff_powerlaw()
                cpl4.K.max_value = 10**4
                cpl4.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=10**4)
                cpl4.xc.prior = Log_uniform_prior(lower_bound=10, upper_bound=1e4)
                cpl4.index.set_uninformative_prior(Uniform_prior)
                # we define a point source model using the spectrum we just specified
                ps4 = PointSource("fourth", ra=0.0, dec=0.0, spectral_shape=cpl4)
                ps_list.append(ps4)
                logging.info("Added PS4")
            if len(self._active_times_float) > 5:
                logging.info(
                    f"Wrong number of splits!!! {len(self._active_times_float)}"
                )
                raise NotImplementedError
            if len(self._active_times_float) == 2:
                self._model = Model(ps1)
            elif len(self._active_times_float) == 3:
                self._model = Model(ps1, ps2)
            elif len(self._active_times_float) == 4:
                self._model = Model(ps1, ps2, ps3)
            elif len(self._active_times_float) == 5:
                self._model = Model(ps1, ps2, ps3, ps4)
            else:
                raise NotImplementedError
            if len(self._active_times_float) >= 3:
                self._model.link(
                    self._model.second.position.ra, self._model.first.position.ra
                )
                self._model.link(
                    self._model.second.position.dec, self._model.first.position.dec
                )
            if len(self._active_times_float) >= 4:
                self._model.link(
                    self._model.third.position.ra, self._model.first.position.ra
                )
                self._model.link(
                    self._model.third.position.dec, self._model.first.position.dec
                )
            if len(self._active_times_float) >= 5:
                self._model.link(
                    self._model.fourth.position.ra, self._model.first.position.ra
                )
                self._model.link(
                    self._model.fourth.position.dec, self._model.first.position.dec
                )
        else:
            raise Exception("Use valid model type: cpl, pl, sbpl, band or solar_flare")

    def fit(self):
        """
        Fit the model to data using multinest
        :return:
        """

        # wrap for ra angle
        wrap = [0] * len(self._model.free_parameters)
        wrap[0] = 1
        print(
            f"These are the free parameters which we will fit:\n{self._model.free_parameters}"
        )
        # define temp chain save path
        self._temp_chains_dir = os.path.join(
            base_dir, self._grb_name, f"c_trig_{self._version}"
        )
        chain_path = os.path.join(self._temp_chains_dir, f"trigdat_{self._version}_")

        # Make temp chains folder if it does not exists already
        if not os.path.exists(self._temp_chains_dir):
            os.mkdir(os.path.join(self._temp_chains_dir))

        # use multinest to sample the posterior
        # set main_path+trigger to whatever you want to use

        self._bayes.set_sampler("multinest", share_spectrum=True)
        self._bayes.sampler.setup(
            n_live_points=800, chain_name=chain_path, wrapped_params=wrap, verbose=True
        )
        self._bayes.sample()
        if rank == 0:
            fig = self._bayes.results.corner_plot()
            fig.savefig(os.path.join(base_dir, self._grb_name, "cc_plots.png"))
