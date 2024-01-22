#!/usr/bin/env python3

from morgoth.auto_loc.utils.fit import MultinestFitTrigdat
from gbm_drm_gen.io.balrog_like import BALROGLike
from gbm_drm_gen.io.balrog_drm import BALROG_DRM
from gbm_drm_gen.drmgen_trig import DRMGenTrig
from threeML.data_list import DataList
from fierywhip.normalizations.normalization_matrix import NormalizationMatrix
from fierywhip.utils.detector_utils import name2id, detector_list, nai_list
from fierywhip.frameworks.grbs import GRB
from fierywhip.model.utils.balrog_like import BALROGLikeMultiple
from fierywhip.timeselection.split_active_time import calculate_active_time_splits
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
        **kwargs,
    ):
        if grb is not None:
            self._grb = grb
        elif grb_file is not None:
            self._grb = GRB.grb_from_file(grb_file)
        else:
            raise ValueError("need to provide either grb object or file to recreate")
        self._version = version
        self._bkg_fit_yaml_file = bkg_fit_yaml_file
        self._time_selection_yaml_file = time_selection_yaml_file
        self._trigdat_file = trigdat_file

        self._use_eff_area = use_eff_area
        if self._use_eff_area:
            self._grb._get_effective_area_correction(
                NormalizationMatrix(
                    os.path.join(os.environ.get("GBMDATA"), "localizing/results.yml")
                ).matrix
            )
        self._spectrum_model = kwargs.get("spectrum", "cpl")
        super().__init__(
            grb_name,
            version,
            trigdat_file,
            bkg_fit_yaml_file,
            time_selection_yaml_file,
            spectrum=self._spectrum_model,
        )

        if det_sel_mode != "default":
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
            except Exception as e:
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
                    self._grb.effective_area_correction(d)
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
            cpl.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=10**4)
            cpl.xc.prior = Log_uniform_prior(lower_bound=1, upper_bound=1e4)
            cpl.index.set_uninformative_prior(Uniform_prior)
            # we define a point source model using the spectrum we just specified
            self._model = Model(PointSource("first", 0.0, 0.0, spectral_shape=cpl))

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

        elif spectrum == "sbpl":
            sbpl = SmoothlyBrokenPowerLaw()
            sbpl.K.min_value = 1e-5
            sbpl.K.max_value = 1e4
            sbpl.K.prior = Log_uniform_prior(lower_bound=1e-5, upper_bound=1e4)
            sbpl.alpha.set_uninformative_prior(Uniform_prior)
            sbpl.beta.set_uninformative_prior(Uniform_prior)
            sbpl.break_energy.min_value = 1
            sbpl.break_energy.prior = Log_uniform_prior(lower_bound=1, upper_bound=1e4)
            self._model = Model(PointSource("first", 0.0, 0.0, spectral_shape=sbpl))
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
        **kwargs,
    ):
        super().__init__(
            grb,
            grb_name,
            version,
            trigdat_file,
            bkg_fit_yaml_file,
            time_selection_yaml_file,
            use_eff_area,
            det_sel_mode,
            grb_file,
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
            self._trigdat_file, self._active_time, self._bkg_fit_files, self._use_dets
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
            except Exception as e:
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
        for l in range(len(self._active_times_float) - 1):
            key = mapping[str(l)]
            trig_reader.set_active_time_interval(
                f"{self._active_times_float[l]}-{self._active_times_float[l+1]}"
            )
            trig_data = []
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
            cpl1 = Cutoff_powerlaw()
            cpl1.K.max_value = 10**4
            cpl1.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=10**4)
            cpl1.xc.prior = Log_uniform_prior(lower_bound=1, upper_bound=1e4)
            cpl1.index.set_uninformative_prior(Uniform_prior)
            # we define a point source model using the spectrum we just specified
            ps1 = PointSource("first", ra=0.0, dec=0.0, spectral_shape=cpl1)
            cpl2 = Cutoff_powerlaw()
            cpl2.K.max_value = 10**4
            cpl2.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=10**4)
            cpl2.xc.prior = Log_uniform_prior(lower_bound=1, upper_bound=1e4)
            cpl2.index.set_uninformative_prior(Uniform_prior)
            # we define a point source model using the spectrum we just specified
            ps2 = PointSource("second", ra=0.0, dec=0.0, spectral_shape=cpl2)
            self._model = Model(ps1, ps2)
            self._model.link(
                self._model.second.position.ra, self._model.first.position.ra
            )
            self._model.link(
                self._model.second.position.dec, self._model.first.position.dec
            )
        # elif spectrum == "band":
        #     band = Band()
        #     band.K.prior = Log_uniform_prior(lower_bound=1e-5, upper_bound=1200)
        #     band.alpha.set_uninformative_prior(Uniform_prior)
        #     band.xp.prior = Log_uniform_prior(lower_bound=10, upper_bound=1e4)
        #     band.beta.set_uninformative_prior(Uniform_prior)

        #     self._model = Model(PointSource("GRB_band", 0.0, 0.0, spectral_shape=band))

        # elif spectrum == "pl":
        #     pl = Powerlaw()
        #     pl.K.max_value = 10**4
        #     pl.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=10**4)
        #     pl.index.set_uninformative_prior(Uniform_prior)
        #     # we define a point source model using the spectrum we just specified
        #     self._model = Model(PointSource("GRB_pl", 0.0, 0.0, spectral_shape=pl))

        # elif spectrum == "sbpl":
        #     sbpl = SmoothlyBrokenPowerLaw()
        #     sbpl.K.min_value = 1e-5
        #     sbpl.K.max_value = 1e4
        #     sbpl.K.prior = Log_uniform_prior(lower_bound=1e-5, upper_bound=1e4)
        #     sbpl.alpha.set_uninformative_prior(Uniform_prior)
        #     sbpl.beta.set_uninformative_prior(Uniform_prior)
        #     sbpl.break_energy.min_value = 1
        #     sbpl.break_energy.prior = Log_uniform_prior(lower_bound=1, upper_bound=1e4)
        #     self._model = Model(PointSource("GRB_sbpl", 0.0, 0.0, spectral_shape=sbpl))

        # elif spectrum == "solar_flare":
        #     # broken powerlaw
        #     bpl = Broken_powerlaw()
        #     bpl.K.max_value = 10**5
        #     bpl.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=10**5)
        #     bpl.xb.prior = Log_uniform_prior(lower_bound=1, upper_bound=1e4)
        #     bpl.alpha.set_uninformative_prior(Uniform_prior)
        #     bpl.beta.set_uninformative_prior(Uniform_prior)

        #     # thermal brems
        #     tb = Thermal_bremsstrahlung_optical_thin()
        #     tb.K.max_value = 1e5
        #     tb.K.min_value = 1e-5
        #     tb.K.prior = Log_uniform_prior(lower_bound=1e-5, upper_bound=10**5)
        #     tb.kT.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=1e4)
        #     tb.Epiv.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=1e4)

        #     # combined
        #     total = bpl + tb

        #     self._model = Model(
        #         PointSource("Solar_flare", 0.0, 0.0, spectral_shape=total)
        #     )
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
