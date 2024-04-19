#!/usr/bin/env python3

from astromodels.functions import Powerlaw, Cutoff_powerlaw
from astromodels.sources.point_source import PointSource
from astromodels.functions.priors import Log_uniform_prior, Uniform_prior
from astromodels.core.model import Model
from threeML.data_list import DataList
from threeML.bayesian.bayesian_analysis import BayesianAnalysis, BayesianResults
from threeML.utils.data_builders.fermi.gbm_data import GBMTTEFile
from threeML.utils.data_builders.time_series_builder import TimeSeriesBuilder
from threeML.utils.time_series.event_list import EventListWithDeadTime
from threeML.utils.spectrum.binned_spectrum import BinnedSpectrumWithDispersion
from threeML.io.plotting.post_process_data_plots import (
    display_spectrum_model_counts,
)
import os
from gbm_drm_gen.io.balrog_like import BALROGLike
from fierywhip.model.utils.balrog_like import BALROGLikeMultiple
from gbm_drm_gen.io.balrog_drm import BALROG_DRM
from gbm_drm_gen.drmgen_tte import DRMGenTTE
from mpi4py import MPI
from fierywhip.config.configuration import fierywhip_config
from fierywhip.timeselection.split_active_time import calculate_active_time_splits
import logging

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


class GRBModel:
    """
    Class for modeling, setting up the data and fitting the GRB
    """

    def __init__(
        self,
        grb,
        model=None,
        base_dir=os.path.join(os.environ.get("GBMDATA"), "localizing"),
        fix_position=True,
        save_lc=False,
        use_eff_area=True,
        **kwargs,
    ):
        self._use_eff_area = use_eff_area
        self.grb = grb
        self._yaml_path = base_dir
        self._base_dir = os.path.join(base_dir, self.grb.name)
        if rank == 0:
            if not os.path.exists(self._base_dir):
                os.makedirs(self._base_dir)
        self._fix_position = fix_position
        self._save_lc = save_lc
        self._smart_ra_dec = kwargs.get("smart_ra_dec", True)
        if model is not None:
            self._model = model
        else:
            self._setup_model()

        # run timeselection for grb
        if self.grb.active_time is None:
            self.grb.run_timeselection()
        self.bkg_fitting()
        self._to_plugin()
        #self.fit()

    def bkg_fitting(self):
        temp_timeseries = {}
        temp_responses = {}
        for d in self.grb.detector_selection.good_dets:
            logging.info(f"Calculating Response for {d}")
            response = BALROG_DRM(
                DRMGenTTE(
                    tte_file=self.grb.tte_files[d],
                    trigdat=self.grb.trigdat,
                    mat_type=2,
                    cspecfile=self.grb.cspec_files[d],
                ),
                self.grb.position.ra.deg,
                self.grb.position.dec.deg,
            )
            tte_file = GBMTTEFile(self.grb.tte_files[d])
            event_list = EventListWithDeadTime(
                arrival_times=tte_file.arrival_times - tte_file.trigger_time,
                measurement=tte_file.energies,
                n_channels=tte_file.n_channels,
                start_time=tte_file.tstart - tte_file.trigger_time,
                stop_time=tte_file.tstop - tte_file.trigger_time,
                dead_time=tte_file.deadtime,
                first_channel=0,
                instrument=tte_file.det_name,
                mission=tte_file.mission,
                verbose=True,
            )
            ts = TimeSeriesBuilder(
                d,
                event_list,
                response=response,
                poly_order=-1,
                unbinned=False,
                verbose=True,
                container_type=BinnedSpectrumWithDispersion,
            )
            ts.set_background_interval(*self.grb.bkg_time)
            ts.set_active_time_interval(self.grb.active_time)
            temp_timeseries[d] = ts
            temp_responses[d] = response
        self._timeseries = temp_timeseries
        self._responses = temp_responses

    def _to_plugin(self):
        if self._fix_position:
            free_position = False
        else:
            free_position = True
        active_time = self.grb.active_time
        active_time = active_time.split("-")
        if len(active_time) == 2:
            start = float(active_time[0])
            stop = float(active_time[-1])
        if len(active_time) == 3:
            start = -float(active_time[1])
            stop = float(active_time[-1])
        elif len(active_time) == 4:
            start = -float(active_time[1])
            stop = -float(active_time[-1])
        assert start < stop, "start is after stop"
        response_time = (float(start) + float(stop)) / 2
        spectrum_likes = []
        for d in self.grb.detector_selection.good_dets:
            if self._timeseries[d]._name not in ("b0", "b1"):
                spectrum_like = self._timeseries[d].to_spectrumlike()
                spectrum_like.set_active_measurements("10-500")
                spectrum_likes.append(spectrum_like)
            else:
                spectrum_like = self._timeseries[d].to_spectrumlike()
                spectrum_like.set_active_measurements("300-30000")
                spectrum_likes.append(spectrum_like)
        balrog_likes = []
        logging.info(f"We are going to use {self.grb.detector_selection.good_dets}")
        for i, d in enumerate(self.grb.detector_selection.good_dets):
            if d not in ("b0", "b1"):
                bl = BALROGLike.from_spectrumlike(
                    spectrum_likes[i],
                    response_time,
                    self._responses[d],
                    free_position=free_position,
                )
                if self._use_eff_area:
                    if d not in (
                        "b0",
                        "b1",
                        self.grb.detector_selection.normalizing_det,
                    ):
                        bl.use_effective_area_correction(
                            min_value=fierywhip_config.config.eff_area_correction.eff_corr_lim_low,
                            max_value=fierywhip_config.config.eff_area_correction.eff_corr_lim_high,
                            use_gaussian_prior=fierywhip_config.config.eff_area_correction.eff_corr_gaussian,
                        )
                    else:
                        bl.fix_effective_area_correction(1)
                balrog_likes.append(bl)
        self._data_list = DataList(*balrog_likes)
        if self._save_lc:
            for d in self.grb.detector_selection.good_dets:
                fig = self._timeseries[d].view_lightcurve()
                plot_path = os.path.join(self._base_dir, self.grb.name, "lightcurves/")
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path)
                fig.savefig(os.path.join(plot_path, f"{d}.pdf"))

    def _setup_model(self):
        """
        setup the model using a cutoff powerlaw aka comptonized
        using values from 10.3847/1538-4357/abf24d and morgoth (github.com/grburgess/morgoth)
        """
        cpl = Cutoff_powerlaw()
        cpl.index.value = -1.1
        cpl.K.value = 1
        cpl.xc.value = 200
        cpl.index.prior = Uniform_prior(lower_bound=-2.5, upper_bound=1)
        cpl.K.prior = Log_uniform_prior(lower_bound=1e-4, upper_bound=1000)
        cpl.xc.prior = Log_uniform_prior(lower_bound=10, upper_bound=10000)
        if self._smart_ra_dec:
            self._model = Model(
                PointSource(
                    "GRB",
                    self.grb.position.ra.deg,
                    self.grb.position.dec.deg,
                    spectral_shape=cpl,
                )
            )
        else:
            self._model = Model(
                PointSource(
                    "GRB",
                    0,
                    0,
                    spectral_shape=cpl,
                )
            )

    def fit(self):
        logging.info("Starting the Fit")
        self._bayes = BayesianAnalysis(self._model, self._data_list)
        # wrap for ra angle
        wrap = [0] * len(self._model.free_parameters)
        wrap[0] = 1

        # define temp chain save path
        self._temp_chains_dir = self._base_dir
        chain_path = os.path.join(self._temp_chains_dir, f"chain_")

        # Make temp chains folder if it does not exists already
        if rank == 0:
            if not os.path.exists(self._temp_chains_dir):
                os.makedirs(os.path.join(self._temp_chains_dir))

        # use multinest to sample the posterior
        # set main_path+trigger to whatever you want to use

        self._bayes.set_sampler("multinest", share_spectrum=True)
        self._bayes.sampler.setup(
            n_live_points=fierywhip_config.config.live_points,
            chain_name=chain_path,
            wrapped_params=wrap,
            verbose=True,
        )
        self._bayes.sample()

        if rank == 0:
            self._results = self._bayes.results
            self._results.data_list = self._data_list
            self._results.write_to(
                os.path.join(self._base_dir, f"{self.grb.name}.fits"),overwrite = True
            )
            logging.info(f"Stored Fit result in {os.path.join(self._base_dir)}")

    @property
    def results(self):
        return self._results

    @property
    def bayes(self):
        return self._bayes


class GRBModelLong(GRBModel):
    """
    Use Multiple Responses for TTE Fit
    """

    def __init__(
        self,
        grb,
        model=None,
        base_dir=os.path.join(os.environ.get("GBMDATA"), "localizing"),
        fix_position=False,
        save_lc=False,
    ):
        super().__init__(grb, model, base_dir, fix_position, save_lc)

    def _to_plugin(self):
        if self._fix_position:
            free_position = False
        else:
            free_position = True
        active_time = self.grb.active_time
        bkg_times = self.grb.bkg_time
        splits = calculate_active_time_splits(
            self.grb.trigdat,
            active_time=active_time,
            use_dets=self.grb.detector_selection,
            grb=self.grb.name,
            bkg_time_intv=bkg_times,
        )
        self._source_names = ["first", "second", "third", "fourth"]
        logging.info(
            f"This is a long GRB, we will use these times to split the active time {splits}"
        )

        spectrum_likes = []
        balrog_likes = []
        for l in range(len(splits) - 1):
            response_time = splits[l] + (splits[l + 1] - splits[l]) / 2
            for d in self.grb.detector_selection.good_dets:
                if self._timeseries[d]._name not in ("b0", "b1"):
                    spectrum_like = self._timeseries[d].to_spectrumlike()
                    spectrum_like.set_active_measurements("40-700")
                else:
                    spectrum_like = self._timeseries[d].to_spectrumlike()
                    spectrum_like.set_active_measurements("300-30000")
                spectrum_likes.append(spectrum_like)
            logging.info(f"We are going to use {self.grb.detector_selection.good_dets}")

            for i, d in enumerate(self.grb.detector_selection.good_dets):

                bl = BALROGLike.from_spectrumlike(
                    spectrum_likes[i],
                    time=response_time,
                    free_position=free_position,
                )
                if d not in ("b0", "b1", self.grb.detector_selection.normalizing_det):
                    bl.use_effective_area_correction(
                        min_value=fierywhip_config.config.eff_area_correction.eff_corr_lim_low,
                        max_value=fierywhip_config.config.eff_area_correction.eff_corr_lim_high,
                        use_gaussian_prior=fierywhip_config.config.eff_area_correction.eff_corr_gaussian,
                    )
                else:
                    bl.fix_effective_area_correction(1)
                bl.assign_to_source(self._source_names[l])
                balrog_likes.append(bl)
        self._data_list = DataList(*balrog_likes)
        if self._save_lc:
            for d in self.grb.detector_selection.good_dets:
                fig = self._timeseries[d].view_lightcurve()
                plot_path = os.path.join(self._base_dir, self.grb.name, "lightcurves/")
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path)
                fig.savefig(os.path.join(plot_path, f"{d}.pdf"))

    def _setup_model(self):
        """
        setup the model using a cutoff powerlaw aka comptonized
        using values from 10.3847/1538-4357/abf24d and morgoth (github.com/grburgess/morgoth)
        """
        cpl1 = Cutoff_powerlaw()
        cpl1.index.value = -1.1
        cpl1.K.value = 1
        cpl1.xp.value = 200
        cpl1.index.prior = Uniform_prior(lower_bound=-2.5, upper_bound=1)
        cpl1.K.prior = Log_uniform_prior(lower_bound=1e-4, upper_bound=1000)
        cpl1.xp.prior = Uniform_prior(lower_bound=10, upper_bound=10000)
        ps1 = PointSource(
            "first",
            self.grb.position.ra.deg,
            self.grb.position.dec.deg,
            spectral_shape=cpl1,
        )

        self._model = Model()
