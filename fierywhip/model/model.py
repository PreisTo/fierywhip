#!/usr/bin/env python3

from astromodels.functions import Powerlaw, Cutoff_powerlaw, Band
from astromodels.sources.point_source import PointSource
from astromodels.functions.priors import Log_uniform_prior, Uniform_prior
from astromodels.core.model import Model
from threeML.data_list import DataList
from threeML.bayesian.bayesian_analysis import BayesianAnalysis, BayesianResults
from threeML.utils.data_builders.fermi.gbm_data import GBMTTEFile
from threeML.utils.data_builders.time_series_builder import TimeSeriesBuilder
from threeML.utils.time_series.event_list import EventListWithDeadTime
from threeML.utils.spectrum.binned_spectrum import BinnedSpectrumWithDispersion
from threeML.io.plotting.post_process_data_plots import display_spectrum_model_counts
from astropy.stats import bayesian_blocks
from threeML.plugins.OGIPLike import OGIPLike
from threeML import *
from threeML.minimizer.minimization import FitFailed
import os
from gbm_drm_gen.io.balrog_like import BALROGLike
from gbm_drm_gen.io.balrog_drm import BALROG_DRM
from gbm_drm_gen.drmgen_tte import DRMGenTTE
from mpi4py import MPI
import matplotlib.pyplot as plt
import yaml
import numpy as np
from fierywhip.detectors.detectors import lu
from fierywhip.config.configuration import fierywhip_config

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
    ):
        self.grb = grb
        self._yaml_path = base_dir
        self._base_dir = os.path.join(base_dir, self.grb.name)
        if not os.path.exists(self._base_dir):
            os.makedirs(self._base_dir)
        self._fix_position = fix_position
        self._save_lc = save_lc
        if model is not None:
            self._model = Model
        else:
            self._setup_model()

        # run timeselection for grb
        self.grb.run_timeselection()
        self.bkg_fitting()
        self._to_plugin()
        self.fit()

    # TODO BKG Fit and Timeseries
    def bkg_fitting(self):
        temp_timeseries = {}
        temp_responses = {}
        for d in self.grb.detector_selection.good_dets:
            print(f"Calculating Response for {d}")
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
                spectrum_like.set_active_measurements("40-700")
                spectrum_likes.append(spectrum_like)
            else:
                spectrum_like = self._timeseries[d].to_spectrumlike()
                spectrum_like.set_active_measurements("300-30000")
                spectrum_likes.append(spectrum_like)
        balrog_likes = []
        print(f"We are going to use {self.grb.detector_selection.good_dets}")
        for i, d in enumerate(self.grb.detector_selection.good_dets):
            bl = BALROGLike.from_spectrumlike(
                spectrum_likes[i],
                response_time,
                self._responses[d],
                free_position=free_position,
            )
            if d not in ("b0", "b1", self.grb.detector_selection.normalizing_det):
                bl.use_effective_area_correction(
                        fierywhip_config.eff_corr_lim_low,
                        fierywhip_config.eff_corr_lim_high,
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
        cpl = Cutoff_powerlaw_Ep()
        cpl.index.value = -1.1
        cpl.K.value = 1
        cpl.xp.value = 200
        cpl.index.prior = Uniform_prior(lower_bound=-2.5, upper_bound=1)
        cpl.K.prior = Log_uniform_prior(lower_bound=1e-4, upper_bound=1000)
        cpl.xp.prior = Uniform_prior(lower_bound=10, upper_bound=10000)

        self._model = Model(
            PointSource(
                "GRB",
                self.grb.position.ra.deg,
                self.grb.position.dec.deg,
                spectral_shape=cpl,
            )
        )

    def fit(self):
        print("Starting the Fit")
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
            n_live_points=fierywhip_config.live_points,
            chain_name=chain_path,
            wrapped_params=wrap,
            verbose=True,
        )
        self._bayes.sample()
        self._results = self._bayes.results
        self._results.data_list = self._data_list

    @property
    def results(self):
        return self._results

    @property
    def bayes(self):
        return self._bayes
