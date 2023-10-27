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
    ):
        self.grb = grb
        self._base_dir = os.path.join(base_dir, self.grb.name)
        if not os.path.exists(self._base_dir):
            os.makedirs(self._base_dir)
        self._fix_position = fix_position
        if model is not None:
            self._model = Model
        else:
            self._setup_model()

        # run timeselection for grb
        self.grb.run_timeselection()

    def _to_plugin(self):
        if self._fix_position:
            free_position = False
        else:
            free_position = True

        response_time
        spectrum_likes = []
        for d in self._use_dets:
            if self._timeseries[d]._name not in ("b0", "b1"):
                spectrum_like = self._timeseries[d].to_spectrumlike()
                spectrum_like.set_active_measurements(self.energy_range)
                spectrum_likes.append(spectrum_like)
            else:
                spectrum_like = self._timeseries[d].to_spectrumlike()
                spectrum_like.set_active_measurements("300-30000")
                spectrum_likes.append(spectrum_like)
        balrog_likes = []
        print(f"We are going to use {self._use_dets}")
        for i, d in enumerate(self._use_dets):
            if free_position:
                bl = BALROGLikePositionPrior.from_spectrumlike(
                    spectrum_likes[i],
                    response_time,
                    self._responses[d],
                    free_position=free_position,
                    swift_position=self.grb_position,
                )
            else:
                bl = BALROGLike.from_spectrumlike(
                    spectrum_likes[i],
                    response_time,
                    self._responses[d],
                    free_position=free_position,
                )
            if fix_correction is None:
                if d not in ("b0", "b1", "n0", "n6"):
                    bl.use_effective_area_correction(0.5, 1.5)
                else:
                    bl.fix_effective_area_correction(1)
            else:
                raise NotImplementedError
            balrog_likes.append(bl)
        self._data_list = DataList(*balrog_likes)

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
        self._temp_chains_dir = os.path.join(
            self._base_dir, self.grb, self.energy_range, "TTE_fit"
        )
        chain_path = os.path.join(self._temp_chains_dir, f"chain")

        # Make temp chains folder if it does not exists already
        if rank == 0:
            if not os.path.exists(self._temp_chains_dir):
                os.makedirs(os.path.join(self._temp_chains_dir))

        # use multinest to sample the posterior
        # set main_path+trigger to whatever you want to use

        self._bayes.set_sampler("multinest", share_spectrum=True)
        self._bayes.sampler.setup(
            n_live_points=800,
            chain_name=chain_path,
            wrapped_params=wrap,
            verbose=True,
        )
        self._bayes.sample()
        self.results = self._bayes.results
        self._results_loaded = True
        if rank == 0:
            fig = self.results.corner_plot()
            fig.savefig(os.path.join(self._temp_chains_dir, "cplot.pdf"))
            plt.close("all")
        if rank == 0:
            try:
                spectrum_plot = display_spectrum_model_counts(self.results)
                ca = spectrum_plot.get_axes()[0]
                y_lims = ca.get_ylim()
                if y_lims[0] < 10e-6:
                    # y_lims_new = [10e-6, y_lims[1]]
                    ca.set_ylim(bottom=10e-6)
                spectrum_plot.tight_layout()
                spectrum_plot.savefig(
                    os.path.join(self._temp_chains_dir, "splot.pdf"),
                    bbox_inches="tight",
                )

            except:
                self.results.data_list = self._data_list
                spectrum_plot = display_spectrum_model_counts(self.results)
                ca = spectrum_plot.get_axes()[0]
                y_lims = ca.get_ylim()
                if y_lims[0] < 10e-6:
                    # y_lims_new = [10e-6, y_lims[1]]
                    ca.set_ylim(bottom=10e-6)

                spectrum_plot.tight_layout()
                spectrum_plot.savefig(
                    os.path.join(self._temp_chains_dir, "splot.pdf"),
                    bbox_inches="tight",
                )

                print("No spectral plot possible...")

            plt.close("all")
