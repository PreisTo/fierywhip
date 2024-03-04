#!/usr/bin/python

from morgoth.utils.trig_reader import TrigReader
from gbm_drm_gen.io.balrog_like import BALROGLike
from fierywhip.frameworks.grbs import GRB
from astromodels.functions import (
    Cutoff_powerlaw,
    Log_uniform_prior,
    Uniform_prior,
    Powerlaw,
)
from astromodels.core.model import Model
from astromodels.sources.point_source import PointSource
from threeML.data_list import DataList
from threeML.bayesian.bayesian_analysis import BayesianAnalysis
from threeML.io.plotting.post_process_data_plots import (
    display_spectrum_model_counts,
)
import os
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

class TrigdatModel:
    def __init__(self, grb: GRB, **kwargs):
        self._grb = grb
        self._base_dir = os.path.join(os.environ.get("GBMDATA"),"fixed_spectrum")
        if not os.path.exists(self._base_dir):
            os.makedirs(self._base_dir)
        self._fix_position = kwargs.get("fix_position", False)
        self._spectrum = kwargs.get("spectrum", "cpl")

    def _setup_essentials(self):
        self._grb.run_timeselection()
        self._trigreader = TrigReader(self._grb.trigdat, fine=True)
        self._trigreader.set_active_time_interval(self._grb.active_time)
        self._trigreader.set_background_selections(*self._grb.bkg_time)
        self._setup_model()

    def _setup_model(self):
        if self._spectrum == "cpl":
            print(f"Using a Cutoff Powerlaw")
            cpl = Cutoff_powerlaw()
            cpl.K.max_value = 10**4
            cpl.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=10**4)
            cpl.xc.prior = Log_uniform_prior(lower_bound=1, upper_bound=1e4)
            cpl.index.set_uninformative_prior(Uniform_prior)
            self._model = Model(
                PointSource(
                    "GRB",
                    self._grb.position.ra.deg,
                    self._grb.position.dec.deg,
                    spectral_shape=cpl,
                )
            )
        elif self._spectrum == "pl":
            print("Using a Powerlaw")
            pl = Powerlaw()
            pl.K.max_value = 10**4
            pl.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=10**4)
            pl.index.set_uninformative_prior(Uniform_prior)
            self._model = Model(
                PointSource(
                    "GRB",
                    self._grb.position.ra.deg,
                    self._grb.position.dec.deg,
                    spectral_shape=pl,
                )
            )
        else:
            raise NotImplementedError("Spectral Model not implemented")

    def _to_plugin(self):
        """

        convert the series to a BALROGLike plugin

        :param detectors: detectors to use
        :return:
        """

        data = []

        for det in self._grb.detector_selection.good_dets:
            # first create a DSL

            speclike = self._trigreader._time_series[det].to_spectrumlike()

            # then we convert to BL

            time = 0.5 * (
                self._trigreader._time_series[det].tstart
                + self._trigreader._time_series[det].tstop
            )

            balrog_like = BALROGLike.from_spectrumlike(speclike, time=time)

            balrog_like.set_active_measurements("c1-c6")
                balrog_like.fix_effective_area_correction(corr)
            else:
                balrog_like.use_effective_area_correction(0.5, 1.5)
            data.append(balrog_like)
        self._data_list = DataList(*data)
    def fit(self):
        """
        Fit the model to data using multinest
        :return:
        """

        # define bayes object with model and data_list
        self._bayes = BayesianAnalysis(self._model, self._data_list)
        # wrap for ra angle
        wrap = [0] * len(self._model.free_parameters)
        wrap[0] = 1

        # define temp chain save path
        self._temp_chains_dir = os.path.join(self._base_dir, self._grb_name)
        chain_path = os.path.join(self._temp_chains_dir, f"chain_")

        # Make temp chains folder if it does not exists already
        if rank == 0:
            if not os.path.exists(self._temp_chains_dir):
                os.makedirs(os.path.join(self._temp_chains_dir))

        # use multinest to sample the posterior
        # set main_path+trigger to whatever you want to use

        self._bayes.set_sampler("multinest", share_spectrum=True)
        self._bayes.sampler.setup(
            n_live_points=fierywhip_config.live_points_trigdat,
            chain_name=chain_path,
            wrapped_params=wrap,
            verbose=True,
        )
        comm.Barrier()
        self._bayes.sample()

    def export_results(self):
        if rank == 0:
            self.results = self._bayes.results
            self.results.data_list = self._data_list
            self.results.write_to(
                os.path.join(self._temp_chains_dir, "result.fits"), overwrite=True
            )
            fig = self.results.corner_plot()
            fig.savefig(os.path.join(self._temp_chains_dir, "cplot.pdf"))
            plt.close("all")
            spectrum_plot = display_spectrum_model_counts(self.results)
            ca = spectrum_plot.get_axes()[0]
            y_lims = ca.get_ylim()
            if y_lims[0] < 10e-6:
                ca.set_ylim(bottom=10e-6)
            if y_lims[1] > 10e4:
                ca.set_ylim(top=10e4)
            spectrum_plot.tight_layout()
            spectrum_plot.savefig(
                os.path.join(self._temp_chains_dir, "splot.pdf"),
                bbox_inches="tight",
            )
            plt.close("all")
