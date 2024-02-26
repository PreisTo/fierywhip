#!/usr/bin/env python3

from fierywhip.model.model import GRBModel
from fierywhip.frameworks.grbs import GRB
from fierywhip.config.configuration import fierywhip_config
from astromodels.functions import Cutoff_powerlaw
from astromodels.sources.point_source import PointSource
from astromodels.core.model import Model
from astromodels.functions.priors import Log_uniform_prior, Uniform_prior, Cosine_Prior
from gbm_drm_gen.io.balrog_drm import BALROG_DRM
from threeML.data_list import DataList
from threeML.bayesian.bayesian_analysis import BayesianResults, BayesianAnalysis
from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike
import os
from mpi4py import MPI
import logging

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class GRBModelIndividualNorm(GRBModel):
    """Individual Norms/Amplitudes for the Spectra of the TTE Fits"""

    def __init__(self, grb: GRB):
        grb.download_files()
        super().__init__(grb, fix_position=False, save_lc=True)

    def _to_plugin(self):
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
                spectrum_like.assign_to_source(f"grb_{d}")
                spectrum_likes.append(spectrum_like)
            else:
                spectrum_like = self._timeseries[d].to_spectrumlike()
                spectrum_like.set_active_measurements("300-30000")
                spectrum_like.assign_to_source(f"grb_{d}")
                spectrum_likes.append(spectrum_like)
        balrog_likes = []
        logging.info(f"We are going to use {self.grb.detector_selection.good_dets}")
        for i, d in enumerate(self.grb.detector_selection.good_dets):
            logging.debug(spectrum_likes[i].name)

            response = BALROG_DRM(self._responses[d], 0.0, 0.0)
            spectrum_likes[i]._observed_spectrum._response = response
            spectrum_likes[i]._observed_spectrum._response.set_time(response_time)
            bl = DispersionSpectrumLike(
                spectrum_likes[i].name,
                spectrum_likes[i]._observed_spectrum,
                spectrum_likes[i]._background_spectrum,
                True,
            )
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
        ps_list = []
        dets = self.grb.detector_selection.good_dets
        for i, d in enumerate(dets):
            cpl = Cutoff_powerlaw()
            cpl.index_value = -1
            cpl.K.value = 10
            cpl.K.prior = Log_uniform_prior(lower_bound=1e-4, upper_bound=1e3)
            cpl.xc.value = 300
            ps = PointSource(f"grb_{d}", ra=0.0, dec=0.0, spectral_shape=cpl)
            ps_list.append(ps)
        self._model = Model(*ps_list)
        logging.debug(self._model.display())

        # Link the position parameters as well as the spectral ones, except the
        # amplitude/norm one
        exec(
            f"self._model.grb_{dets[0]}.position.ra.prior"
            + " = Uniform_prior(lower_bound=0,upper_bound=360)"
        )
        exec(f"self._model.grb_{dets[0]}.position.ra.fix=False")

        exec(
            f"self._model.grb_{dets[0]}.position.dec.prior"
            + " = Cosine_Prior(lower_bound=-90,upper_bound=90)"
        )
        exec(f"self._model.grb_{dets[0]}.position.dec.fix=False")
        exec(
            f"self._model.grb_{dets[0]}.spectrum.main.Cutoff_powerlaw.index.prior"
            + " = Uniform_prior(lower_bound=-8,upper_bound=8)"
        )
        exec(
            f"self._model.grb_{dets[0]}.spectrum.main.Cutoff_powerlaw.xc.prior"
            + " = Log_uniform_prior(lower_bound=10,upper_bound=1e4)"
        )

        for j, d in enumerate(dets[1:]):
            for p in [
                "position.ra",
                "position.dec",
                "spectrum.main.Cutoff_powerlaw.index",
                "spectrum.main.Cutoff_powerlaw.xc",
            ]:
                exec(
                    f"self._model.link(self._model.grb_{d}.{p}"
                    + f",self._model.grb_{dets[0]}.{p})"
                )

    def fit(self):
        logging.info("Starting the Fit")
        self._bayes = BayesianAnalysis(self._model, self._data_list)
        # wrap for ra angle
        wrap = [0] * len(self._model.free_parameters)
        wrap[0] = 1

        # define temp chain save path
        self._temp_chains_dir = self._base_dir
        chain_path = os.path.join(self._temp_chains_dir, "chain_")

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
        if rank == 0:
            self._results = self._bayes.results
            self._results.data_list = self._data_list
            self._results.write_to(os.path.join(self._temp_chains_dir, "results.fits"))
            fig = self._results.corner_plot()
            fig.savefig(os.path.join(self._temp_chains_dir, "corner.png"))
