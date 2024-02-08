#!/usr/bin/env python3

from fierywhip.model.model import GRBModel
from fierywhip.frameworks.grbs import GRB
from astromodels.functions import Cutoff_powerlaw
from astromodels.sources.point_source import PointSource
from astromodels.core.model import Model
from astromodels.functions.priors import Log_uniform_prior, Uniform_prior
from gbm_drm_gen.io.balrog_like import BALROGLike
from threeML.data_list import DataList
import os


class GRBModelIndividualNorm(GRBModel):
    def __init__(self, grb: GRB):
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
        print(f"We are going to use {self.grb.detector_selection.good_dets}")
        for i, d in enumerate(self.grb.detector_selection.good_dets):
            bl = BALROGLike.from_spectrumlike(
                spectrum_likes[i],
                response_time,
                self._responses[d],
                free_position=True,
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
            cpl.xc.value = 300
            cpl.index.prior = Uniform_prior(lower_bound=-8, upper_bound=8)
            cpl.K.prior = Log_uniform_prior(lower_bound=1e-4, upper_bound=10e3)
            cpl.xc.prior = Log_uniform_prior(lower_bound=10, upper_bound=10e4)
            ps = PointSource(f"grb_{d}", ra=0.0, dec=0.0, spectral_shape=cpl)
            ps_list.append(ps)
        self._model = Model(*ps_list)
        print(self._model.display())
        for j, d in enumerate(dets[1:]):
            for p in [
                "position.ra",
                "position.dec",
                "spectrum.main.Cutoff_powerlaw.index",
                "spectrum.main.Cutoff_powerlaw.xc",
            ]:
                exec(
                    f"self._model.link(self._model.grb_{d}.{p},self._model.grb_{dets[0]}.{p})"
                )
