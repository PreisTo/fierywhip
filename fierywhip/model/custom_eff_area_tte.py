from fierywhip.model.model import GRBModel
from astromodels.functions import Cutoff_powerlaw, Cutoff_powerlaw_Ep
from astromodels.functions.priors import Uniform_prior, Log_uniform_prior
from astromodels.sources.point_source import PointSource
from astromodels.core.model import Model
import os
import logging
from gbm_drm_gen.io.balrog_like import BALROGLike
from threeML.data_list import DataList

class CustomEffAreaCorrections(GRBModel):

    def __init__(
        self,
        grb,
        fix_position=False,
        use_eff_area=True,
        base_dir=None,
        eff_area_dict=None,
        smart_ra_dec=False,
        fix_spectrum=None,
    ):
        self._fix_sectrum = fix_spectrum
        self._eff_area_dict = eff_area_dict
        self._fix_use_area = use_eff_area
        super().__init__(
            grb,
            fix_position=fix_position,
            base_dir=base_dir,
            smart_ra_dec=smart_ra_dec,
        )

    def _to_plugin(self):
        logging.info("ACTUALLY USING THE CORRECT ONE")
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
                if self._fix_use_area:
                    bl.fix_effective_area_correction(self._eff_area_dict[d])
                    logging.info(
                        f"Fixing eff area correction for {d} to be {self._eff_area_dict[d]}"
                    )

                else:
                    bl.fix_effective_area_correction(1)
                balrog_likes.append(bl)
            else:
                pass
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
        if self._fix_sectrum is None:
            cpl = Cutoff_powerlaw()
            cpl.index.value = -1.1
            cpl.K.value = 1
            cpl.xc.value = 200
            cpl.index.prior = Uniform_prior(lower_bound=-2.5, upper_bound=1)
            cpl.K.prior = Log_uniform_prior(lower_bound=1e-4, upper_bound=1000)
            cpl.xc.prior = Log_uniform_prior(lower_bound=10, upper_bound=10000)
        else:
            cpl = Cutoff_powerlaw_Ep()
            cpl.index.value = self._fix_sectrum["index"]
            cpl.index.free = False
            cpl.piv.value = 100
            cpl.K.value = self._fix_sectrum["K"]
            cpl.K.free = True
            cpl.K.prior = Log_uniform_prior(lower_bound=1e-4, upper_bound=1000)
            cpl.xp.value = self._fix_sectrum["xc"]
            cpl.xp.free = False

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
