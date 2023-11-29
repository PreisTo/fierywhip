#!/usr/bin/env python3

from morgoth.auto_loc.utils.fit import MultinestFitTrigdat
from gbm_drm_gen.io.balrog_like import BALROGLike
from gbm_drm_gen.io.balrog_drm import BALROG_DRM
from gbm_drm_gen.drmgen_trig import DRMGenTrig
from threeML.utils.data_list import DataList
from fierywhip.normalizations.normalization_matrix import NormalizationMatrix
from fierywhip.utils.detector_utils import name_to_id
from fierywhip.frameworks.grbs import GRB
import os


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
    ):
        self._grb = grb
        self._use_eff_area = use_eff_area
        if self._use_eff_area:
            self._grb._get_effective_area_correction(
                NormalizationMatrix(
                    os.path.join(os.environ.get("GBMDATA"), "localizing/results.yml")
                ).matrix
            )
        if det_sel_mode != "default":
            if det_sel_mode == "max_sig":
                self._grb._get_detector_selection(
                    max_number_nai=5, min_number_nai=5, mode=det_sel_mode
                )
            else:
                raise AssertionError("This detector selection mode is not supported")
        # TODO detetector selection needs to happen here and and safe to bkg_fit_yaml
        # set self._normalizing_det
        super().__init__(
            grb_name, version, trigdat_file, bkg_fit_yaml_file, time_selection_yaml_file
        )

    def _set_plugins(self):
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
            except:
                pass
            i += 1
            if i == 50:
                raise AssertionError("Can not restore background fit...")

        trig_reader.set_active_time_interval(self._active_time)

        # trig_data = trig_reader.to_plugin(*self._use_dets)
        trig_data = []
        for d in self._use_dets:
            speclike = trig_reader._time_series[d].to_spectrumlike()
            time = 0.5 * (
                trig_reader._time_series[det].tstart
                + trig_reader._time_series[det].tstop
            )
            balrog_like = BALROGLike.from_spectrumlike(speclike, time=time)
            balrog_like.set_active_measurements("c1-c6")
            if self._use_eff_area:
                balrog_like.fix_eff_area_correction(
                    self._grb.effective_area_correction(d)
                )
            trig_data.append(balrog_like)
        self._data_list = DataList(*trig_data)
