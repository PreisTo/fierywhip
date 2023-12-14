#!/usr/bin/env python3

from morgoth.auto_loc.utils.fit import MultinestFitTrigdat
from gbm_drm_gen.io.balrog_like import BALROGLike
from gbm_drm_gen.io.balrog_drm import BALROG_DRM
from gbm_drm_gen.drmgen_trig import DRMGenTrig
from threeML.data_list import DataList
from fierywhip.normalizations.normalization_matrix import NormalizationMatrix
from fierywhip.utils.detector_utils import name_to_id, detector_list, nai_list
from fierywhip.frameworks.grbs import GRB
import yaml
import os
from time import time
from morgoth.utils.trig_reader import TrigReader


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

        super().__init__(
            grb_name, version, trigdat_file, bkg_fit_yaml_file, time_selection_yaml_file
        )

        if det_sel_mode != "default":
            if det_sel_mode == "max_sig_old":
                self._grb._get_detector_selection(
                    max_number_nai=5, min_number_nai=5, mode=det_sel_mode
                )
                self._normalizing_det = self._grb.detector_selection.normalizing_det
                self._use_dets = self._grb.detector_selection.good_dets
                print(f"\n\n USING DETS {self._use_dets}")
                with open(bkg_fit_yaml_file, "r") as f:
                    data = yaml.safe_load(f)
                with open(bkg_fit_yaml_file, "w") as f:
                    data["use_dets"] = list(map(name_to_id, self._use_dets))
                    yaml.safe_dump(data, f)
            elif det_sel_mode == "max_sig_and_lowest_old":
                self._grb._get_detector_selection(
                    max_number_nai=5, min_number_nai=5, mode=det_sel_mode
                )
                self._normalizing_det = (
                    self._grb.detector_selection.sorted_significances[0][0]
                )
                use_dets = []
                number_nais_high = 5
                number_nais_low = 1

                i = 0
                while number_nais_high > 0:
                    det = self._grb.detector_selection.sorted_significances[i][0]
                    if det in nai_list():
                        use_dets.append(det)
                        number_nais_high -= 1

                while number_nais_low > 0:
                    det = self._grb.detector_selection.sorted_significances[i][0]
                    if det in nai_list():
                        use_dets.append(det)
                        number_nais_low -= 1
                if (
                    self._grb.detector_selection.significances["b0"]
                    >= self._grb.detector_selection.significances["b1"]
                ):
                    use_dets.append("b0")
                else:
                    use_dets.append("b1")
                self._use_dets = self._grb.detector_selection.good_dets
                print(f"\n\n USING DETS {self._use_dets}\n\n")
                with open(bkg_fit_yaml_file, "r") as f:
                    data = yaml.safe_load(f)
                with open(bkg_fit_yaml_file, "w") as f:
                    data["use_dets"] = list(map(name_to_id, self._use_dets))
                    yaml.safe_dump(data, f)
            elif det_sel_mode == "max_sig":
                print("Using pre-set detectors from bkg yaml file")

            else:
                raise NotImplementedError("det_sel_mode not supported (yet)")
            self.setup_essentials()
        else:
            if self._use_eff_area:
                print(
                    "Currently doing this is absolutely useless and will likely worsen the results"
                )
                super().setup_essentials()
                # just use the first one as normalizing det
                self._normalizing_det = self._use_dets[0]
            else:
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
            except Exception as e:
                print(e)
                time.sleep(1)
                pass
            i += 1
            if i == 50:
                raise AssertionError("Can not restore background fit...")

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
