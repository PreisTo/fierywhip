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
from morgoth.utils.trig_reader import TrigReader
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


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
                print(f"\n\n USING DETS {self._use_dets}")
                if rank == 0:
                    with open(bkg_fit_yaml_file, "r") as f:
                        data = yaml.safe_load(f)
                        self._bkg_fit_files = data["bkg_fit_files"]
                    with open(bkg_fit_yaml_file, "w") as f:
                        data["use_dets"] = list(map(name_to_id, self._use_dets))
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
                print(f"\n\n USING DETS {self._use_dets}\n\n")
                if rank == 0:
                    with open(bkg_fit_yaml_file, "r") as f:
                        data = yaml.safe_load(f)
                        self._bkg_fit_files = data["bkg_fit_files"]
                    with open(bkg_fit_yaml_file, "w") as f:
                        data["use_dets"] = list(map(name_to_id, self._use_dets))
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
                print(f"\n\n USING DETS {self._use_dets}")
                if rank == 0:
                    with open(bkg_fit_yaml_file, "r") as f:
                        data = yaml.safe_load(f)
                        self._bkg_fit_files = data["bkg_fit_files"]
                    with open(bkg_fit_yaml_file, "w") as f:
                        data["use_dets"] = list(map(name_to_id, self._use_dets))
                        yaml.safe_dump(data, f)
                else:
                    with open(bkg_fit_yaml_file, "r") as f:
                        data1 = yaml.safe_load(f)
                        self._bkg_fit_files = data1["bkg_fit_files"]

            elif (
                det_sel_mode == "max_sig"
                or det_sel_mode == "max_sig_and_lowest"
                or det_sel_mode == "max_sig_triplets"
                or det_sel_mode == "bgo_sides_no_bgo"
            ):
                print("Using pre-set detectors from bkg yaml file")
                with open(bkg_fit_yaml_file, "r") as f:
                    data = yaml.safe_load(f)
                    self._bkg_fit_files = data["bkg_fit_files"]

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
                print(self._bkg_fit_files)
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
                print(e)
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
