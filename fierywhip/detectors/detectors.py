#!/usr/bin/python
from gbmgeometry.gbm import GBM
from gbmgeometry.position_interpolator import PositionInterpolator
from gbmgeometry.gbm_frame import GBMFrame
import astropy.units as u
import numpy as np
from fierywhip.config.configuration import fierywhip_config
from fierywhip.utils.detector_utils import id2name
from morgoth.utils.trig_reader import TrigReader
from morgoth.auto_loc.bkg_fit import BkgFittingTrigdat
import pandas as pd
import pkg_resources
import yaml
import logging

lu = [
    "n0",
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
    "n7",
    "n8",
    "n9",
    "na",
    "nb",
    "b0",
    "b1",
]
triplets = {
    "n0": ["n0", "n1", "n2"],
    "n1": ["n0", "n1", "n2"],
    "n2": ["n0", "n1", "n2"],
    "n3": ["n3", "n4", "n5"],
    "n4": ["n3", "n4", "n5"],
    "n5": ["n3", "n4", "n5"],
    "n6": ["n6", "n7", "n8"],
    "n7": ["n6", "n7", "n8"],
    "n8": ["n6", "n7", "n8"],
    "n9": ["n9", "na", "nb"],
    "na": ["n9", "na", "nb"],
    "nb": ["n9", "na", "nb"],
}


class DetectorSelection:
    """
    Class used for selecting detectors
    """

    def __init__(
        self,
        grb,
        max_sep=fierywhip_config.max_sep,
        max_sep_normalizing=fierywhip_config.max_sep_norm,
        min_number_nai=fierywhip_config.min_number_det,
        max_number_nai=fierywhip_config.max_number_det,
        mode=fierywhip_config.det_sel.mode,
        exclude_blocked_dets=fierywhip_config.det_sel.exclude_blocked_dets,
        **kwargs,
    ):
        """
        initialize the Detector selection, uses defaults value from config
        :param grb: GRB object
        :type grb: fierywhip.frameworks.grbs.GRB
        """

        self.grb = grb
        self._max_sep = max_sep
        self._max_sep_normalizing = max_sep_normalizing
        self._max_number_nai = max_number_nai
        self._min_number_nai = min_number_nai
        self._mode = mode
        self._exclude_blocked_dets = exclude_blocked_dets
        self._set_position_interpolator()
        self._set_gbm()
        self._set_gbm_frame()
        if self._mode == "min_sep":
            logging.info("Using minimum separation mode")
            self._seps = self.gbm.get_separation(self.grb.position)
            self._set_good_dets()
            self._set_normalizing_det()
        elif self._mode == "max_sig" or self._mode == "max_sig_and_lowest":
            logging.info("Using maximum significance mode")
            self._trigdat_path = self.grb.trigdat
            self._set_good_dets_significance()
        elif self._mode == "max_sig_triplets":
            logging.info(f"Running detector selection mode {self._mode}")
            self._trigdat_path = self.grb.trigdat
            self._set_good_dets_significance_triplets()
        elif self._mode == "bgo_sides_no_bgo":
            logging.info(f"Running detector selection mode {self._mode}")
            self._trigdat_path = self.grb.trigdat
            self._bkg_yaml = kwargs.get("bkg_yaml", None)
            if self._bkg_yaml is None:
                raise ValueError("need to pass bkg_yaml for bgo_sides_no_bgo")
            with open(self._bkg_yaml, "r") as f:
                data = yaml.safe_load(f)
                dets = list(map(id2name, data["use_dets"]))
                if "b0" in dets:
                    dets.pop(dets.index("b0"))
                elif "b1" in dets:
                    dets.pop(dets.index("b1"))
            self._good_dets = dets
            self._normalizing_det = dets[0]
        elif self._mode == "bgo_sides":
            logging.info(f"Running detector selection mode {self._mode}")
            self._trigdat_path = self.grb.trigdat
            self.grb.save_timeselection()
            bkg_fit = BkgFittingTrigdat(
                "grb", "v00", self._trigdat_path, self.grb.timeselection_path
            )
            dets = list(map(id2name, bkg_fit.use_dets))
            self._good_dets = dets
            self._normalizing_det = dets[0]

        elif self._mode == "huntsville":
            logging.info(f"Running detector selection mode {self._mode}")
            self._trigdat_path = self.grb.trigdat
            self._set_good_dets_huntsville()
        elif self._mode == "all":
            logging.info(f"Running detector selection mode {self._mode}")
            self._good_dets = lu
            self._normalizing_det = "n0"
        else:
            raise NotImplementedError("Mode not implemented")

    def _set_good_dets_huntsville(self):
        table = pd.read_csv(
            pkg_resources.resource_filename(
                "fierywhip", "/data/huntsville_det_mapping.csv"
            ),
            index_col=0,
        )
        # convert to percent just for convenience
        for i in range(len(table)):
            table.iloc[i] = table.iloc[i] / table.iloc[i].sum() * 100
        tr = TrigReader(self._trigdat_path, fine=True, verbose=False)
        self.grb.run_timeselection()
        tr.set_active_time_interval(self.grb.active_time)
        tr.set_background_selections(*self.grb.bkg_time)
        tstart, tstop = tr.tstart_tstop()
        split = self.grb.active_time.split("-")
        if len(split) == 2:
            trigger_start, trigger_stop = list(map(float, split))
        elif len(split) == 3:
            trigger_start, trigger_stop = -float(split[1]), float(split[1])
        elif len(split) == 4:
            trigger_start, trigger_stop = -float(split[1]), -float(split[-1])
        else:
            raise ValueError
        res_dict = {}
        for d in lu:
            signs = tr.time_series[d].significance_per_interval
            median = np.median(signs)
            res_dict[d] = median
        sorted_sig = sorted(res_dict.items(), key=lambda x: x[1])
        i = -1
        flag = True
        min_percentage = 10
        use_dets = []
        logging.info(
            "Now setting the detector combinations which have "
            + f"at least {min_percentage}% occurence probability"
        )
        while flag:
            det = sorted_sig[i][0]
            if det not in use_dets and det not in ("b0", "b1"):
                # append the high sig itself
                logging.info(
                    f"Det {det} has a high significance - " + "will use it for refrence"
                )
                use_dets.append(det)
                temp = table.loc[det].to_dict()
                temp_sorted = sorted(temp.items(), key=lambda x: x[1])
                for x in temp_sorted:
                    if x[1] >= min_percentage and x[0] not in use_dets:
                        # appending dets which have the needed
                        # visibility probability
                        use_dets.append(x[0])
            if (
                len(use_dets) < self._min_number_nai
                and len(use_dets) < self._max_number_nai
            ):
                i -= 1
            else:
                flag = False

        if res_dict["b0"] >= res_dict["b1"]:
            use_dets.append("b0")
        else:
            use_dets.append("b1")
        self._good_dets = use_dets
        self._normalizing_det = use_dets[0]

    def _set_good_dets_significance_triplets(self):
        tr = TrigReader(self._trigdat_path, fine=True, verbose=False)
        self.grb.run_timeselection()
        tr.set_active_time_interval(self.grb.active_time)
        tr.set_background_selections(*self.grb.bkg_time)
        self._significances = {}
        tstart, tstop = tr.tstart_tstop()
        split = self.grb.active_time.split("-")
        if len(split) == 2:
            trigger_start, trigger_stop = list(map(float, split))
        elif len(split) == 3:
            trigger_start, trigger_stop = -float(split[1]), float(split[1])
        elif len(split) == 4:
            trigger_start, trigger_stop = -float(split[1]), -float(split[-1])
        else:
            raise ValueError
        logging.debug(
            f"The trigger start {trigger_start} and stop {trigger_stop}" + " times"
        )
        lowerid = np.argwhere(tstart >= trigger_start)[0, 0]
        upperid = np.argwhere(tstart > trigger_stop)[0, 0]

        for d in lu:
            signs = tr.time_series[d].significance_per_interval
            signs[:lowerid] = 0
            signs[upperid:] = 0
            self._significances[d] = np.max(signs)
        lu_nai = lu[:-2]
        sorted_sig = sorted(self._significances.items(), key=lambda x: x[1])
        logging.debug(sorted_sig)
        good_dets = []
        flag = True
        iterator = -1
        while flag:
            d = sorted_sig[iterator][0]
            if d not in good_dets and d in lu_nai:
                if len(good_dets) == 0:
                    self._normalizing_det = d
                logging.debug(f"adding corner of {d}")
                good_dets.extend(triplets[d])
            iterator -= 1
            if (
                len(good_dets) >= self._min_number_nai
                and len(good_dets) <= self._max_number_nai
            ):
                flag = False
        if self._significances["b0"] >= self._significances["b1"]:
            good_dets.append("b0")
        else:
            good_dets.append("b1")
        self._good_dets = good_dets
        self._sorted_significances = sorted_sig

    def _set_good_dets_significance(self):
        tr = TrigReader(self._trigdat_path, fine=True, verbose=False)
        self.grb.run_timeselection()
        tr.set_active_time_interval(self.grb.active_time)
        tr.set_background_selections(*self.grb.bkg_time)
        self._significances = {}
        # TODO set the significance outside the trigger area zero
        tstart, tstop = tr.tstart_tstop()
        split = self.grb.active_time.split("-")
        if len(split) == 2:
            trigger_start, trigger_stop = list(map(float, split))
        elif len(split) == 3:
            trigger_start, trigger_stop = -float(split[1]), float(split[1])
        elif len(split) == 4:
            trigger_start, trigger_stop = -float(split[1]), -float(split[-1])
        else:
            raise ValueError
        logging.debug(
            f"These are the trigger start {trigger_start} and stop {trigger_stop} times"
        )
        lowerid = np.argwhere(tstart >= trigger_start)[0, 0]
        upperid = np.argwhere(tstart > trigger_stop)[0, 0]

        for d in lu:
            signs = tr.time_series[d].significance_per_interval
            signs[:lowerid] = 0
            signs[upperid:] = 0
            self._significances[d] = np.max(signs)
        lu_nai = lu[:-2]
        sorted_sig = sorted(self._significances.items(), key=lambda x: x[1])
        logging.debug(sorted_sig)
        good_dets = []
        flag = True
        iterator = -1
        if self._exclude_blocked_dets:
            blocked_dets_table = pd.read_csv(
                pkg_resources.resource_filename("fierywhip", "data/blocked_dets.csv")
            )
            # blocked_dets = list(blocked_dets_table["grb" == self.grb.name]["blocked"])
            temp = blocked_dets_table[blocked_dets_table["grb"] == self.grb.name][
                "blocked"
            ].to_list()[0]

            temp = temp.strip(" ").strip("[").strip("]")
            if len(temp) == 0:
                blocked_dets = []
            else:
                temp = temp.split(",")
                if len(temp) > 0:
                    blocked_dets = list(map(id2name, temp))
                else:
                    blocked_dets = []
        else:
            blocked_dets = []
        logging.debug(blocked_dets)
        while flag:
            det = sorted_sig[iterator][0]
            if det not in good_dets and det in lu_nai and det not in blocked_dets:
                logging.debug(f"adding {det}")
                good_dets.append(det)
            iterator -= 1
            if (
                len(good_dets) >= self._min_number_nai
                and len(good_dets) <= self._max_number_nai
            ):
                flag = False
        if self._mode == "max_sig_and_lowest":
            logging.debug("Replacing")
            i = 0
            det = sorted_sig[i][0]
            while det not in lu_nai and det not in good_dets:
                i += 1
                det = sorted_sig[i][0]
            good_dets[len(good_dets) - 1] = det
        if self._significances["b0"] >= self._significances["b1"]:
            good_dets.append("b0")
        else:
            good_dets.append("b1")
        self._good_dets = good_dets
        self._normalizing_det = good_dets[0]
        self._sorted_significances = sorted_sig

    def _set_good_dets(self):
        temp = self._gbm.get_good_detectors(self.grb.position, self._max_sep)
        det_counter = 0
        good_dets = []
        for d in temp:
            if d not in ("b0", "b1"):
                det_counter += 1
                good_dets.append(d)
        if det_counter < self._min_number_nai:
            raise DetectorSelectionError("Too litle NaI dets")
        elif det_counter >= self._max_number_nai:
            temp = np.zeros(det_counter)
            i = 0
            for d in good_dets:
                temp[i] = np.abs(self._seps[d])
            good_dets_new = []
            counter = 0
            for el in temp.argsort():
                if counter < self._max_number_nai:
                    good_dets_new.append(good_dets[el])
                counter += 1
            good_dets = good_dets_new
        else:
            good_dets = good_dets
        self._good_dets = good_dets

    def _set_normalizing_det(self):
        seps = self._seps
        min_sep = 360
        min_sep_det = ""
        for d in self._good_dets:
            if np.abs(seps[d]) < min_sep:
                min_sep = np.abs(seps[d])
                min_sep_det = d

        if min_sep > self._max_sep_normalizing:
            raise DetectorSelectionError
        else:
            self._normalizing_det = min_sep_det
        if seps["b0"] <= seps["b1"]:
            self._good_dets.append("b0")
        else:
            self._good_dets.append("b1")

    def _set_position_interpolator(self):
        self._position_interpolator = PositionInterpolator.from_trigdat(
            self.grb.trigdat
        )

    def _set_gbm(self):
        self._gbm = GBM(
            self._position_interpolator.quaternion(0),
            sc_pos=self._position_interpolator.sc_pos(0) * u.km,
        )

    def _set_gbm_frame(self):
        quats = self._position_interpolator.quaternion(0)
        sc_pos = self._position_interpolator.sc_pos(0) * u.km
        self._gbm_frame = GBMFrame(
            quaternion_1=quats[0],
            quaternion_2=quats[1],
            quaternion_3=quats[2],
            quaternion_4=quats[3],
            sc_pos_X=sc_pos[0],
            sc_pos_Y=sc_pos[1],
            sc_pos_Z=sc_pos[2],
        )

    def set_good_dets(self, *dets):
        """
        Manually set good dets

        :param dets: list with dets
        :type dets: list
        """
        self._good_dets = dets

    @property
    def gbm(self):
        return self._gbm

    @property
    def gbm_frame(self):
        return self._gbm_frame

    @property
    def position_interpolator(self):
        return self._position_interpolator

    @property
    def normalizing_det(self):
        return self._normalizing_det

    @property
    def good_dets(self):
        return self._good_dets

    @property
    def sorted_significances(self):
        return self._sorted_significances

    @property
    def significances(self):
        return self._significances

    def _create_output_dict(self):
        return_dict = {}
        return_dict["grb"] = {}
        return_dict["grb"]["lon"] = float(
            self.grb.position.transform_to(self._gbm_frame).lon.deg
        )
        return_dict["grb"]["lat"] = float(
            self.grb.position.transform_to(self._gbm_frame).lat.deg
        )
        return_dict["grb"]["ra"] = float(self.grb.position.ra.deg)
        return_dict["grb"]["dec"] = float(self.grb.position.dec.deg)
        return_dict["separation"] = {}
        return_dict["significance"] = {}
        for d in self._good_dets:
            try:
                return_dict[d]["lon"] = None
            except KeyError:
                return_dict[d] = {}
            return_dict[d]["lon"] = float(self._gbm.get_centers([d])[0].lon.deg)
            return_dict[d]["lat"] = float(self._gbm.get_centers([d])[0].lat.deg)
            if self._mode == "min_sep":
                return_dict["separation"][d] = float(self._seps[d])
            elif self._mode == "max_sig":
                return_dict["significance"][d] = float(self._significances[d])
        return return_dict


class DetectorSelectionError(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "Failed to select detectors"
        super().__init__(message)
