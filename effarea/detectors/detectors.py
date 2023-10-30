#!/usr/bin/python
from gbmgeometry.gbm import GBM
from gbmgeometry.position_interpolator import PositionInterpolator
from gbmgeometry.gbm_frame import GBMFrame
import astropy.units as u
import numpy as np

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


class DetectorSelection:
    def __init__(self, grb, max_sep=60, max_sep_normalizing=20, max_number_nai=3):
        self.grb = grb
        self._max_sep = max_sep
        self._max_sep_normalizing = max_sep_normalizing
        self._max_numer_nai = max_number_nai
        self._set_position_interpolator()
        self._set_gbm()
        self._set_gbm_frame()
        self._seps = self.gbm.get_separation(self.grb.position)
        self._set_good_dets()
        self._set_normalizing_det()

    def _set_good_dets(self):
        temp = self._gbm.get_good_detectors(self.grb.position, self._max_sep)
        det_counter = 0
        good_dets = []
        for d in temp:
            if d not in ("b0", "b1"):
                det_counter += 1
                good_dets.append(d)
        if det_counter < self._max_numer_nai:
            raise DetectorSelectionError("Too litle NaI dets")
        elif det_counter > self._max_numer_nai:
            temp = np.zeros(det_counter)
            i = 0
            for d in good_dets:
                temp[i] = self._seps[d]
            good_dets_new = []
            counter = 0
            for el in temp.argsort():
                if counter < self._max_numer_nai:
                    good_dets_new.append(good_dets[el])
                counter += 1
            good_dets = good_dets_new
        self._good_dets = good_dets

    def _set_normalizing_det(self):
        seps = self._seps
        min_sep = 180
        min_sep_det = ""
        for d in self._good_dets:
            if seps[d] < min_sep:
                min_sep = seps[d]
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
        for d in self._good_dets:
            try:
                return_dict[d]["lon"] = None
            except KeyError:
                return_dict[d] = {}
            return_dict[d]["lon"] = float(self._gbm.get_centers([d])[0].lon.deg)
            return_dict[d]["lat"] = float(self._gbm.get_centers([d])[0].lat.deg)
            return_dict["separation"][d] = float(self._seps[d])
        return return_dict


class DetectorSelectionError(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "Failed to select detectors"
        super().__init__(message)
