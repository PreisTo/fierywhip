#!/usr/bin/python

from effarea.data.data import GRB
from gbmgeometry.position_interpolator import PositionInterpolator
from gbmgeometry.gbm_frame import GBMFrame
from gbmgeometry.gbm import GBM
import astropy.units as u


class DetectorSelection:
    def __init__(self, grb):
        assert type(grb) is GRB, "grb needs to be an GRB object"
        self._set_position_interpolator()
        self._set_gbm()

    def _set_position_interpolator(self):
        self._position_interpolator = PositionInterpolator.from_trigdat(
            self.grb.trigdat
        )

    def _set_gbm(self):
        self._gbm = GBM(
            self._position_interpolator.quaternion(0),
            sc_pos=self._position_interpolator.sc_pos(0) * u.km,
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
