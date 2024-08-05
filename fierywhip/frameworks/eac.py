#!/usr/bin/python

lu = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb"]


class EffectiveAreaNormalization:
    def __init__(self, effective_area_dict, reference_detector="n0"):
        self._eff_area_dict = effective_area_dict
        self._ref_d = reference_detector
        self._create_complete_dict()

    def _create_complete_dict(self):
        complete_dict = {}
        complete_dict[self._ref_d] = self._eff_area_dict
        for d in lu:
            if d not in complete_dict.keys():
                factor = 1 / complete_dict[self._ref_d][d]
                for k in lu:
                    complete_dict[d][k] = complete_dict[self._ref_d][k] * factor
        self._complete_dict = complete_dict

    @property
    def complete_dict(self):
        return self._complete_dict

    def get_eac_for_det(self, det, norm="n0"):
        return self._complete_dict[norm][det]
