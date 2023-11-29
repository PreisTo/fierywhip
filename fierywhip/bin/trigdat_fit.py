#!/usr/bin/env python3
from fierywhip.normalizations.normalization_matrix import NormalizationMatrix
from fierywhip.io.export import matrix_from_yaml
from fierywhip.frameworks.grbs import GRB, GRBList
from fierywhip.model.trigdat import GRBModel
from fierywhip.normalizations.normalization_matrix import NormalizationMatrix
import os
import numpy as np

if __name__ == "__main__":
    use_eff_area = False
    if use_eff_area:
        nm_object = NormalizationMatrix(
            result_yml=os.path.join(os.environ.get("GBMDATA"), "localizing/results.yml")
        )
        norm_matrix = nm_object.matrix.copy()
    grb_list = GRBList(check_finished=False)

    for grb in grb_list.grbs:
        if use_eff_area:
            grb._get_effective_area_correction(norm_matrix)
        grb_model = GRBModel(grb)
        grb_model.fit()
        grb_model.export_results()
