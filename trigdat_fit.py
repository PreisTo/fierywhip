#!/usr/bin/env python3
from fierywhip.normalizations.normalization_matrix import NormalizationMatrix
from fierywhip.io.export import matrix_from_yaml
from fierywhip.data.grbs import GRB, GRBList
from fierywhip.model.trigdat import GRBModel
from fierywhip.normalizations.normalization_matrix import NormalizationMatrix
import os

if __name__ == "__main__":
    norm_matrix = NormalizationMatrix(
        result_yml=os.path.join(os.environ.get("GBMDATA"), "localizing/results.yml")
    )
    grb_list = GRBList(check_finished=False)

    for grb in grb_list.grbs:
        grb._get_effective_area_correction(norm_matrix)
        grb_model = GRBModel(grb)
        grb_model.fit()
        grb_model.export_results()
