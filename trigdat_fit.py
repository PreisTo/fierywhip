#!/usr/bin/env python3
from fierywhip.normalizations.normalization_matrix import NormalizationMatrix
from fierywhip.io.export import matrix_from_yaml
from fierywhip.data.grbs import GRB, GRBList
from fierywhip.model.trigdat import GRBModel
import os

if __name__ == "__main__":
    norm_matrix = NormalizationMatrix(
        result_yml=os.path.join(os.environ.get("GBMDATA"), "localizing/results.yml")
    )
    grb_list = GRBList(normalizing_matrix=norm_matrix)
    for grb in grb_list.grbs:
        grb_model = GRBModel(grb)
        grb_model.fit()
        grb_model.export_results()
