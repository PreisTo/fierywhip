#!/usr/bin/env python3

from threeML.analysis_results import BayesianResults
from fierywhip.frameworks.grbs import GRB
import logging


parameter_look_up = {
    "short": ["ra", "dec", "K", "index", "xc"],
    "long": ["ra", "dec", "K1", "index1", "xc1", "K2", "index2", "xc2"],
}


class ResultReader:
    def __init__(
        self,
        results: BayesianResults,
        grb: GRB,
    ):
        self._bayesian_results = results
        self._grb = grb
        # Displaying result
        self._bayesian_results.display()
        self._base_dir = os.path.join(
            os.environ.get("GBM_TRIGGER_DATA_DIR"), self._grb.name, "trigdat/v00"
        )
        # Storing
        self._bayesian_results.write_to(
            os.path.join(self._base_dir, "trigdat_results.fits"), overwrite=True
        )

    def _get_parameters_with_errors(self, mode="hpd"):
        dataframe = self._bayesian_results.get_data_frame(mode)
        self._parameters = {}
        for i, row in dataframe.iterrows():
            paraname = str(i).split(".")
            if "ra" in paraname or "dec" in paraname:
                self._parameters[paraname[-1]] = {}
                for k in dataframe.columns:
                    self._parameters[paraname[-1]][k] = row[k]
