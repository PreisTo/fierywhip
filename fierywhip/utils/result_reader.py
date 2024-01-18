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
                # only on ra/dec - if not so I fucked up badly
                self._parameters[paraname[-1]] = {}
                for k in dataframe.columns:
                    self._parameters[paraname[-1]][k] = row[k]

    # TODO ra,dec,1 und 2 sigma,

    def _build_report(self):
        raise NotImplementedError("This needs to be adapted correctly")
        self._report = {
            "general": {
                "grb_name": f"{self.grb_name}",
                "grb_name_gcn": f"{self._grb_name_gcn}",
                "report_type": f"{self.report_type}",
                "version": f"{self.version}",
                "trigger_number": self._trigger_number,
                "trigger_timestamp": self._trigger_timestamp,
                "data_timestamp": self._data_timestamp,
                "localization_timestamp": datetime.utcnow().strftime(
                    "%Y-%m-%dT%H:%M:%S.%fZ"
                ),
                "most_likely": self._most_likely,
                "second_most_likely": self._second_most_likely,
                "swift": self._swift,
            },
            "fit_result": {
                "model": self._model,
                "ra": convert_to_float(self._ra),
                "ra_err": convert_to_float(self._ra_err),
                "dec": convert_to_float(self._dec),
                "dec_err": convert_to_float(self._dec_err),
                "spec_K": convert_to_float(self._K),
                "spec_K_err": convert_to_float(self._K_err),
                "spec_index": convert_to_float(self._index),
                "spec_index_err": convert_to_float(self._index_err),
                "spec_xc": convert_to_float(self._xc),
                "spec_xc_err": convert_to_float(self._xc_err),
                "spec_alpha": convert_to_float(self._alpha),
                "spec_alpha_err": convert_to_float(self._alpha_err),
                "spec_xp": convert_to_float(self._xp),
                "spec_xp_err": convert_to_float(self._xp_err),
                "spec_beta": convert_to_float(self._beta),
                "spec_beta_err": convert_to_float(self._beta_err),
                "sat_phi": convert_to_float(self._phi_sat),
                "sat_theta": convert_to_float(self._theta_sat),
                "balrog_one_sig_err_circle": convert_to_float(
                    self._balrog_one_sig_err_circle
                ),
                "balrog_two_sig_err_circle": convert_to_float(
                    self._balrog_two_sig_err_circle
                ),
            },
            "time_selection": {
                "bkg_neg_start": self._bkg_neg_start,
                "bkg_neg_stop": self._bkg_neg_stop,
                "bkg_pos_start": self._bkg_pos_start,
                "bkg_pos_stop": self._bkg_pos_stop,
                "active_time_start": self._active_time_start,
                "active_time_stop": self._active_time_stop,
                "used_detectors": self._used_detectors,
            },
            "separation_values": {
                "bright_sources": self._dic_bright_sources,
                "SGRs": self._dic_SGRs,
                "Sun": {
                    "sun_separation": convert_to_float(self._sun_sep_center),
                    "sun_within_error": bool(self._sun_sep_error),
                },
            },
        }

    def save_result_yml(self, file_path):
        with open(file_path, "w") as f:
            yaml.dump(self._report, f, default_flow_style=False)
