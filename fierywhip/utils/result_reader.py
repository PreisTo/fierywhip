#!/usr/bin/env python3

from threeML.analysis_results import BayesianResults, load_analysis_results
from fierywhip.frameworks.grbs import GRB
import logging
from chainconsumer import ChainConsumer
import os
import numpy as np
import yaml


class ResultReader:
    def __init__(
        self, grb: GRB, post_equal_weights_file: str, results_file: str, **kwargs
    ):
        self._post_equal_weights_file = post_equal_weights_file
        self._bayesian_results = load_analysis_results(results_file)
        self._grb = grb
        self._get_parameters_with_errors()
        self._get_error_radii()
        self._build_report()
        self._create_plots()

    def _get_parameters_with_errors(self, mode="hpd"):
        lu_comps = {"first": "1", "second": "2", "third": "3", "fourth": "4"}
        dataframe = self._bayesian_results.get_data_frame(mode)
        self._parameters = {}
        for i, row in dataframe.iterrows():
            paraname = str(i).split(".")
            if "ra" in paraname or "dec" in paraname:
                # only on ra/dec - if not so I fucked up badly
                self._parameters[paraname[-1]] = {}
                for k in dataframe.columns:
                    self._parameters[paraname[-1]][k] = row[k]
            else:
                self._parameters[f"{paraname[-1]}{lu_comps[paraname[0]]}"] = {}
                for k in dataframe.columns:
                    self._parameters[f"{paraname[-1]}{lu_comps[paraname[0]]}"][k] = row[
                        k
                    ]

        self._ra = float(self._parameters["ra"]["value"])
        if np.absolute(self._parameters["ra"]["positive_error"]) >= np.absolute(
            self._parameters["ra"]["negative_error"]
        ):
            self._ra_err = np.absolute(self._parameters["ra"]["positive_error"])
        else:
            self._ra_err = np.absolute(self._parameters["ra"]["negative_error"])

        self._dec = float(self._parameters["dec"]["value"])
        if np.absolute(self._parameters["dec"]["positive_error"]) >= np.absolute(
            self._parameters["dec"]["negative_error"]
        ):
            self._dec_err = np.absolute(self._parameters["dec"]["positive_error"])
        else:
            self._dec_err = np.absolute(self._parameters["dec"]["negative_error"])

        for para in self._parameters.keys():
            if para not in ("ra", "dec"):
                val = self._parameters[para]["value"]
                if np.absolute(self._parameters[para]["positive_error"]) >= np.absolute(
                    self._parameters[para]["negative_error"]
                ):
                    err = np.absolute(self._parameters[para]["positive_error"])
                else:
                    err = np.absolute(self._parameters[para]["negative_error"])
                paran = "self._" + para
                exec(paran + "=val")
                para_err = "self._" + para + "_err"
                logging.info(para_err)
                exec(para_err + "=err")

    def _get_error_radii(self):
        chain = np.loadtxt(self._post_equal_weights_file)
        c = ChainConsumer()
        c.add_chain(chain[:, :-1], parameters=list(self._parameters.keys())).configure(
            plot_hists=False,
            contour_labels="sigma",
            colors="#cd5c5c",
            flip=False,
            max_ticks=3,
        )
        chains, parameters, truth, extents, blind, log_scales = c.plotter._sanitise(
            None, None, None, None, color_p=True, blind=None
        )

        hist, x_contour, y_contour = c.plotter._get_smoothed_histogram2d(
            chains[0], "ra", "dec"
        )
        hist[hist == 0] = 1e-16
        val_contour = c.plotter._convert_to_stdev(hist.T)

        # Truth Values
        ra = float(self._parameters["ra"]["value"])
        dec = float(self._parameters["dec"]["value"])

        # One Sigma Error Radius
        mask = val_contour < 0.68
        points = []
        for i in range(len(mask)):
            for j in range(len(mask[i])):
                if mask[i][j]:
                    points.append([x_contour[j], y_contour[i]])
        points = np.array(points)
        best_fit_point = [ra, dec]
        best_fit_point_vec = [
            np.cos(best_fit_point[1] * np.pi / 180)
            * np.cos(best_fit_point[0] * np.pi / 180),
            np.cos(best_fit_point[1] * np.pi / 180)
            * np.sin(best_fit_point[0] * np.pi / 180),
            np.sin(best_fit_point[1] * np.pi / 180),
        ]
        alpha_largest = 0

        for point_2 in points:
            point_2_vec = [
                np.cos(point_2[1] * np.pi / 180) * np.cos(point_2[0] * np.pi / 180),
                np.cos(point_2[1] * np.pi / 180) * np.sin(point_2[0] * np.pi / 180),
                np.sin(point_2[1] * np.pi / 180),
            ]
            alpha = np.arccos(np.dot(point_2_vec, best_fit_point_vec)) * 180 / np.pi
            if alpha > alpha_largest:
                alpha_largest = alpha
        alpha_one_sigma = alpha_largest

        mask = val_contour < 0.95
        points = []
        for i in range(len(mask)):
            for j in range(len(mask[i])):
                if mask[i][j]:
                    points.append([x_contour[j], y_contour[i]])
        points = np.array(points)
        alpha_largest = 0

        for point_2 in points:
            point_2_vec = [
                np.cos(point_2[1] * np.pi / 180) * np.cos(point_2[0] * np.pi / 180),
                np.cos(point_2[1] * np.pi / 180) * np.sin(point_2[0] * np.pi / 180),
                np.sin(point_2[1] * np.pi / 180),
            ]
            alpha = np.arccos(np.dot(point_2_vec, best_fit_point_vec)) * 180 / np.pi
            if alpha > alpha_largest:
                alpha_largest = alpha
        alpha_two_sigma = alpha_largest

        self._balrog_1_sigma = alpha_one_sigma
        self._balrog_2_sigma = alpha_two_sigma
        self._balrog_one_sig_err_cirle = alpha_one_sigma
        self._balrog_two_sig_err_circle = alpha_two_sigma
        logging.info(
            f"Calculated 1 and 2 sigma errors to {round(self._balrog_1_sigma,3)}"
            + f" and {round(self._balrog_2_sigma,3)}"
        )

    def _build_report(self):
        temp = list(self._grb.bkg_time)
        if len(temp) == 0:
            raise NotImplementedError
        else:
            bkg_neg, bkg_pos = temp
            active_time = self._grb.active_time
        bkg_neg_start, bkg_neg_stop = time_splitter(bkg_neg)
        bkg_pos_start, bkg_pos_stop = time_splitter(bkg_pos)
        active_time_start, active_time_stop = time_splitter(active_time)
        self._report = {
            "general": {
                "grb_name": f"{self._grb.name}",
            },
            "fit_result": {
                "model": "cpl",
                "ra": convert_to_float(self._ra),
                "ra_err": convert_to_float(self._ra_err),
                "dec": convert_to_float(self._dec),
                "dec_err": convert_to_float(self._dec_err),
                "spec_K": convert_to_float(self._K1),
                "spec_K_err": convert_to_float(self._K1_err),
                "spec_index": convert_to_float(self._index1),
                "spec_index_err": convert_to_float(self._index1_err),
                "spec_xc": convert_to_float(self._xc1),
                "spec_xc_err": convert_to_float(self._xc1_err),
                "balrog_one_sig_err_circle": convert_to_float(self._balrog_1_sigma),
                "balrog_two_sig_err_circle": convert_to_float(self._balrog_2_sigma),
            },
            "time_selection": {
                "bkg_neg_start": bkg_neg_start,
                "bkg_neg_stop": bkg_neg_stop,
                "bkg_pos_start": bkg_pos_start,
                "bkg_pos_stop": bkg_pos_stop,
                "active_time_start": active_time_start,
                "active_time_stop": active_time_stop,
                "used_detectors": self._grb.detector_selection.good_dets,
            },
        }
        if len(self._parameters.keys()) >= 8:
            logging.debug("exporting parameters for second spectrum")
            self._report["fit_result"]["spec_K2"] = convert_to_float(self._K2)
            self._report["fit_result"]["spec_K2_err"] = convert_to_float(self._K2_err)
            self._report["fit_result"]["spec_index2"] = convert_to_float(self._index2)
            self._report["fit_result"]["spec_index2_err"] = convert_to_float(
                self._index2_err
            )
            self._report["fit_result"]["spec_xc2"] = convert_to_float(self._xc2)
            self._report["fit_result"]["spec_xc2_err"] = convert_to_float(self._xc2_err)

        elif len(self._parameters.keys()) >= 11:
            logging.debug("exporting parameters for third spectrum")
            self._report["fit_result"]["spec_K3"] = convert_to_float(self._K3)
            self._report["fit_result"]["spec_K3_err"] = convert_to_float(self._K3_err)
            self._report["fit_result"]["spec_index3"] = convert_to_float(self._index3)
            self._report["fit_result"]["spec_index3_err"] = convert_to_float(
                self._index3_err
            )
            self._report["fit_result"]["spec_xc3"] = convert_to_float(self._xc3)
            self._report["fit_result"]["spec_xc3_err"] = convert_to_float(self._xc3_err)
        elif len(self._parameters.keys()) >= 14:
            logging.debug("exporting parameters for fourth spectrum")
            self._report["fit_result"]["spec_K4"] = convert_to_float(self._K4)
            self._report["fit_result"]["spec_K4_err"] = convert_to_float(self._K4_err)
            self._report["fit_result"]["spec_index4"] = convert_to_float(self._index4)
            self._report["fit_result"]["spec_index4_err"] = convert_to_float(
                self._index4_err
            )
            self._report["fit_result"]["spec_xc4"] = convert_to_float(self._xc4)
            self._report["fit_result"]["spec_xc4_err"] = convert_to_float(self._xc4_err)

    def save_result_yml(self, file_path):
        with open(file_path, "w") as f:
            yaml.dump(self._report, f, default_flow_style=False)

    def _create_plots(self):
        if False:
            fig = self._bayesian_results.corner_plot()
            fig.savefig()

    @property
    def ra(self):
        return self._ra, self._ra_err

    @property
    def dec(self):
        return self._dec, self._dec_err

    @property
    def K(self):
        return self._K1, self._K1_err

    @property
    def index(self):
        return self._index1, self._index1_err

    @property
    def balrog_1_sigma(self):
        return self._balrog_1_sigma

    @property
    def balrog_2_sigma(self):
        return self._balrog_2_sigma


def convert_to_float(value):
    if value is not None:
        return float(value)
    else:
        return None


def time_splitter(time):
    times = time.split("-")
    if len(times) == 2:
        start = float(times[0])
        stop = float(times[1])
    elif len(times) == 3:
        start = -float(times[1])
        stop = float(times[-1])
    elif len(times) == 4:
        start = -float(times[1])
        stop = -float(times[-1])
    else:
        raise ValueError(f"Something wrong with the passed time {time}")
    return start, stop
