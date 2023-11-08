#!/usr/bin/env python3

import matplotlib.pyplot as plt
from threeML import *
from fierywhip.utils.detectors import name_to_id, detector_list
from mpi4py import MPI
import yaml

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import os
import yaml
import numpy as np

lu = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb"]
lu_id = {
    "n0": 0,
    "n1": 1,
    "n2": 2,
    "n3": 3,
    "n4": 4,
    "n5": 5,
    "n6": 6,
    "n7": 7,
    "n8": 8,
    "n9": 9,
    "na": 10,
    "nb": 11,
}


class Exporter:
    def __init__(self, model):
        self._results = model.results
        self._model = model
        self._temp_chains_dir = model._temp_chains_dir
        self.grb = model.grb
        self._yaml_path = model._yaml_path
        self.plot_corners()
        self.plot_spectrum()

    def plot_corners(self):
        if rank == 0:
            fig = self._results.corner_plot()
            fig.savefig(os.path.join(self._temp_chains_dir, "cplot.pdf"))
            plt.close("all")

    def plot_spectrum(self):
        if rank == 0:
            spectrum_plot = display_spectrum_model_counts(self._results)
            ca = spectrum_plot.get_axes()[0]
            y_lims = ca.get_ylim()
            if y_lims[0] < 10e-6:
                ca.set_ylim(bottom=10e-6)
            if y_lims[1] > 10e4:
                ca.set_ylim(top=10e4)
            spectrum_plot.tight_layout()
            spectrum_plot.savefig(
                os.path.join(self._temp_chains_dir, "splot.pdf"),
                bbox_inches="tight",
            )
            plt.close("all")

    def export_yaml(self):
        if rank == 0:
            df = self._results.get_data_frame("hpd")
            result_dict = self.grb.detector_selection._create_output_dict()
            result_dict["fit"] = {}
            result_dict["fit"]["values"] = {}
            result_dict["fit"]["errors"] = {}
            for fp in self._results.optimized_model.free_parameters:
                result_dict["fit"]["values"][
                    self._results.optimized_model.free_parameters[fp].name
                ] = float(self._results.optimized_model.free_parameters[fp].value)
            for i in df.index:
                result_dict["fit"]["errors"][df.loc[i].name] = {}
                result_dict["fit"]["errors"][df.loc[i].name]["negative_error"] = float(
                    df.loc[i]["negative_error"]
                )
                result_dict["fit"]["errors"][df.loc[i].name]["positive_error"] = float(
                    df.loc[i]["positive_error"]
                )
            try:
                with open(os.path.join(self._yaml_path, "results.yml"), "r") as f:
                    loaded = yaml.safe_load(f)
            except FileNotFoundError:
                loaded = {}
            loaded[self.grb.name] = result_dict
            with open(os.path.join(self._yaml_path, "results.yml"), "w") as f:
                yaml.dump(loaded, f)

    def export_matrix(self):
        if rank == 0:
            res_df = self._results.get_data_frame("hpd")

            norm = self.grb.detector_selection.normalizing_det
            try:
                data = np.load(
                    os.path.join(self._yaml_path, "det_matrix.npy"), allow_pickle=True
                )
            except FileNotFoundError:
                data = np.empty((12, 12, 3), dtype=list)
                for i in range(12):
                    for j in range(12):
                        for k in range(3):
                            data[i, j, k] = []
            norm_id = lu_id[norm]
            for det in self.grb.detector_selection.good_dets:
                if det not in ("b0","b1"):
                    det_id = lu_id[det]
                    for fp in self._results.optimized_model.free_parameters:
                        para_name = self._results.optimized_model.free_parameters[fp].name
                        if f"cons_{det}" in para_name:
                            data[norm_id, det_id, 0].append(
                                float(res_df.loc[para_name]["value"])
                            )
                            data[norm_id, det_id, 1].append(
                                float(res_df.loc[para_name]["negative_error"])
                            )
                            data[norm_id, det_id, 2].append(
                                float(res_df.loc[para_name]["positive_error"])
                            )
            data[norm_id, norm_id, 0].append(1)
            data[norm_id, norm_id, 1].append(0)
            data[norm_id, norm_id, 2].append(0)
            np.save(os.path.join(self._yaml_path, "det_matrix.npy"), data)
            print("Successfully saved matrix to file")


def matrix_from_yaml(path, exclude=[]):
    if rank == 0:
        data = np.empty((12, 12, 3), dtype=list)
        for i in range(12):
            for j in range(12):
                for k in range(3):
                    data[i, j, k] = []

        with open(path, "r") as f:
            res = yaml.safe_load(f)
        for grb in res.keys():
            if grb not in exclude:
                dets = list(res[grb]["separation"].keys())
                use_dets = []
                for d in dets:
                    if f"cons_{d}" not in res[grb]["fit"]["values"].keys():
                        norm_det = d
                    else:
                        use_dets.append(d)
                norm_id = name_to_id(norm_det)
                for d in use_dets:
                    det_id = name_to_id(d)
                    err_neg = float(
                        res[grb]["fit"]["errors"][f"cons_{d}"]["negative_error"]
                    )
                    err_pos = float(
                        res[grb]["fit"]["errors"][f"cons_{d}"]["positive_error"]
                    )
                    val = float(res[grb]["fit"]["values"][f"cons_{d}"])
                    data[norm_id, det_id, 0].append(val)
                    data[norm_id, det_id, 1].append(np.abs(err_neg))
                    data[norm_id, det_id, 2].append(np.abs(err_pos))
                data[norm_id, norm_id, 0].append(1)
                data[norm_id, norm_id, 1].append(0)
                data[norm_id, norm_id, 2].append(0)

    return data
