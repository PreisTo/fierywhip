#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from fierywhip.config.configuration import fierywhip_config
import os

lu = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb"]


class DetDistPlot:
    def __init__(
        self,
        result_path=os.path.join(
            os.environ.get("GBMDATA"), "localizing/det_matrix.npy"
        ),
        matrix=None,
        lims=(fierywhip_config.config.eff_corr_lim_low, fierywhip_config.config.eff_corr_lim_high),
        selection_plot=True,
    ):
        if matrix is None:
            self._matrix = np.load(result_path, allow_pickle=True)
        else:
            self._matrix = matrix
        self._lims = lims
        self._make_selection_plot = selection_plot
        self._eff_area_correction_plot()
        self._selection_plot()

    def _eff_area_correction_plot(self):
        # TODO use normalization/normalization_matrix
        if self._make_selection_plot:
            self._selection_matrix = np.zeros((12, 12), dtype=np.float64)
        fig, ax = plt.subplots(1, figsize=(20, 21))
        matrix = self._matrix[:, :, 0]
        error_pos = np.array(self._matrix[:, :, 1])
        error_neg = np.array(self._matrix[:, :, 2])
        print(matrix)
        blank = np.empty((12, 12))
        self._rejected = {}
        for i in range(12):
            for j in range(12):
                try:
                    if i != j:
                        ep = error_pos[i, j]
                        en = error_neg[i, j]
                        if self._lims is not None:
                            pop_indices = []
                            for ind in range(len(matrix[i, j])):
                                if (
                                    np.abs(matrix[i, j][ind] / self._lims[0] - 1) < 0.1
                                    or np.abs(matrix[i, j][ind] / self._lims[1] + 1)
                                    < 0.1
                                ):
                                    if (np.abs(en[ind]) + np.abs(en[ind])) < 0.1:
                                        pop_indices.append(ind)
                            try:
                                self._rejected[f"{i},{j}"] = len(pop_indices)
                                for p in pop_indices.reverse():
                                    matrix[i, j].pop(p)
                                    en.pop(p)
                                    ep.pop(p)
                            except TypeError:
                                pass
                        ep = np.abs(np.array(ep))
                        en = np.abs(np.array(en))
                        try:
                            blank[i, j] = round(
                                np.average(matrix[i, j], weights=1 / (ep + en)),
                                3,
                            )
                        except ZeroDivisionError:
                            blank[i, j] = round(np.mean(matrix[i, j]), 3)
                    else:
                        blank[i, j] = 100
                except ValueError:
                    blank[i, j] = np.nan

        print(blank)
        im = ax.imshow(
            blank,
            cmap="coolwarm",
            vmin=fierywhip_config.config.eff_corr_lim_low,
            vmax=fierywhip_config.config.eff_corr_lim_high,
        )
        for i in range(12):
            blank[i, i] = int(np.sum(matrix[i, i]))
        for index, label in np.ndenumerate(blank):
            i = index[0]
            j = index[1]
            if self._make_selection_plot:
                if f"{i},{j}" in self._rejected.keys():
                    self._selection_matrix[i, j] = (
                        len(self._matrix[i, j, 0]) - self._rejected[f"{i},{j}"]
                    )
                else:
                    self._selection_matrix[i, j] = len(self._matrix[i, j, 0])
            if str(label) != "nan":
                try:
                    label = (
                        str(label)
                        + f", {len(self._matrix[i,j,0])}, {self._rejected[f'{i},{j}']}"
                    )
                except KeyError:
                    label = str(label) + f", {len(self._matrix[i,j,0])}, 0"
                ax.text(j, i, label, ha="center", va="center")
        ax.axvline(x=5.5, color="black")
        ax.axhline(y=5.5, color="black")
        ax.set_title(
            f"Rel. Normalization of dets - 3 or 4 NaI together + 1 BGO - 60/40deg sep/norm_sep\n{np.trace(blank)} total of grbs"
        )
        ax.set_ylabel("Normalizing Det")
        ax.set_yticks(range(12), labels=lu)
        ax.set_xticks(range(12), labels=lu)
        ax.set_xlabel("Detector")
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.2, top=0.8)
        cax = fig.add_axes([0.2, 0.1, 0.6, 0.05])
        fig.colorbar(im, cax=cax, orientation="horizontal")
        fig.savefig(os.path.join(fierywhip_config.config.default_plot_path, "test.pdf"))
        print(self._rejected)

    def _selection_plot(self):
        fig, ax = plt.subplots(1, figsize=(20, 21))
        matrix = self._selection_matrix.copy().astype(np.float64)
        total_bursts = np.trace(matrix)
        matrix = matrix / total_bursts * 100
        np.fill_diagonal(matrix, np.nan)
        im = ax.imshow(matrix, cmap="coolwarm")
        for i in range(12):
            for j in range(12):
                ax.text(
                    j,
                    i,
                    round(self._selection_matrix[i, j] / total_bursts * 100, 3),
                    ha="center",
                    va="center",
                )
        ax.set_ylabel("Normalizing Det")
        ax.set_xlabel("Detector")
        ax.set_xticks(range(12), labels=lu)
        ax.set_yticks(range(12), labels=lu)
        ax.set_title(
            f"Relative Selection of dets for {total_bursts} GRBs\n Mode: {fierywhip_config.config.det_sel.mode}"
        )
        fig.tight_layout()

        plt.subplots_adjust(bottom=0.2, top=0.8)
        cax = fig.add_axes([0.2, 0.1, 0.6, 0.05])
        fig.colorbar(im, cax=cax, orientation="horizontal")
        fig.savefig(
            os.path.join(fierywhip_config.config.default_plot_path, "test_selection.pdf")
        )
