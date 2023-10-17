#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import yaml
from effarea.utils.detectors import detector_list, nai_list, name_to_id
import math
import os

lu = detector_list()
nai = nai_list()


class Plots:
    def __init__(self):
        return None

    def from_result_yaml(self, yaml_path, base_dir=None):
        """ """
        if base_dir is None:
            self._base_dir = os.path.join(
                os.environ.get("GBMDATA"), "localizing/result_plots/"
            )
            if not os.path.exists(self._base_dir):
                os.makedirs(self._base_dir)
        else:
            self._base_dir = base_dir

        self._yaml_path = yaml_path
        if not os.path.exists(self._yaml_path):
            raise FileNotFoundError(
                "you have to supply the path to the correct yaml file"
            )
        with open(self._yaml_path, "r") as f:
            self._result_dict = yaml.safe_load(f)

        self._grbs = list(self._result_dict.keys())
        self.detectors_array = np.empty(len(nai), dtype=dict)
        self._energies = []

        for d, det in enumerate(nai):
            self.detectors_array[d] = self._detector_array(det)

    def _detector_array(self, det):
        energies_dict = {}
        for g in self._grbs:
            sep = self._result_dict[g]["separations"][det]
            for e in self._result_dict[g].keys():
                if e != "separations":
                    norm = self._result_dict[g][e][det]
                    if e not in energies_dict.keys():
                        energies_dict[e] = [[], []]
                        if e not in self._energies:
                            self._energies.append(e)
                    energies_dict[e][0].append(sep)
                    energies_dict[e][1].append(norm)
        return energies_dict

    def energy_scatter_plot(self, energies=None, det=None, path=None):
        if det is None:
            det = nai
        else:
            assert type(det) == list, "Det must be a list"
        if energies is None:
            energies = self._energies
            try:
                energies.pop(energies.index("8.1-700"))
            except IndexError:
                pass
            starting_energies = []
            for i, e in enumerate(energies):
                starting_energies.append(float(e.split("-")[0]))
            sorting_indices = np.argsort(np.array(starting_energies))
            sorted_energies = []
            for i in sorting_indices:
                sorted_energies.append(energies[i])
            energies = sorted_energies
        else:
            assert type(energies) == list, "Energies must be list"
        if path is None:
            plot_path = os.path.join(self._base_dir, "dets")
        else:
            plot_path = path

        for d in det:
            num_plots = len(energies)

            energy_dict = self.detectors_array[name_to_id(d)]
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            num_axes = math.ceil(len(energies) / num_plots)
            fig, axes = plt.subplots(num_axes)
            if type(axes) != list:
                axes = [axes]
            counter = 0
            ax = 0
            colors = list(mcolors.TABLEAU_COLORS.values())
            for e in energies:
                e_start = round(float(e.split("-")[0]), 3)
                e_stop = round(float(e.split("-")[-1]), 3)

                if counter < num_plots:
                    plot_data = np.sort(energy_dict[e][0:2], axis=1)
                    axes[ax].plot(
                        plot_data[0],
                        plot_data[1],
                        marker="+",
                        linestyle="--",
                        color=colors[counter],
                        label=f"Energy {e_start}-{e_stop}",
                    )
                    counter += 1
                else:
                    ax += 1
                    counter = 0
                    plot_data = np.sort(energy_dict[e][0:2], axis=1)
                    axes[ax].plot(
                        plot_data[0],
                        plot_data[1],
                        marker="+",
                        linestyle="--",
                        color=colors[counter],
                        label=f"Energy {e_start}-{e_stop}",
                    )
                    counter += 1
            for ax in axes:
                ax.legend()

            fig.tight_layout()
            fig.savefig(os.path.join(plot_path, f"{d}.pdf"))
            plt.close(fig)

    def det_scatter_plot(self, plot_path=None, dets=None):
        if plot_path is None:
            plot_path = os.path.join(self._base_dir, "energies")
        if dets is None:
            dets = nai_list()
        for e in self._energies:
            num_plots = 6

            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            num_axes = math.ceil(len(dets) / num_plots)
            fig, axes = plt.subplots(num_axes, figsize=(5, 7), sharex=True)
            if type(axes) not in (list, np.ndarray):
                axes = [axes]
            counter = 0
            ax = 0
            colors = list(mcolors.TABLEAU_COLORS.values())
            sides = [
                ["n0", "n1", "n2", "n3", "n4", "n5"],
                ["n6", "n7", "n8", "n9", "na", "nb"],
            ]
            for s in sides:
                for d in s:
                    energy_dict = self.detectors_array[name_to_id(d)]
                    if counter < num_plots:
                        plot_data = np.sort(energy_dict[e][0:2], axis=1)
                        axes[ax].plot(
                            plot_data[0],
                            plot_data[1],
                            marker="+",
                            linestyle="--",
                            color=colors[counter],
                            label=f"Det {d}",
                        )
                        counter += 1
                    else:
                        ax += 1
                        counter = 0
                        plot_data = np.sort(energy_dict[e][0:2], axis=1)
                        axes[ax].plot(
                            plot_data[0],
                            plot_data[1],
                            marker="+",
                            linestyle="--",
                            color=colors[counter],
                            label=f"Det {d}",
                        )
                        counter += 1
            for ax in axes:
                ax.legend()
            e_start = round(float(e.split("-")[0]), 3)
            e_stop = round(float(e.split("-")[-1]), 3)

            axes[0].set_title("b0 side")
            axes[1].set_title("b1 side")
            fig.tight_layout()
            fig.savefig(os.path.join(plot_path, f"{e_start}-{e_stop}.pdf"))
            plt.close(fig)
