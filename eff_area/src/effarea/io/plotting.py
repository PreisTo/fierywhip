#!/usr/bin/env python3

import matplotlib.pyplot as plt
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
        self.__init__()
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
        self.detectors_array = np.empty(len(nai))
        self._energies = []

        for d, det in enumerate(nai):
            self.detectors_array[i] = self._detector_array(det)

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

    def energy_scatter_plot(self, energies=None, det=None):
        if det is None:
            det = nai
        else:
            assert type(det) == list, "Det must be a list"
        if energies is None:
            energies = self._energies
        else:
            assert type(energies) == list, "Energies must be list"

        for d in det:
            plot_path = os.path.join(self._base_dir)
            energy_dict = self.detectors_array[name_to_id(d)]
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            num_axes = math.ceil(len(energies) / 3)
            fig, axes = plt.subplots(num_axes)
            counter = 0
            ax = 0
            colors = ["red", "blue", "green"]
            for e in energies:
                if counter < 3:
                    axes[ax].scatter(
                        energy_dict[e][0],
                        energy_dict[e][1],
                        color=colors[counter],
                        label=f"Bins {e}",
                    )
                    counter += 1
                else:
                    ax += 1
                    counter = 0
            axes[0].set
            fig.tight_layout()
            fig.savefig(os.path.join(plot_path, f"{d}_.pdf"))
