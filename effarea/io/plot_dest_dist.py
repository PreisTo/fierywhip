#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


class DetDistPlot:
    def __init__(self, result_path="/home/tobi/Schreibtisch/localizing/det_matrix.npy"):
        self._matrix = np.load(result_path, allow_pickle=True)

        fig, ax = plt.subplots(1, figsize=(15, 15))
        cax = fig.add_axes([0.8, 0.1, 0.15, 0.6])
        matrix = self._matrix[:, :, 0]
        blank = np.empty((12, 12))
        for i in range(12):
            for j in range(12):
                blank[i, j] = np.nan_to_num(np.mean(matrix[i, j]))
        im = ax.imshow(blank, cmap="plasma")
        for (j, i), label in np.ndenumerate(blank):
            ax.text(i, j, round(float(label), 3), ha="center", va="center")
        fig.colorbar(im, cax=cax)
        fig.savefig("/home/tobi/Schreibtisch/test.pdf")
