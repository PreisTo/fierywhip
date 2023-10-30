#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


class DetDistPlot:
    def __init__(self, result_path="/home/tobi/Schreibtisch/localizing/det_matrix.npy"):
        self._matrix = np.load(result_path, allow_pickle=True)

        fig, ax = plt.subplots(1, figsize=(20, 20))
        cax = fig.add_axes([0.1, 0.9, 0.6, 0.05])
        matrix = self._matrix[:, :, 0]
        print(matrix)
        blank = np.empty((12, 12))
        for i in range(12):
            for j in range(12):
                blank[i, j] = np.nan_to_num(np.mean(matrix[i, j]))
        print(blank)
        im = ax.imshow(blank, cmap="coolwarm")
        for (j, i), label in np.ndenumerate(blank):
            ax.text(i, j, round(float(label), 3), ha="center", va="center")
        ax.axvline(x=5.5, color="black")
        ax.axhline(y=5.5, color="black")
        ax.set_title(
            "Rel. Normalization of dets - max angle 60deg, max angle norm det 20deg"
        )
        fig.colorbar(im, cax=cax, orientation="horizontal")
        fig.savefig("/home/tobi/Schreibtisch/test.pdf")
