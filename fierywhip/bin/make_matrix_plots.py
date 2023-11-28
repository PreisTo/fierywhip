#!/usr/bin/env python3

from fierywhip.io.export import matrix_from_yaml
from fierywhip.io.plotting.plot_dest_dist import DetDistPlot
from fierywhip.config.configuration import fierywhip_config
import os

if __name__ == "__main__":
    excludes = []
    print(f"These GRBs are excluded: {excludes}")

    matrix = matrix_from_yaml(
        os.path.join(fierywhip_config.default_plot_path, "localizing/results.yml"),
        exclude=excludes,
    )
    plot = DetDistPlot(matrix=matrix)
