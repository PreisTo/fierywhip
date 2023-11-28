#!/usr/bin/env python3

from fierywhip.io.export import matrix_from_yaml
from fierywhip.io.plot_dest_dist import DetDistPlot
from fierywhip.config.configuration import fierywhip_config
import os

if __name__ == "__main__":
    excludes = [
        "GRB230812790",
        "GRB220310933",
        "GRB210119121",
        "GRB190123513",
        "GRB190515190",
        "GRB200109074",
        "GRB200528436",
        "GRB210308276",
        "GRB211129410",
    ]
    excludes = []
    print(f"These GRBs are excluded: {excludes}")

    matrix = matrix_from_yaml(
        os.path.join(fierywhip_config.default_plot_path, "localizing/results.yml"),
        exclude=excludes,
    )
    plot = DetDistPlot(matrix=matrix)
