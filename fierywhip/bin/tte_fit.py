#!/usr/bin/python

from fierywhip.frameworks.grbs import GRB, GRBList
from fierywhip.model.model import GRBModel
from fierywhip.io.export import Exporter
from threeML.minimizer.minimization import FitFailed
from fierywhip.model.tte_individual_norm import GRBModelIndividualNorm


def old():
    grb_list = GRBList()
    for grb in grb_list.grbs:
        print(f"Started for {grb.name}\n\n")
        try:
            model = GRBModel(grb)
            exporter = Exporter(model)
            exporter.export_yaml()
            exporter.export_matrix()
        except (FitFailed, TypeError, IndexError, RuntimeError, FileNotFoundError) as e:
            print(e)


def run_individual_norms():
    grb_list = GRBList()
    for grb in grb_list.grbs:
        model = GRBModelIndividualNorm(grb)
        exporter = Exporter(model)
        exporter.export_yaml()


if __name__ == "__main__":
    run_individual_norms()
