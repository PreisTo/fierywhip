#!/usr/bin/python

from fierywhip.frameworks.grbs import GRB, GRBList
from fierywhip.model.model import GRBModel
from fierywhip.io.export import Exporter
from threeML.minimizer.minimization import FitFailed
from fierywhip.model.tte_individual_norm import GRBModelIndividualNorm
import subprocess
import pkg_resources
import os

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
    grb_list = GRBList(run_det_sel=False)
    for grb in grb_list.grbs:
        grb_yaml = grb.save_grb(os.path.join(os.environ.get("GBMDATA"),"dumpy_dump.yml"))
        fit_script = pkg_resources.resource_filename("fierywhip", "utils/tte_fit.py")
        subprocess.check_output("mpiexec -n 8 --bind-to core python
            {fit_script} {grb_yaml}", shell=True, env=os.environ, stdin=subprocess.PIPE)


if __name__ == "__main__":
    run_individual_norms()
