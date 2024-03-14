#!/usr/bin/python

from fierywhip.frameworks.grbs import GRB, GRBList
from fierywhip.model.model import GRBModel
from fierywhip.io.export import Exporter
from threeML.minimizer.minimization import FitFailed
from fierywhip.model.tte_individual_norm import GRBModelIndividualNorm
from fierywhip.config.configuration import fierywhip_config
import subprocess
import pkg_resources
import os
import yaml
import sys
import logging
from astromodels import *
from astromodels.functions import Gaussian, Log_uniform_prior, Uniform_prior
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def passed_arguments():
    if len(sys.argv) > 1:
        if sys.argv[1] != "-f":
            grb_selection = sys.argv[1].split(",")
        else:
            with open(sys.argv[2], "r") as f:
                grb_selection = f.read().split(",")
    else:
        grb_selection = None
    return grb_selection


def old(ts_path=None):
    selection = passed_arguments()
    if selection is None:
        grb_list = GRBList(run_det_sel=False)
        run_list = grb_list.grbs
    else:
        run_list = []
        for s in selection:
            run_list.append(GRB(name=s.strip("\n"), ra=0, dec=0, run_det_sel=False))
    for grb in run_list:
        if ts_path is not None:
            grb.timeselection_from_yaml(
                os.path.join(ts_path, grb.name, "timeselection.yml")
            )
            grb.detector_selection()
        print(f"Started for {grb.name}\n\n")
        try:
            grb.download_files(dets="all")

            model = GRBModel(
                grb,
                fix_position=fierywhip_config.config.tte.fix_position,
                use_eff_area=fierywhip_config.config.eff_area_correction.use_eff_area,
            )
            if rank == 0:
                exporter = Exporter(model)
        #            exporter.export_yaml()
        #            exporter.export_matrix()
        except (FitFailed, TypeError, IndexError, RuntimeError, FileNotFoundError) as e:
            print(e)


def run_individual_norms(ts_path=None):
    grb_list = GRBList()
    for grb in grb_list.grbs:
        grb.timeselection_from_yaml(
            os.path.join(ts_path, grb.name, "timeselection.yml")
        )
        grb_yaml = grb.save_grb(
            os.path.join(os.environ.get("GBMDATA"), "dumpy_dump.yml")
        )
        grb_yaml = os.path.join(os.environ.get("GBMDATA"), "dumpy_dump.yml")

        fit_script = pkg_resources.resource_filename("fierywhip", "utils/tte_fit.py")
        print(fit_script)
        subprocess.check_output(
            f"{fierywhip_config.config.mpiexec_path} -n {fierywhip_config.config.mulinest_nr_cores} --bind-to core python {fit_script} {grb_yaml}",
            shell=True,
            env=os.environ,
            stdin=subprocess.PIPE,
        )


if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")
    old()
