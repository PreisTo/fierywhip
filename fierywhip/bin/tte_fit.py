#!/usr/bin/python

from fierywhip.frameworks.grbs import GRB, GRBList
from fierywhip.model.model import GRBModel
from fierywhip.io.export import Exporter
from threeML.minimizer.minimization import FitFailed
from fierywhip.model.tte_individual_norm import GRBModelIndividualNorm
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

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
    if rank == 0:
        grb_list = GRBList(run_det_sel=False)
    else:
        print(f"Hello i am rank {rank}")
        grb_list = None
    grb_list = comm.bcast(grb_list,root = 0)
    comm.Barrier()
    for grb in grb_list.grbs:
        model = GRBModelIndividualNorm(grb)
        exporter = Exporter(model)
        exporter.export_yaml()


if __name__ == "__main__":
    run_individual_norms()
