#!/usr/bin/env python3

import sys
import warnings

from fierywhip.utils.eff_area_morgoth import MultinestFitTrigdatEffArea

warnings.simplefilter("ignore")

try:
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size > 1:
        using_mpi = True

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        using_mpi = False
except:
    using_mpi = False

grb_name = sys.argv[1]
version = sys.argv[2]
trigdat_file = sys.argv[3]
bkg_fit_yaml_file = sys.argv[4]
time_selection_yaml_file = sys.argv[5]
data_type = sys.argv[6]
grb_file = sys.argv[7]
# get fit object

multinest_fit = MultinestFitTrigdatEffArea(
    grb=None,
    grb_name=grb_name,
    version=version,
    trigdat_file=trigdat_file,
    bkg_fit_yaml_file=bkg_fit_yaml_file,
    time_selection_yaml_file=time_selection_yaml_file,
    grb_file=grb_file,
)
multinest_fit.fit()
multinest_fit.save_fit_result()
multinest_fit.create_spectrum_plot()
multinest_fit.move_chains_dir()
