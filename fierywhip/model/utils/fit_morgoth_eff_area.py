#!/usr/bin/env python3

import sys
import warnings
import logging
from fierywhip.model.eff_area_morgoth import (
    MultinestFitTrigdatMultipleSelections,
    MultinestFitTrigdatEffArea,
)

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
det_sel_mode = sys.argv[6]
use_eff_area = sys.argv[7]
grb_file = sys.argv[8]
spectrum = sys.argv[9]
if len(sys.argv) > 10:
    long_grb = sys.argv[10]
else:
    long_grb = False
# get fit object


logging.info(f"Using spectrum {spectrum}")
if str(long_grb).lower() == "true":
    logging.info(
        f"This is a long GRB (active-time > 10s) - we will use two spectra and responses"
    )
    multinest_fit = MultinestFitTrigdatMultipleSelections(
        grb=None,
        grb_name=grb_name,
        version=version,
        trigdat_file=trigdat_file,
        bkg_fit_yaml_file=bkg_fit_yaml_file,
        time_selection_yaml_file=time_selection_yaml_file,
        grb_file=grb_file,
        det_sel_mode=det_sel_mode,
        use_eff_area=False,
        spectrum=spectrum,
    )
else:
    multinest_fit = MultinestFitTrigdatEffArea(
        grb=None,
        grb_name=grb_name,
        version=version,
        trigdat_file=trigdat_file,
        bkg_fit_yaml_file=bkg_fit_yaml_file,
        time_selection_yaml_file=time_selection_yaml_file,
        grb_file=grb_file,
        det_sel_mode=det_sel_mode,
        use_eff_area=False,
        spectrum=spectrum,
    )
multinest_fit.fit()
multinest_fit.save_fit_result()
multinest_fit.create_spectrum_plot()
multinest_fit.move_chains_dir()
