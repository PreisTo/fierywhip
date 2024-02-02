#!/usr/bin/env python3

from fierywhip.utils.default_morgoth import RunMorgoth, RunEffAreaMorgoth
from fierywhip.frameworks.grbs import GRBList, GRB
from threeML.minimizer.minimization import FitFailed
import pandas as pd
import os
import logging
import sys


def default(already_run):
    excludes = []
    grb_list = GRBList(
        run_det_sel=False, check_finished=False, testing=False, reverse=False
    )
    logging.info(f"We will be running Morgoth for {len(grb_list.grbs)} GRBs")

    for g in grb_list.grbs:
        logging.debug(f"Checking {g.name}")
        if already_run is not None:
            if (
                g.name not in list(already_run["grb"])
                and g.name not in excludes
                and not os.path.exists(
                    os.path.join(os.environ.get("GBM_TRIGGER_DATA_DIR"), g.name)
                )
            ):
                logging.info(f"Starting Morgoth for {g.name}")
                try:
                    # rm = RunMorgoth(g,spectrum = "pl")
                    rm = RunEffAreaMorgoth(
                        g,
                        use_eff_area=False,
                        det_sel_mode="huntsville",
                        spectrum="cpl",
                        max_trigger_duration=30,
                    )
                    rm.run_fit()
                except (RuntimeError, FitFailed, IndexError):
                    pass
            else:
                logging.info(f"Skipping Morgoth for {g.name}")
        else:
            logging.info(f"Starting Morgoth for {g.name}")
            try:
                # rm = RunMorgoth(g,spectrum = "pl")
                rm = RunEffAreaMorgoth(
                    g,
                    use_eff_area=False,
                    det_sel_mode="huntsville",
                    spectrum="cpl",
                    max_trigger_duration=30,
                )
                rm.run_fit()
            except (RuntimeError, FitFailed, IndexError, NotImplementedError):
                pass
def check_grb_fit_result(grb_name):
    path = os.path.join(os.environ.get("GBMDATA"), grb_name, "trigdat/v00/","trigdat_v00_loc_results.fits")
    if os.path.exists(path) and os.path.isfile(path):
        return False
    else:
        return True

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    if len(sys.argv) > 1:
        grb_selection = sys.argv[1].split(",")
    else:
        grb_selection = None
    if os.path.exists(
        os.path.join(os.environ.get("GBM_TRIGGER_DATA_DIR"), "morgoth_results.csv")
    ):
        already_run = pd.read_csv(
            os.path.join(os.environ.get("GBM_TRIGGER_DATA_DIR"), "morgoth_results.csv"),
        )
    else:
        already_run = None
    if grb_selection is None:
        logging.info("No GRBs passed as argument - will do my usual thing")
        default(already_run)
    else:
        for g in grb_selection:
            logging.info(f"This is the grb{g}")
            run = False
            try:
                if check_grb_fit_result(g):
                    grb = GRB(name=g)
                    run = True
            except AttributeError:
                logging.info(f"No swift position available, will set to ra=0 and dec=0!")
                if check_grb_fit_result(g):
                    grb = GRB(name=g, ra = 0, dec =0,run_det_sel = False)
                    run = True
            if run:
                rm = RunEffAreaMorgoth(
                    grb,
                    use_eff_area=False,
                    det_sel_mode="max_sig_triplets",
                    spectrum="cpl",
                    max_trigger_duration=22,
                )

                rm.run_fit()
