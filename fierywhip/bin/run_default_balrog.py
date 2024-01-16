#!/usr/bin/env python3

from fierywhip.utils.default_morgoth import RunMorgoth, RunEffAreaMorgoth
from fierywhip.frameworks.grbs import GRBList, GRB
from threeML.minimizer.minimization import FitFailed
import pandas as pd
import os
import logging
import sys

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
        default(already_run)
    else:
        for g in grb_selection:
            logging.info(f"This is the grb{g}")
            grb = GRB(name=g)
            rm = RunEffAreaMorgoth(
                    grb,
                    use_eff_area=False,
                    det_sel_mode="max_sig_triplets",
                    spectrum="cpl",
                    max_trigger_duration=22,
                )

            rm.run_fit()


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
                        det_sel_mode="triplets",
                        spectrum="cpl",
                        max_trigger_duration=22,
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
                    det_sel_mode="triplets",
                    spectrum="cpl",
                    max_trigger_duration=22,
                )
                rm.run_fit()
            except (RuntimeError, FitFailed, IndexError):
                pass
