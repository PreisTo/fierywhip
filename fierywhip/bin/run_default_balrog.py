#!/usr/bin/env python3

from fierywhip.utils.default_morgoth import RunMorgoth, RunEffAreaMorgoth
from fierywhip.frameworks.grbs import GRBList, GRB
from threeML.minimizer.minimization import FitFailed
import pandas as pd
import os
import logging

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    if os.path.exists(
        os.path.join(os.environ.get("GBM_TRIGGER_DATA_DIR"), "morgoth_results.csv")
    ):
        already_run = pd.read_csv(
            os.path.join(os.environ.get("GBM_TRIGGER_DATA_DIR"), "morgoth_results.csv"),
        )
    else:
        already_run = None
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
                        det_sel_mode="bgo_sides_no_bgo",
                        spectrum="cpl",
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
                    det_sel_mode="bgo_sides_no_bgo",
                    spectrum="cpl",
                )
                rm.run_fit()
            except (RuntimeError, FitFailed, IndexError):
                pass
