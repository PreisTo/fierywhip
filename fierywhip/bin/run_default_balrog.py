#!/usr/bin/env python3

from fierywhip.utils.default_morgoth import RunMorgoth, RunEffAreaMorgoth
from fierywhip.frameworks.grbs import GRBList, GRB
from threeML.minimizer.minimization import FitFailed
import pandas as pd
import os

if __name__ == "__main__":
    if os.path.exists(
        os.path.join(os.environ.get("GBM_TRIGGER_DATA_DIR"), "morgoth_results.csv")
    ):
        already_run = pd.read_csv(
            os.path.join(os.environ.get("GBM_TRIGGER_DATA_DIR"), "morgoth_results.csv"),
        )
    else:
        already_run = None
    excludes = [
        "GRB091208410",
        "GRB110108977",
        "GRB110709642",
        "GRB160325291",
        "GRB110102788",
        "GRB121128212",
        "GRB130306991",
        "GRB130610133",
        "GRB150309958",
        "GRB150430015",
        "GRB161004964",
        "GRB180314030",
        "GRB180404091",
        "GRB200227306",
        "GRB201216963",
        "GRB220107615",
    ]
    grb_list = GRBList(run_det_sel=False, check_finished=False, testing=False)
    print(f"We will be running Morgoth for {len(grb_list.grbs)} GRBs")
    for g in grb_list.grbs:
        print(f"Checking {g.name}")
        if already_run is not None:
            if g.name not in list(already_run["grb"]) and g.name not in excludes:
                print(f"Starting Morgoth for {g.name}")
                try:
                    # rm = RunMorgoth(g)
                    rm = RunEffAreaMorgoth(
                        g,
                        use_eff_area=False,
                        det_sel_mode="max_sig",
                    )
                    rm.run_fit()
                except (RuntimeError, FitFailed, IndexError):
                    pass
            else:
                print(f"Skipping Morgoth for {g.name}")
        else:
            print(f"Starting Morgoth for {g.name}")
            try:
                # rm = RunMorgoth(g)
                rm = RunEffAreaMorgoth(g, use_eff_area=False, det_sel_mode="max_sig")
                rm.run_fit()
            except (RuntimeError, FitFailed, IndexError):
                pass
