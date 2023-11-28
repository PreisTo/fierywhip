#!/usr/bin/env python3

from fierywhip.utils.default_morgoth import RunMorgoth
from fierywhip.frameworks.grbs import GRBList, GRB
import pandas as pd
import os

if __name__ == "__main__":
    try:
        already_run = pd.read_csv(
            os.path.join(os.environ.get("GBM_TRIGGER_DATA_DIR"), "morgoth_results.csv"),
            index_col=None,
        )
    except FileNotFoundError:
        already_run = None
    grb_list = GRBList(run_det_sel=False, check_finished=False, testing=50)
    for g in grb_list.grbs:
        print(f"Checking {g.name}")
        if already_run is not None:
            if g.name not in already_run["grb"]:
                print(f"Starting Morgoth for {g.name}")
                rm = RunMorgoth(g)
        else:
            print(f"Starting Morgoth for {g.name}")
            rm = RunMorgoth(g)
