#!/usr/bin/env python3

from fierywhip.utils.default_morgoth import RunMorgoth
from fierywhip.data.grbs import GRBList, GRB
import pandas as pd
import os

if __name__ == "__main__":
    try:
        already_run = pd.read_csv(
            os.path.join(os.environ.get("GBM_TRIGGER_DATA_DIR"), "morgoth_results.csv")
        )
    except FileNotFoundError:
        already_run = None
    grb_list = GRBList(run_det_sel=False, testing=True)
    for g in grb_list.grbs:
        if already_run is not None:
            if g.name in already_run["grb"]:
                print(f"Starting Morgoth for {g.name}")
                rm = RunMorgoth(g)
        else:
            print(f"Starting Morgoth for {g.name}")
            rm = RunMorgoth(g)
