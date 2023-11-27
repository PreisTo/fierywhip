#!/usr/bin/env python3

from fierywhip.data.grbs import GRBList, GRB
from fierywhip.data.balrog_localizations import save_df, BalrogLocalization, result_df

if __name__ == "__main__":
    grb_list = GRBList(run_det_sel=False)
    for grb in grb_list.grbs:
        bl = BalrogLocalization(grb, result_df)
        if bl.exists:
            bl.add_row_df()
    save_df(result_df)
