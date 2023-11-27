#!/usr/bin/env python3

from fierywhip.data.grbs import GRBList, GRB
from fierywhip.data.balrog_localizations import save_df, BalrogLocalization, result_df
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
import pandas as pd

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if __name__ == "__main__":
    grb_list = GRBList(run_det_sel=False)
    comm.Barrier()
    if rank == 0:
        size_per_rank = len(grb_list.grbs) // size
        rest = len(grb_list.grbs) % size
    else:
        size_per_rank = None
        rest = None
    size_per_rank = comm.bcast(size_per_rank, root=0)
    start_rank = rank * size_per_rank
    stop_rank = start_rank + size_per_rank
    if rank == size - 1:
        stop_rank = len(grb_list.grbs)
    for grb in grb_list.grbs[start_rank:stop_rank]:
        bl = BalrogLocalization(grb, result_df)
        if bl.exists:
            bl.add_row_df()
    result_df = comm.gather(result_df, root=0)

    if rank == 0:
        result_df = pd.concat(result_df)
        save_df(result_df)

        # plotting

        fig, ax = plt.subplots(1)
        bins = np.geomspace(1e-1, 360, 25)
        ax.hist(
            [
                result_df["separation"],
                result_df["balrog_1sigma"],
                result_df["balrog_2sigma"],
            ],
            bins=bins,
            alpha=0.4,
            histtype="stepfilled",
            label=[
                "Separation to Swift/IPN position",
                "BALROG 1 sigma",
                "BALROG 2 sigma",
            ],
        )
        # ax.hist(
        #    result_df["balrog_1sigma"],
        #    bins=bins,
        #    density=True,
        #    alpha=0.4,
        #    label="BALROG 1 sigma",
        # )
        # ax.hist(
        #    result_df["balrog_2sigma"],
        #    bins=bins,
        #    density=True,
        #    alpha=0.4,
        #    label="BALROG 2 sigma",
        # )
        ax.legend()
        ax.set_xscale("log")
        ax.set_title(f"Error Comparison for {len(result_df['balrog_1sigma'])} GRBs")
        fig.tight_layout()
        fig.savefig("/home/tobi/Schreibtisch/comparison.pdf")

        print("\nAnalysis:")
        inside_1sig = len(result_df["balrog_1sigma"] <= result_df["separation"]) / len(
            result_df["balrog_1sigma"]
        )
        inside_2sig = len(result_df["balrog_2sigma"] <= result_df["separation"]) / len(
            result_df["balrog_2sigma"]
        )
        print(inside_1sig, inside_2sig)
    MPI.Finalize()
