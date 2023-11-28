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
    if False:
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
    else:
        result_df = None
    if rank == 0:
        if result_df is None:
            result_df = pd.read_csv(
                "/home/tobi/data/GBMDATA/localizing/comparison/comparison.csv"
            )
        else:
            result_df = pd.concat(result_df)
            save_df(result_df)

        # plotting
        keys = ["balrog_1sigma", "balrog_2sigma"]
        for key in keys:
            plt.close("all")
            fig, ax = plt.subplots(1)
            bins = np.linspace(0, 21, 25)  # np.geomspace(1e-1, 360, 25)
            ax.hist(
                result_df["separation"],
                bins=bins,
                alpha=0.4,
                histtype="stepfilled",
                label="Separation to Swift/IPN position",
            )
            ax.hist(
                result_df[key],
                bins=bins,
                alpha=0.4,
                label=key,
            )

            ax.legend()
            ax.set_xlim(0, 20)
            ax.set_title(
                f"Error Comparison for {key} using {len(result_df['balrog_1sigma'])} GRBs"
            )
            fig.tight_layout()
            fig.savefig(f"/home/tobi/Schreibtisch/comparison_{key}.pdf")

        print("\nAnalysis:")

        def calc_percentage(sys):
            inside_1sig = np.sum(
                result_df["balrog_1sigma"] + sys >= result_df["separation"]
            ) / np.sum(result_df["balrog_1sigma"] != np.nan)
            inside_2sig = np.sum(
                result_df["balrog_2sigma"] + sys >= result_df["separation"]
            ) / np.sum(result_df["balrog_2sigma"] != np.nan)
            return inside_1sig, inside_2sig

        fig, ax = plt.subplots(1)
        steps = 100
        upper_lim = 90
        sys_errors = np.geomspace(1e-2, upper_lim, steps)
        percentage = np.zeros((2, steps), dtype=float)
        for i in range(steps):
            res = calc_percentage(sys_errors[i])
            percentage[0, i] = res[0]
            percentage[1, i] = res[1]
        for i in range(2):
            ax.plot(
                sys_errors, percentage[i], label=["1sigma + sys", "2sigma + sys"][i]
            )
        ax.set_xlabel("Sys Error [deg]")
        ax.set_ylabel("Percentage")
        ax.hlines(
            0.68, 0, upper_lim, linestyles="dashed", label=r"$1\:\sigma$", color="black"
        )
        ax.hlines(
            0.95, 0, upper_lim, linestyles="dotted", label=r"$2\:\sigma$", color="black"
        )
        ax.set_xscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig("/home/tobi/Schreibtisch/sys_error.pdf")
    MPI.Finalize()
