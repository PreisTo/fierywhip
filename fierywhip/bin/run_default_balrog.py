#!/usr/bin/env python3

from fierywhip.utils.default_morgoth import RunMorgoth, RunEffAreaMorgoth
from fierywhip.frameworks.grbs import GRBList, GRB
from threeML.minimizer.minimization import FitFailed
import pandas as pd
import os
import logging
import sys
import yaml


def default(already_run):
    """
    Default way to run morgoth/balrog for trigdat, when no explicit function
    supplied
    """
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
                    rm = RunEffAreaMorgoth(
                        g,
                        use_eff_area=False,
                        det_sel_mode="max_sig_triplets",
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
                    det_sel_mode="max_sig_triplets",
                    spectrum="cpl",
                    max_trigger_duration=30,
                    det_sel_mode="bgo_sides_no_bgo",
                    spectrum="cpl",
                )
                rm.run_fit()
            except (RuntimeError, FitFailed, IndexError, NotImplementedError):
                pass


def check_grb_fit_result(grb_name):
    """
    Check if the .fits file created after the fit exists in the
    default path for a given grb

    :param grb_name: name of grb
    :type grb_name: str

    :return: bool True if exists and False if not
    """
    path = os.path.join(
        os.environ.get("GBMDATA"),
        grb_name,
        "trigdat/v00/",
        "trigdat_v00_loc_results.fits",
    )
    if os.path.exists(path) and os.path.isfile(path):
        return False
    else:
        return True


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    if len(sys.argv) > 1:
        timeselection_preload = False
        if sys.argv[1] != "-f":
            grb_selection = sys.argv[1].split(",")
        else:
            with open(sys.argv[2], "r") as f:
                grb_selection = f.read().split(",")
        if "-t" in sys.argv:
            flag_id = sys.argv.index("-t")
            ts = sys.argv[flag_id + 1]
            timeselection_preload = True
    else:
        timeselection_preload = False
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
                logging.info(
                    f"No swift position available, will set to ra=0 and dec=0!"
                )
                if check_grb_fit_result(g):
                    grb = GRB(
                        name=g,
                        ra=0,
                        dec=0,
                        run_det_sel=False,
                        run_ts=~timeselection_preload,
                    )
                    run = True
                if timeselection_preload:
                    with open(ts, "r") as f:
                        ts_dict = yaml.safe_load(f)
                        grb._active_time = f"{ts_dict['active_time']['start']}-{ts_dict['active_time']['start']}"
                        grb._bkg_time = [
                            f"{ts_dict['background_time']['before']['start']}-{ts_dict['background_time']['before']['start']}",
                            f"{ts_dict['background_time']['after']['start']}-{ts_dict['background_time']['after']['start']}",
                        ]
            if run:
                rm = RunEffAreaMorgoth(
                    grb,
                    use_eff_area=False,
                    det_sel_mode="max_sig_triplets",
                    spectrum="cpl",
                    max_trigger_duration=16,
                )

                rm.run_fit()
