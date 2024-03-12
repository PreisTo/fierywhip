#!/usr/bin/env python3
from fierywhip.utils.default_morgoth import RunMorgoth, RunEffAreaMorgoth
from fierywhip.config.configuration import fierywhip_config
from fierywhip.frameworks.grbs import GRBList, GRB
from threeML.minimizer.minimization import FitFailed
import pandas as pd
import os
import logging
import sys
import yaml


def check_grb_fit_result(grb_name):
    """
    Check if the .fits file created after the fit exists in the
    default path for a given grb

    :param grb_name: name of grb
    :type grb_name: str

    :returns: bool True if exists and False if not
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


def check_exclude(grb: str) -> bool:
    """
    Check if we already run the pipeline for this GRB
    or shall be exclude by manual decision

    :param grb: name of the grb
    :type grb: str

    :returns: bool
    """

    excludes = []

    logging.debug(f"Checking {g.name}")
    if os.path.exists(
        os.path.join(os.environ.get("GBM_TRIGGER_DATA_DIR"), "morgoth_results.csv")
    ):
        already_run = pd.read_csv(
            os.path.join(os.environ.get("GBM_TRIGGER_DATA_DIR"), "morgoth_results.csv"),
        )
    else:
        already_run = None

    if already_run is not None:
        if grb not in list(already_run["grb"]) and grb not in excludes:
            if not check_grb_fit_result(grb):
                return False

    return True


def run_morgoth(grb):
    """
    Runs morgoth for a passed GRB object with the settings
    """
    rm = RunEffAreaMorgoth(
        grb,
        use_eff_area=False,
        det_sel_mode="max_sig_triplets",
        spectrum="cpl",
        max_trigger_duration=30,
    )
    rm.run_morgoth()


def default():
    """
    Default way to run morgoth/balrog for trigdat, when no explicit function
    supplied
    """
    grb_list = GRBList(
        run_det_sel=False, check_finished=False, testing=False, reverse=False
    )
    logging.info(f"We will be running Morgoth for {len(grb_list.grbs)} GRBs")

    for g in grb_list.grbs:
        if not check_exclude(g.name):
            logging.info(f"Starting Morgoth for {g.name}")
            try:
                run_morgoth(g)
            except (RuntimeError, FitFailed, IndexError):
                pass
        else:
            logging.info(f"Skipping Morgoth for {g.name}")


def run_selection(grb_selection):
    """
    Run Morgoth for a selection of grbs

    :param grb_selection: list with grb names
    :type grb_selection: list
    """
    for g in grb_selection:
        logging.info(f"This is the grb{g}")
        if not check_exclude(g):
            try:
                grb = GRB(name=g)
            except AttributeERror:
                logging.info(
                    "No Swift Position, but no worries we will set it to ra = dec = 0 deg"
                )
                grb = GRB(name=g, ra=0, dec=0)
            try:
                run_morgoth(grb)
            except (RuntimeError, FitFailed, IndexError):
                pass


def argv_parsing():
    """
    Check the passed args

    flag -f: specify file with comma split grb names
    or pass grb comma split grbs ! caution no whitespace

    :returns: list with grb names
    """
    # TODO usage for config!!!
    grb_selection = None
    morgoth_config = {}
    if len(sys.argv) > 1:
        if "-c" in sys.argv:
            config_index = sys.argv.index("-c") + 1
            with open(sys.argv[config_index], "r") as f:
                morgoth_config = yaml.safe_load(f)
        if "-f" in sys.argv:
            file_index = sys.argv.index("-f") + 1
            with open(sys.argv[file_index], "r") as f:
                grb_selection = f.read().split(",")
        else:
            grb_selection = sys.argv[1].split(",")
    return grb_selection


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    grb_selection = argv_parsing()
    if grb_selection is None:
        logging.info("No GRBs passed as argument - will do my usual thing")
        default(already_run)
    else:
        run_selection(grb_selection)
