#!/usr/bin/env python3
from fierywhip.config.configuration import fierywhip_config
from fierywhip.utils.default_morgoth import RunMorgoth, RunEffAreaMorgoth
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
        os.environ.get("GBM_TRIGGER_DATA_DIR"),
        grb_name,
        "trigdat/v00/",
        "trigdat_v00_loc_results.fits",
    )
    if os.path.exists(path) and os.path.isfile(path):
        return True
    else:
        return False


def check_exclude(grb: str) -> bool:
    """
    Check if we already run the pipeline for this GRB
    or shall be exclude by manual decision

    :param grb: name of the grb
    :type grb: str

    :returns: bool
    """

    excludes = []

    logging.debug(f"Checking {grb}")
    if os.path.exists(
        os.path.join(os.environ.get("GBM_TRIGGER_DATA_DIR"), "morgoth_results.csv")
    ):
        already_run = pd.read_csv(
            os.path.join(os.environ.get("GBM_TRIGGER_DATA_DIR"), "morgoth_results.csv"),
        )
    else:
        already_run = None

    if already_run is not None:
        if grb in list(already_run["grb"]) or grb in excludes:
            if check_grb_fit_result(grb):
                return False
            if grb in excludes:
                return False
        else:
            return True
    return True


def run_morgoth(grb):
    """
    Runs morgoth for a passed GRB object with the settings
    """
    rm = RunEffAreaMorgoth(
        grb,
        use_eff_area=fierywhip_config.config.eff_area_correction.use_eff_corr,
        det_sel_mode=fierywhip_config.config.det_sel.mode,
        spectrum="cpl",
        max_trigger_duration=fierywhip_config.config.timeselection.max_trigger_duration,
    )
    rm.run_fit()


def default(force):
    """
    Default way to run morgoth/balrog for trigdat, when no explicit function
    supplied
    """
    grb_list = GRBList(
        run_det_sel=False, check_finished=False, testing=False, reverse=False
    )
    logging.info(f"We will be running Morgoth for {len(grb_list.grbs)} GRBs")

    for g in grb_list.grbs:
        if check_exclude(g.name) or force:
            logging.info(f"Starting Morgoth for {g.name}")
            try:
                run_morgoth(g)
            except FitFailed:
                logging.info("Bkg Fit Failed")
        else:
            logging.info(f"Skipping Morgoth for {g.name}")


def run_selection(grb_selection, force):
    """
    Run Morgoth for a selection of grbs

    :param grb_selection: list with grb names
    :type grb_selection: list
    """
    for g in grb_selection:
        logging.debug(f"This is the grb {g}")
        if not check_exclude(g) or force:
            try:
                grb = GRB(name=g)
            except AttributeError:
                logging.info(
                    "No Swift Position, but no worries we will set it to ra = dec = 0 deg"
                )
                grb = GRB(name=g, ra=0, dec=0)
            try:
                run_morgoth(grb)
            except (RuntimeError, FitFailed, IndexError):
                pass
        else:
            logging.info(f"{g} was already run")


def argv_parsing():
    """
    Check the passed args

    flag -f: specify file with comma split grb names
    or pass grb comma split grbs ! caution no whitespace

    :returns: list with grb names
    """
    # TODO usage for config!!!
    selection = None
    morgoth_config = {}
    force_run = False
    if len(sys.argv) > 1:
        if "-c" in sys.argv:
            config_index = sys.argv.index("-c") + 1
            with open(sys.argv[config_index], "r") as f:
                morgoth_config = yaml.safe_load(f)
        if "-f" in sys.argv:
            file_index = sys.argv.index("-f") + 1
            with open(sys.argv[file_index], "r") as f:
                selection = f.read().split(",")
        if "--force" in sys.argv:
            force_run = True
            logging.info("FORCE: We will not check if the GRB has already been run")
        else:
            force_run = False
        if "-g" in sys.argv:
            grb_index = sys.argv.index("-g") + 1
            selection = sys.argv[grb_index].split(",")
    return selection, force_run, morgoth_config


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    grb_selection, force, morgoth_config = argv_parsing()

    fierywhip_config.update_config(morgoth_config)
    logging.info(fierywhip_config.config)
    if grb_selection is None:
        logging.info("No GRBs passed as argument - will do my usual thing")
        default(force=force)
    else:
        run_selection(grb_selection, force=force)
