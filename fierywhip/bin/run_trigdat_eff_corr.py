#!/usr/bin/python

from fierywhip.config.configuration import fierywhip_config
from fierywhip.utils.default_morgoth import RunMorgoth, RunEffAreaMorgoth
from fierywhip.frameworks.grbs import GRBList, GRB, GRBInitError
from fierywhip.timeselection.timeselection import TimeSelectionError
from threeML.minimizer.minimization import FitFailed
import pandas as pd
import os
import logging
import sys
import yaml


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
    logging.getLogger().setLevel(logging.DEBUG)
    grb_selection, force, morgoth_config = argv_parsing()

    fierywhip_config.update_config(morgoth_config)
    logging.info(fierywhip_config.config)
    if grb_selection is None:
        logging.info("No GRBs passed as argument - will do my usual thing")
        default(force=force)
    else:
        run_selection(grb_selection, force=force)
