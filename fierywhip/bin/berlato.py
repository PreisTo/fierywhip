#!/usr/bin/python

from fierywhip.frameworks.grbs import GRBList, GRB
from fierywhip.config.configuration import fierywhip_config
import yaml
import sys


def check_args():
    if "-c" in sys.argv:
        with open(sys.argv[sys.argv.index("-c") + 1], "r") as f:
            config_up = yaml.safe_load(f)
        fierywhip_config.update_config(config_up)


if __name__ == "__main__":
    check_args()
    gl = GRBList()
    raise NotImplementedError
