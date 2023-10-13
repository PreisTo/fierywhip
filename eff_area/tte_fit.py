#!/usr/bin/env python3
from gbmgeometry import PositionInterpolator, GBM
from astropy.coordinates import SkyCoord
from gbmgeometry.utils.gbm_time import GBMTime
import astropy.time as time
import astropy.units as u
import pandas as pd
from datetime import datetime, timedelta
from morgoth.utils.trig_reader import TrigReader
from morgoth.auto_loc.time_selection import TimeSelectionBB
from astromodels.functions import Powerlaw, Cutoff_powerlaw, Band
from astromodels.sources.point_source import PointSource
from astromodels.functions.priors import Log_uniform_prior, Uniform_prior
from astromodels.core.model import Model
from threeML.data_list import DataList
from threeML.bayesian.bayesian_analysis import BayesianAnalysis, BayesianResults
from astropy.stats import bayesian_blocks
from threeML.plugins.OGIPLike import OGIPLike
from threeML import *
from gbm_drm_gen.io.balrog_like import BALROGLike
import os
from mpi4py import MPI
import numpy as np
import yaml
import matplotlib.pyplot as plt
from effarea.run_eff import check_swift
from effarea.io.downloading import download_tte_file, download_cspec_file


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if size > 1:
    using_mpi = True
else:
    using_mpi = False
lu = [
    "n0",
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
    "n7",
    "n8",
    "n9",
    "na",
    "nb",
    "b0",
    "b1",
]


class FitTTE:
    def __init__(self, grb):
        self.grb = grb
        self._set_grb_time()
        self.download_files()
        self.timeselection()

        # TODO
        # bkg fitting TTE and storing
        # https://github.com/PreisTo/morgoth/blob/master/morgoth/auto_loc/utils/fit.py#L536

    def download_files(self):
        """
        Downloading TTE and CSPEC files from FTP
        """
        for d in lu:
            download_tte_file(self.grb, d)
            download_cspec_file(self.grb, d)

    def timeselection(self):
        """
        get active time using morgoths TimeSelectionBB
        """

        trigdat = download_trigdata_file(f"bn{self.grb.strip('GRB')}")

        self.tsbb = TimeSelectionBB(self.grb, trigdat, fine=True)

    def get_swift(self):
        """ """
        swift_grb = check_swift(self.grb, self.grb_time)
        if swift_grb is not None:
            swift_grb = swift_grb.to_dict()
            poshist = os.path.join(os.environ.get("GBMDATA"), "poshist")

    def _set_grb_time(self):
        """
        sets the grb_time (datetime object)
        """
        total_seconds = 24 * 60 * 60
        trigger = self.grb.strip("GRB")
        year = int(f"20{trigger[:2]}")
        month = int(trigger[2:4])
        day = int(trigger[4:6])
        frac = int(trigger[6:])
        dt = datetime(year, month, day) + timedelta(
            seconds=float(total_seconds * frac / 1000)
        )
        self.grb_time = dt


if __name__ == "__main__":
    GRB = FitTTE("GRB230903724")
