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
from threeML.utils.data_builders.fermi.gbm_data import GBMTTEFile
from threeML.utils.data_builders.time_series_builder import TimeSeriesBuilder
from threeML.utils.time_series.event_list import EventListWithDeadTime
from threeML.utils.spectrum.binned_spectrum import BinnedSpectrumWithDispersion
from astropy.stats import bayesian_blocks
from threeML.plugins.OGIPLike import OGIPLike
from threeML import *
from gbm_drm_gen.io.balrog_like import BALROGLike
from gbm_drm_gen.io.balrog_drm import BALROG_DRM
from gbm_drm_gen.drmgen_tte import DRMGenTTE
import os
from mpi4py import MPI
import numpy as np
import yaml
import matplotlib.pyplot as plt
from effarea.utils.swift import check_swift
from effarea.io.downloading import download_tte_file, download_cspec_file
from gbmbkgpy.io.downloading import download_trigdata_file

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
        self.get_swift()
        self.timeselection()
        self.bkg_fitting()
        self._setup_model()
        # TODO
        # bkg fitting TTE and storing
        # https://github.com/PreisTo/morgoth/blob/master/morgoth/auto_loc/utils/fit.py#L536

    def download_files(self):
        """
        Downloading TTE and CSPEC files from FTP
        """
        self.tte_files = {}
        self.cspec_files = {}
        for d in lu:
            self.tte_files[d] = download_tte_file(self.grb, d)
            self.cspec_files[d] = download_cspec_file(self.grb, d)

    def timeselection(self):
        """
        get active time using morgoths TimeSelectionBB
        """

        self.trigdat = download_trigdata_file(f"bn{self.grb.strip('GRB')}")

        self.tsbb = TimeSelectionBB(self.grb, self.trigdat, fine=True)

    def get_swift(self):
        """ """
        swift_grb, swift_position = check_swift(self.grb, self.grb_time)
        assert swift_grb is not None, "No conciding Swift GRB found"
        assert swift_position is not None, "Only BAT localization available"
        self._swift_grb_dict = swift_grb
        self.grb_position = swift_position

    def bkg_fitting(self):
        """
        Fitting the TTE Background and creating the Plugins
        """
        self._timeseries = {}
        self._responses = {}
        for d in lu:
            print(f"Calculating Response for {d}")
            response = BALROG_DRM(
                DRMGenTTE(
                    tte_file=self.tte_files[d],
                    trigdat=self.trigdat,
                    mat_type=2,
                    cspecfile=self.cspec_files[d],
                ),
                self.grb_position.ra,
                self.grb_position.dec,
            )
            self._responses[d] = response
            tte_file = GBMTTEFile(self.tte_files[d])
            event_list = EventListWithDeadTime(
                arrival_times=tte_file.arrival_times - tte_file.trigger_time,
                measurement=tte_file.energies,
                n_channels=tte_file.n_channels,
                start_time=tte_file.tstart - tte_file.trigger_time,
                stop_time=tte_file.tstop - tte_file.trigger_time,
                dead_time=tte_file.deadtime,
                first_channel=0,
                instrument=tte_file.det_name,
                mission=tte_file.mission,
                verbose=True,
            )
            ts = TimeSeriesBuilder(
                d,
                event_list,
                response=response,
                poly_order=-1,
                unbinned=False,
                verbose=True,
                container_type=BinnedSpectrumWithDispersion,
            )
            ts.set_background_interval(
                tsbb.background_time_neg, tsbb.background_time_pos
            )
            ts.set_active_time_interval(tsbb.active_time)
            self._timeseries[d] = ts
        response_time = tsbb.stop_trigger - tsbb.start_trigger
        spectrum_likes = []
        for d in lu:
            if self._timeseries[d].name not in ("b0", "b1"):
                spectrum_like = self._timeseries[d].to_spectrumlike()
                spectrum_like.set_active_measurement("8.1-700")
                spectrum_likes.append(spectrum_like)
            else:
                spectrum_like = self._timeseries[d].to_spectrumlike()
                spectrum_like.set_active_measurement("350-25000")
                spectrum_likes.append(spectrum_like)
        balrog_likes = []
        for i, d in enumerate(lu):
            balrog_likes.append(
                BALROGLike.from_spectrumlike(
                    spectrum_likes[i],
                    response_time,
                    self._responses[d],
                    free_position=False,
                )
            )
        self._data_list = DataList(*balrog_likes)

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

    def _setup_model(self):
        band = Band()
        band.K.prior = Log_uniform_prior(lower_bound=1e-5, upper_bound=1200)
        band.alpha.set_uninformative_prior(Uniform_prior)
        band.xp.prior = Log_uniform_prior(lower_bound=10, upper_bound=1e4)
        band.beta.set_uninformative_prior(Uniform_prior)
        self._model = Model(
            PointSource(
                "GRB", self.grb_position.ra, self.grb_position.dec, spectral_shape=band
            )
        )

    def fit(self):
        self._bayes = BayesianAnalysis(self._model, self._data_list)
        # wrap for ra angle
        wrap = [0] * len(self._model.free_parameters)
        wrap[0] = 1

        # define temp chain save path
        self._temp_chains_dir = os.path.join(
            os.environ.get("GBMDATA"), "localizing", self.GRB, "TTE_fit"
        )
        chain_path = os.path.join(self._temp_chains_dir, f"chain")

        # Make temp chains folder if it does not exists already
        if not os.path.exists(self._temp_chains_dir):
            os.mkdir(os.path.join(self._temp_chains_dir))

        # use multinest to sample the posterior
        # set main_path+trigger to whatever you want to use

        self._bayes.set_sampler("multinest", share_spectrum=True)
        self._bayes.sampler.setup(
            n_live_points=800, chain_name=chain_path, wrapped_params=wrap, verbose=True
        )
        self._bayes.sample()
        results = self._bayes.results
        fig = results.corner_plot()
        fig.savefig(os.path.join(self._temp_chains_dir), "cplot.pdf")


if __name__ == "__main__":
    GRB = FitTTE("GRB230903724")
    GRB.fit()
