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
from gbmbkgpy.io.downloading import download_trigdata_file, download_gbm_file
import pkg_resources

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


class AlreadyRun(Exception):
    print("Already run")
    pass


class FitTTE:
    def __init__(self, grb, energy_range="8.1-700"):
        self.grb = grb
        assert type(energy_range) == str, "energy_range must be string"
        assert (
            len(energy_range.split("-")) == 2
        ), "energy_range must consist of two floats separated by a -, e.g 8.1-700"
        self.energy_range = energy_range

        self._base_dir = os.path.join(os.environ.get("GBMDATA"), "localizing")
        if not self._already_run_check():
            self._set_grb_time()
            self.download_files()
            self.get_swift()
            self.timeselection()
            self.bkg_fitting()
            self._to_plugin()
            self._setup_model()
        else:
            raise AlreadyRun

    def set_energy_range(self, energy_range):
        self.energy_range = energy_range
        if not self._already_run_check():
            print(f"new energy range set to {self.energy_range}")
            print("Setting new TimeSeries and setting up plugins")
            self._to_plugin()
            self._setup_model()
            self.fit()
            self.save_results()

    def download_files(self):
        """
        Downloading TTE and CSPEC files from FTP
        """
        print("Downloading TTE and CSPEC files")
        self.tte_files = {}
        self.cspec_files = {}
        for d in lu:
            self.tte_files[d] = download_tte_file(self.grb, d)
            self.cspec_files[d] = download_cspec_file(self.grb, d)

    def timeselection(self):
        """
        get active time using morgoths TimeSelectionBB
        """
        print("Starting Timeselection Trigdat")
        trigdat = download_trigdata_file(f"bn{self.grb.strip('GRB')}")

        tsbb = TimeSelectionBB(self.grb, trigdat, fine=True)
        self.trigdat = trigdat
        self.tsbb = tsbb

    def get_swift(self):
        """ """
        print("Getting coinciding Swift GRB")
        swift_grb, swift_position = check_swift(self.grb, self.grb_time)
        assert swift_grb is not None, "No conciding Swift GRB found"
        assert swift_position is not None, "Only BAT localization available"
        self._swift_grb_dict = swift_grb
        self.grb_position = swift_position

    def bkg_fitting(self):
        """
        Fitting the TTE Background and creating the Plugins
        """
        print("Fitting the Background for TTE")
        temp_timeseries = {}
        temp_responses = {}

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
            temp_responses[d] = response
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
                self.tsbb.background_time_neg, self.tsbb.background_time_pos
            )
            ts.set_active_time_interval(self.tsbb.active_time)
            temp_timeseries[d] = ts
        self._timeseries = temp_timeseries
        self._responses = temp_responses

    def _to_plugin(self, fix_correction=None, free_position=False):
        print("Creating BALROG like plugins")
        response_time = self.tsbb.stop_trigger - self.tsbb.start_trigger
        spectrum_likes = []
        for d in lu:
            if self._timeseries[d]._name not in ("b0", "b1"):
                spectrum_like = self._timeseries[d].to_spectrumlike()
                spectrum_like.set_active_measurements(self.energy_range)
                spectrum_likes.append(spectrum_like)
            else:
                spectrum_like = self._timeseries[d].to_spectrumlike()
                spectrum_like.set_active_measurements("350-25000")
                spectrum_likes.append(spectrum_like)
        balrog_likes = []
        for i, d in enumerate(lu):
            bl = BALROGLike.from_spectrumlike(
                spectrum_likes[i],
                response_time,
                self._responses[d],
                free_position=free_position,
            )
            if fix_correction is None:
                if d not in ("b0", "b1"):
                    bl.use_effective_area_correction(0.7, 1.3)
                else:
                    bl.fix_effective_area_correction(1)
            else:
                raise NotImplementedError
            balrog_likes.append(bl)
        self._data_list = DataList(*balrog_likes)

    def _set_grb_time(self):
        """
        sets the grb_time (datetime object)
        """
        print("Setting the GRB time")
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
        print("Setting up the model for the fit")
        band = Band()
        band.K.prior = Log_uniform_prior(lower_bound=1e-5, upper_bound=1200)
        band.alpha.set_uninformative_prior(Uniform_prior)
        band.xp.prior = Log_uniform_prior(lower_bound=10, upper_bound=1e4)
        band.beta.set_uninformative_prior(Uniform_prior)
        self._model = Model(
            PointSource(
                "GRB",
                self.grb_position.ra.deg,
                self.grb_position.dec.deg,
                spectral_shape=band,
            )
        )

    def fit(self):
        print("Starting the Fit")
        self._bayes = BayesianAnalysis(self._model, self._data_list)
        # wrap for ra angle
        wrap = [0] * len(self._model.free_parameters)
        wrap[0] = 1

        # define temp chain save path
        self._temp_chains_dir = os.path.join(
            self._base_dir, self.grb, self.energy_range, "TTE_fit"
        )
        chain_path = os.path.join(self._temp_chains_dir, f"chain")

        # Make temp chains folder if it does not exists already
        if not os.path.exists(self._temp_chains_dir):
            os.makedirs(os.path.join(self._temp_chains_dir))

        # use multinest to sample the posterior
        # set main_path+trigger to whatever you want to use

        self._bayes.set_sampler("multinest", share_spectrum=True)
        self._bayes.sampler.setup(
            n_live_points=800, chain_name=chain_path, wrapped_params=wrap, verbose=True
        )
        self._bayes.sample()
        self.results = self._bayes.results
        fig = self.results.corner_plot()
        fig.savefig(os.path.join(self._temp_chains_dir, "cplot.pdf"))
        plt.close("all")

    def get_separations(self):
        poshist = os.path.join(
            os.environ.get("GBMDATA"),
            "poshist",
            self.grb.strip("GRB")[:-3],
            f"glg_poshist_all_{self.grb.strip('GRB')[:-3]}_v00.fit",
        )
        if not os.path.exists(poshist):
            download_gbm_file(date=self.grb.strip("GRB")[:-3], data_type="poshist")
            print("Done downloading poshist")
        self.interpolator = PositionInterpolator.from_poshist(poshist)

        t0 = time.Time(self.grb_time, format="datetime", scale="utc")
        gbm_time = GBMTime(t0)
        self.gbm = GBM(
            self.interpolator.quaternion(gbm_time.met),
            sc_pos=self.interpolator.sc_pos(gbm_time.met) * u.km,
        )
        sep = self.gbm.get_separation(self.grb_position)
        self.separations = {}
        for d in lu:
            self.separations[d] = float(sep[d])
        return {"separations": self.separations}

    def save_results(self):
        self.results.write_to(
            os.path.join(self._temp_chains_dir, "fit_results.fits"), overwrite=True
        )

        if os.path.exists(os.path.join(self._base_dir, "results.yml")):
            with open(os.path.join(self._base_dir, "results.yml"), "r") as f:
                results_yaml_dict = yaml.safe_load(f)
            if self.grb in results_yaml_dict.keys():
                if self.energy_range in results_yaml_dict[self.grb].keys():
                    temp = results_yaml_dict[self.grb]
                else:
                    temp = {}
            else:
                results_yaml_dict[self.grb] = self.get_separations()
                temp = {}
        else:
            results_yaml_dict = {}
            results_yaml_dict[self.grb] = self.get_separations()
            results_yaml_dict[self.grb][self.energy_range] = {}
            temp = {}
        for d in lu:
            if d not in ("b0", "b1"):
                temp[d] = float(
                    self.results.optimized_model.free_parameters[f"cons_{d}"].value
                )
        results_yaml_dict[self.grb][self.energy_range] = temp
        with open(os.path.join(self._base_dir, "results.yml"), "w+") as f:
            yaml.dump(results_yaml_dict, f)

    def _already_run_check(self):
        with open(os.path.join(self._base_dir, "results.yml"), "r") as f:
            res_dict = yaml.safe_load(f)
        try:
            grb_dict = res_dict[self.grb]
            energy_dict = grb_dict[self.energy_range]
            for d in lu[:2]:
                test = energy_dict[d]
            return True
        except KeyError:
            return False


def get_grbs(csv=pkg_resources.resource_filename("effarea", "data/grbs.txt")):
    """
    returns a list of GRBs with Swift localization
    """
    csv_content = pd.read_csv(csv, sep="\t", index_col=False)
    grbs = csv_content["name"].loc[csv_content["swift_ra"] != None]
    grbs = grbs.to_list()
    return grbs


if __name__ == "__main__":
    bin_start = 10
    bin_stop = 700
    num_selections = 10
    tot_bins = np.geomspace(bin_start, bin_stop)
    bins = np.array_split(tot_bins, num_selections)
    energy_list = [f"{i[0]}-{i[-1]}" for i in bins]
    GRBS = get_grbs()
    for G in GRBS:
        try:
            GRB = FitTTE(G)
            GRB.fit()
            GRB.save_results()
            for energy in energy_list:
                GRB.set_energy_range(energy)
        except e in (ZeroDivisionError, AlreadyRun):
            print("passing")
    # TODO fix MPI
    # TODO OUtput
    # TODO spectrum
    # TODO errors on paramaters
    #
