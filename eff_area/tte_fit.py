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
from threeML.io.plotting.post_process_data_plots import display_spectrum_model_counts
from astropy.stats import bayesian_blocks
from threeML.plugins.OGIPLike import OGIPLike
from threeML import *
from threeML.minimizer.minimization import FitFailed
from gbm_drm_gen.io.balrog_like import BALROGLike
from gbm_drm_gen.io.balrog_drm import BALROG_DRM
from gbm_drm_gen.drmgen_tte import DRMGenTTE
from effarea.io.balrog_like_add import BALROGLikePositionPrior
import os
from mpi4py import MPI
import numpy as np
import yaml
import matplotlib.pyplot as plt
from effarea.utils.swift import check_swift
from effarea.utils.detectors import calc_angular_incident
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
    def __init__(self, grb, energy_range="10-900", fix_position=True):
        self._results_loaded = False
        self._fix_position = fix_position
        self.grb = grb
        assert type(energy_range) == str, "energy_range must be string"
        assert (
            len(energy_range.split("-")) == 2
        ), "energy_range must consist of two floats separated by a -, e.g 8.1-700"
        self.energy_range = energy_range
        print(f"Running {self.grb}")
        self._base_dir = os.path.join(os.environ.get("GBMDATA"), "localizing")
        if not self._already_run_check():
            self._set_grb_time()
            self.download_files()
            self.get_swift()
            runable = self.check_normalizing_det()
            if not runable:
                raise (AlreadyRun("n0 and n6 have a too high separation"))
            self.tsbb, temp_the_tempitemp, self.trigdat = self.timeselection()
            self._normalizing_det = self.calc_separations()
            if self._normalizing_det is None:
                raise (AlreadyRun(f"No Normalizing Detector available for {self.grb}"))
            self.bkg_fitting()
            self._to_plugin()
            self._setup_model()
        else:
            raise AlreadyRun

    def check_normalizing_det(self):
        grb = self.grb.strip("GRB")
        if len(grb) == 9:
            g = f"GRB{grb}"
            bn = f"bn{grb}"
            year = int(f"20{grb[:2]}")
            month = int(grb[2:4])
            day = int(grb[4:6])
            frac = int(grb[6:])
        else:
            g = f"GRB0{grb}"
            bn = f"bn0{grb}"
            year = int(f"200{grb[0]}")
            month = int(grb[1:3])
            day = int(grb[3:5])
            frac = int(grb[5:])
        try:
            trigdat = os.path.join(
                os.environ.get("GBMDATA"),
                "trigdat",
                str(year),
                f"glg_trigdat_all_{bn}_v00.fit",
            )
            pi = PositionInterpolator.from_trigdat(trigdat)
            gbm = GBM(pi.quaternion(0), pi.sc_pos(0))
            seps = gbm.get_separation(self.grb_position)
            """if seps["n0"] <= 30:
                return True
            elif seps["n6"] <= 30:
                return True
            else:
                return False"""
            smaller_60 = []
            for d, s in seps.items():
                if s <= 60:
                    smaller_60.append(d)
            if len(smaller_60) < 3:
                return False
            else:
                possible = {}
                for d, s in seps.items():
                    if s <= 60:
                        possible[s] = d
                res = list(sorted(possible).values()[:3])
                if sep["b0"] <= sep["b1"]:
                    res.append("b0")
                else:
                    res.append("b1")
                self._use_dets = res
                return True
        except FileNotFoundError:
            return False

    def set_energy_range(self, energy_range, optimized_model=None):
        self.energy_range = energy_range
        if not self._already_run_check():
            print(f"new energy range set to {self.energy_range}")
            print("Setting new TimeSeries and setting up plugins")
            self._to_plugin()
            if optimized_model is None:
                self._setup_model()
            else:
                self._model = optimized_model
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
        highest_sig = tsbb.detector_selection
        highest_sig_temp = [highest_sig[i : i + 2] for i in range(0, 5, 2)]
        highest_sig = highest_sig_temp
        side_0 = ["b0", "n0", "n1", "n2", "n3", "n4", "n5"]
        side_1 = ["b1", "n6", "n7", "n8", "n9", "na", "nb"]
        if highest_sig[0] in side_0:
            use_dets = side_0
        elif highest_sig[0] in side_1:
            use_dets = side_1
        else:
            print(highest_sig)
        return tsbb, use_dets, trigdat

    def get_swift(self):
        """ """
        print("Getting coinciding Swift GRB")
        swift_grb, swift_position = check_swift(self.grb, self.grb_time)
        try:
            assert swift_grb is not None, "No conciding Swift GRB found"
            assert swift_position is not None, "Only BAT localization available"
        except AssertionError:
            raise AlreadyRun(f"No swift position for {self.grb}")

        self._swift_grb_dict = swift_grb
        self.grb_position = swift_position

    def bkg_fitting(self):
        """
        Fitting the TTE Background and creating the Plugins
        """
        print("Fitting the Background for TTE")
        temp_timeseries = {}
        temp_responses = {}
        for d in self._use_dets:
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

            temp_responses[d] = response
            # skip_cache = True by default
        self._timeseries = temp_timeseries
        self._responses = temp_responses

    def _to_plugin(self, fix_correction=None):
        if self._fix_position:
            free_position = False
        else:
            free_position = True

        print("Creating BALROG like plugins")
        response_time = self.tsbb.stop_trigger - self.tsbb.start_trigger
        spectrum_likes = []
        for d in self._use_dets:
            if self._timeseries[d]._name not in ("b0", "b1"):
                spectrum_like = self._timeseries[d].to_spectrumlike()
                spectrum_like.set_active_measurements(self.energy_range)
                spectrum_likes.append(spectrum_like)
            else:
                spectrum_like = self._timeseries[d].to_spectrumlike()
                spectrum_like.set_active_measurements("300-30000")
                spectrum_likes.append(spectrum_like)
        balrog_likes = []
        print(f"We are going to use {self._use_dets}")
        for i, d in enumerate(self._use_dets):
            if free_position:
                bl = BALROGLikePositionPrior.from_spectrumlike(
                    spectrum_likes[i],
                    response_time,
                    self._responses[d],
                    free_position=free_position,
                    swift_position=self.grb_position,
                )
            else:
                bl = BALROGLike.from_spectrumlike(
                    spectrum_likes[i],
                    response_time,
                    self._responses[d],
                    free_position=free_position,
                )
            if fix_correction is None:
                if d not in ("b0", "b1", "n0", "n6"):
                    bl.use_effective_area_correction(0.5, 1.5)
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
        spectrum = Cutoff_powerlaw_Ep()
        spectrum.K.prior = Log_uniform_prior(lower_bound=1e-4, upper_bound=1000)
        spectrum.K.value = 1
        spectrum.index.value = -3
        spectrum.xp.value = 200
        spectrum.index.prior = Uniform_prior(lower_bound=-3, upper_bound=1)
        spectrum.xp.prior = Uniform_prior(lower_bound=10, upper_bound=10000)

        self._model = Model(
            PointSource(
                "GRB",
                self.grb_position.ra.deg,
                self.grb_position.dec.deg,
                spectral_shape=spectrum,
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
        if rank == 0:
            if not os.path.exists(self._temp_chains_dir):
                os.makedirs(os.path.join(self._temp_chains_dir))

        # use multinest to sample the posterior
        # set main_path+trigger to whatever you want to use

        self._bayes.set_sampler("multinest", share_spectrum=True)
        self._bayes.sampler.setup(
            n_live_points=800,
            chain_name=chain_path,
            wrapped_params=wrap,
            verbose=True,
        )
        self._bayes.sample()
        self.results = self._bayes.results
        self._results_loaded = True
        if rank == 0:
            fig = self.results.corner_plot()
            fig.savefig(os.path.join(self._temp_chains_dir, "cplot.pdf"))
            plt.close("all")
        color_dict = {
            "n0": "#FF9AA2",
            "n1": "#FFB7B2",
            "n2": "#FFDAC1",
            "n3": "#E2F0CB",
            "n4": "#B5EAD7",
            "n5": "#C7CEEA",
            "n6": "#DF9881",
            "n7": "#FCE2C2",
            "n8": "#B3C8C8",
            "n9": "#DFD8DC",
            "na": "#D2C1CE",
            "nb": "#6CB2D1",
            "b0": "#58949C",
            "b1": "#4F9EC4",
        }
        if rank == 0:
            try:
                spectrum_plot = display_spectrum_model_counts(self.results)
                ca = spectrum_plot.get_axes()[0]
                y_lims = ca.get_ylim()
                if y_lims[0] < 10e-6:
                    # y_lims_new = [10e-6, y_lims[1]]
                    ca.set_ylim(bottom=10e-6)
                spectrum_plot.tight_layout()
                spectrum_plot.savefig(
                    os.path.join(self._temp_chains_dir, "splot.pdf"),
                    bbox_inches="tight",
                )

            except:
                self.results.data_list = self._data_list
                spectrum_plot = display_spectrum_model_counts(self.results)
                ca = spectrum_plot.get_axes()[0]
                y_lims = ca.get_ylim()
                if y_lims[0] < 10e-6:
                    # y_lims_new = [10e-6, y_lims[1]]
                    ca.set_ylim(bottom=10e-6)

                spectrum_plot.tight_layout()
                spectrum_plot.savefig(
                    os.path.join(self._temp_chains_dir, "splot.pdf"),
                    bbox_inches="tight",
                )

                print("No spectral plot possible...")

            plt.close("all")

    def calc_separations(self):
        poshist = os.path.join(
            os.environ.get("GBMDATA"),
            "poshist",
            self.grb.strip("GRB")[:-3],
            f"glg_poshist_all_{self.grb.strip('GRB')[:-3]}_v00.fit",
        )
        trigdat = os.path.join(
            os.environ.get("GBMDATA"),
            "trigdat",
            f"20{self.grb.strip('GRB')[0:2]}",
            f"glg_trigdat_all_bn{self.grb.strip('GRB')}_v00.fit",
        )
        if not os.path.exists(poshist):
            download_gbm_file(date=self.grb.strip("GRB")[:-3], data_type="poshist")
            print("Done downloading poshist")
        if not os.path.exists(trigdat):
            download_trigdata_file(f"bn{self.grb.strip('GRB')}")
        # self.interpolator = PositionInterpolator.from_poshist(poshist)
        self.interpolator = PositionInterpolator.from_trigdat(trigdat)
        t0 = time.Time(self.grb_time, format="datetime", scale="utc")
        gbm_time = GBMTime(t0)
        self._gbm_time = gbm_time
        self.gbm = GBM(
            self.interpolator.quaternion(0),
            sc_pos=self.interpolator.sc_pos(0) * u.km,
        )
        sep = self.gbm.get_separation(self.grb_position)
        if "n0" in self._use_dets:
            normalizing_det = "n0"
        elif "n6" in self._use_dets:
            normalizing_det = "n6"
        normalizing_det_separation = sep[normalizing_det]
        if normalizing_det_separation > 30:
            normalizing_det = None

        self.separations = {}
        for d in lu:
            self.separations[d] = float(sep[d])
        self._angular_incident, self._use_dets = calc_angular_incident(
            self.grb_position,
            self.gbm,
            self._gbm_time,
            self.interpolator,
            self._use_dets,
        )
        print(self._use_dets)
        counter = 0
        for d in self._use_dets:
            if d not in ("b0", "b1"):
                counter += 1
        if counter < 3:
            raise RuntimeError("Too little number of dets seeing the burst")
        return normalizing_det

    def save_results(self):
        if rank == 0:
            df = self.results.get_data_frame("hpd")
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
                    results_yaml_dict[self.grb] = {"separations": self.separations}
                    temp = {}
            else:
                results_yaml_dict = {}
                results_yaml_dict[self.grb] = {"separations": self.separations}
                results_yaml_dict[self.grb][self.energy_range] = {}
                temp = {}
            for fp in self.results.optimized_model.free_parameters.keys():
                temp[fp] = float(self.results.optimized_model.free_parameters[fp].value)
            temp["confidence"] = {}

            print(df)
            for i in df.index:
                print(f"Index {i}")
                try:
                    temp["confidence"][df.loc[i].name] = {}
                    temp["confidence"][df.loc[i].name]["negative_error"] = float(
                        df.loc[i]["negative_error"]
                    )
                    temp["confidence"][df.loc[i].name]["positive_error"] = float(
                        df.loc[i]["positive_error"]
                    )
                except KeyError:
                    print(f"Did not find {i}")
            temp["angles"] = self._angular_incident
            results_yaml_dict[self.grb][self.energy_range] = temp
            with open(os.path.join(self._base_dir, "results.yml"), "w+") as f:
                yaml.dump(results_yaml_dict, f)

    def _already_run_check(self):
        try:
            with open(os.path.join(self._base_dir, "results.yml"), "r") as f:
                res_dict = yaml.safe_load(f)
        except FileNotFoundError:
            return False
        try:
            grb_dict = res_dict[self.grb]
            energy_dict = grb_dict[self.energy_range]
            for d in lu[:2]:
                test = energy_dict[d]
            return True
        except KeyError:
            return False

    def get_optimized_model(self):
        if self._results_loaded:
            return self.results.optimized_model
        else:
            raise ValueError("Model was not fitted! No optimized one available!")


def alread_run_externally(
    grb, result_yaml=os.path.join(os.environ.get("GBMDATA"), "localizing/results.yml")
):
    if rank == 0:
        if grb in ("GRB230818977", "GRB230812790"):
            return True
        if not os.path.exists(result_yaml):
            return False
        with open(result_yaml, "r") as f:
            res_dict = yaml.safe_load(f)
        if grb in res_dict.keys():
            return True
        else:
            return False


def get_grbs(csv=pkg_resources.resource_filename("effarea", "data/Fermi_Swift.lis")):
    """
    returns a list of GRBs with Swift localization
    """
    csv_content = pd.read_csv(csv, sep=" ", index_col=False, header=None)
    # grbs = csv_content["name"].loc[csv_content["swift_ra"] != None]
    grbs = csv_content.iloc[:, 0].copy()
    grbs = list(map(str, grbs.to_list()))
    return grbs


if __name__ == "__main__":
    # bin_start = 10
    # bin_stop = 700
    # num_selections = 10
    # tot_bins = np.geomspace(bin_start, bin_stop)
    # bins = np.array_split(tot_bins, num_selections)
    # energy_list = [f"{i[0]}-{i[-1]}" for i in bins]
    GRBS = None
    if rank == 0:
        GRBS = get_grbs()
    GRBS = comm.bcast(GRBS, root=0)

    for G in GRBS:
        if len(G) < 9:
            if rank == 0:
                G = "0" + G
            G = comm.bcast(G, root=0)
        if not alread_run_externally(f"GRB{G}"):
            G = f"GRB{G}"
            print(f"{G} on rank {rank}")
            try:
                GRB = FitTTE(G, fix_position=True)
                comm.Barrier()
                GRB.fit()
                comm.Barrier()
                GRB.save_results()
                comm.Barrier()
            except (FitFailed, AlreadyRun, IndexError, RuntimeError, TypeError) as e:
                print(e)
                comm.Call_errhandler(1)

            #    for energy in energy_list:
            #        GRB.set_energy_range(energy)
            # except (ZeroDivisionError, AlreadyRun) as e:
            #    print(f"passing  because {e}")
        # TODO fix MPI
        # TODO OUtput
        # TODO spectrum
        # TODO errors on paramaters
        #
