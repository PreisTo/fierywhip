#!/usr/bin/env python3

from gbmbkgpy.io.downloading import download_gbm_file, download_trigdata_file
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
from gbm_drm_gen.io.balrog_like import BALROGLike
import os
from mpi4py import MPI
import numpy as np
import yaml
import matplotlib.pyplot as plt


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


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

GRB = "GRB230826814"
day_seconds = 24 * 60 * 60
grb_date = datetime(
    int(f"20{GRB.strip('GRB')[:2]}"),
    int(GRB.strip("GRB")[2:4]),
    int(GRB.strip("GRB")[4:6]),
)
grb_time = grb_date + timedelta(seconds=int(GRB[-3:]) * day_seconds / 1000)

# get swift grb table and look for coinciding
swift_table = pd.read_csv(
    "swift_grbs.txt", sep="\t", decimal=".", encoding="latin-1", index_col=False
)
swift_table.insert(1, "Date", [i[0:-1] for i in swift_table["GRB"]], True)
coinc = swift_table.loc[swift_table["Date"] == GRB.strip("GRB")[:-3]]

print(f"Total number of {len(coinc['Date'])} Swift trigger(s) found")

swift_grb = None
for c in coinc["Time [UT]"]:
    cd = datetime.strptime(c, "%H:%M:%S")
    cd = cd.replace(year=grb_date.year, month=grb_date.month, day=grb_date.day)
    if grb_time >= cd - timedelta(minutes=2) and grb_time <= cd + timedelta(minutes=2):
        swift_grb = coinc.loc[coinc["Time [UT]"] == c]
    else:
        print(cd)
        print(grb_time)
        print((grb_time - cd).total_seconds())

if swift_grb is not None:
    swift_grb = swift_grb.to_dict()

    poshist = os.path.join(
        os.environ.get("GBMDATA"),
        "poshist",
        GRB.strip("GRB")[:-3],
        f"glg_poshist_all_{GRB.strip('GRB')[:-3]}_v00.fit",
    )
    trigdat = os.path.join(
        os.environ.get("GBMDATA"),
        "trigdat",
        str(grb_time.year),
        f"glg_trigdat_all_bn{GRB.strip('GRB')}_v00.fit",
    )
    if not os.path.exists(poshist):
        download_gbm_file(date=GRB.strip("GRB")[:-3], data_type="poshist")
        print("Done downloading poshist")
    if not os.path.exists(trigdat):
        download_trigdata_file(f"bn{GRB.strip('GRB')}")
        print("Done downloading trigdat")
    interpolator = PositionInterpolator.from_poshist(poshist)

    if swift_grb["XRT RA (J2000)"] != "n/a":
        sgd = list(swift_grb["Date"].keys())
        swift_position = SkyCoord(
            ra=swift_grb["XRT RA (J2000)"][sgd[0]],
            dec=swift_grb["XRT Dec (J2000)"][sgd[0]],
            unit=(u.hourangle, u.deg),
        )
        print(swift_position)
    else:
        print("Only BAT localization available")
    interp_trigdat = PositionInterpolator.from_trigdat(trigdat)

t0 = time.Time(grb_time, format="datetime", scale="utc")
gbm_time = GBMTime(t0)
gbm = GBM(
    interpolator.quaternion(gbm_time.met),
    sc_pos=interpolator.sc_pos(gbm_time.met) * u.km,
)
sep = gbm.get_separation(swift_position)
for d in lu:
    print(d, sep[d])

tsbb = TimeSelectionBB(GRB, trigdat, fine=True)


obs_array, _ = tsbb.trigreader_object.observed_and_background()
start_times, end_times = tsbb.trigreader_object.tstart_tstop()
start_times = np.array(start_times)
start_times, nduplicates = np.unique(start_times, return_index=True)
dup_counter = 0
duplicates = []
for d in range(len(lu)):
    obs_array[d] = list(obs_array[d])
for i in range(len(obs_array[0])):
    if i in nduplicates:
        pass
    else:
        duplicates.append(i)
end_times = list(end_times)
for duplicate in duplicates:
    print(duplicate)
    for d in range(len(lu)):
        obs_array[d].pop(duplicate - dup_counter)
    end_times.pop(duplicate - dup_counter)
    dup_counter += 1
for d in range(len(lu)):
    obs_array[d] = np.array(obs_array[d])
obs_array = np.array(obs_array)
end_times = np.array(end_times)

mask = np.zeros_like(start_times)
start_id, stop_id = (
    np.argwhere(start_times < tsbb.start_trigger)[-1, 0],
    np.argwhere(start_times > tsbb.stop_trigger)[0, 0],
)
mask[start_id:stop_id] = 1
mask = mask.astype(bool)

counts_sum = np.zeros_like(start_times)
detsel = [
    tsbb.detector_selection[i : i + 2]
    for i in range(0, len(tsbb.detector_selection), 2)
]
for i, d in enumerate(lu):
    if d in detsel:
        counts_sum += obs_array[i]
def get_bb_edges(start_times,mask,count_sum):
    bb_edges = bayesian_blocks(
        start_times[mask], counts_sum[mask], fitness="events", p0=0.025
    )
    bb_edges_start = []
    bb_edges_stop = []
    for e, edge in enumerate(bb_edges):
        if e < len(bb_edges) - 1:
            start_temp = start_times[np.argwhere(start_times <= edge)[-1, 0]]
            stop_temp =  start_times[np.argwhere(start_times <= bb_edges[e+1])[-1,0]]
            if start_temp == stop_temp:
                stop_temp = start_times[np.argwhere(start_times == start_temp)[-1,0]+1]
            bb_edges_start.append(start_temp)
            bb_edges_stop.append(stop_temp)
    return bb_edges_start,bb_edges_stop

len_bins_trigger = len(start_times[mask]>0)
block_len = int(len_bins_trigger/5)
mod_len = len_bins_trigger%5

bb_edges_start = []
bb_edges_stop = []

bb_edges_start.append(tsbb.start_trigger)
bb_edges_stop.append(tsbb.stop_trigger)
for i in range(5):
    if i<4:
        bb_edges_start.append(start_times[start_id +i*block_len])
        bb_edges_stop.append(start_times[start_id+(i+1)*block_len])
    else:
        bb_edges_start.append(start_times[start_id + i*block_len])
        bb_edges_stop.append(start_times[start_id+(i+1)*block_len + mod_len])

lc_path = os.path.join(os.environ.get("GBMDATA"), f"localizing/{GRB}/lightcurves")
try:
    os.makedirs(lc_path)
except FileExistsError:
    pass
if rank == 0:
    figs = tsbb.trigreader_object.view_lightcurve(-10, 30, return_plots=True)
    for fig in figs:
        ax = fig[1].axes[0]
        yl = ax.get_ylim()
        ax.vlines(bb_edges_start, 0, 10e4)
        ax.set_ylim(yl)
        fig[1].savefig(os.path.join(lc_path, f"{fig[0]}.pdf"))
# bb_edges_t = #start_times[bb_edges]
bkg = (tsbb.background_time_neg, tsbb.background_time_pos)

# result file
results_yml = os.path.join(os.environ.get("GBMDATA"), f"localizing/trigdat_norms.yml")
final_dict = {}
dir_list = os.listdir(os.path.join(os.environ.get("GBMDATA"), f"localizing/{GRB}"))
dir_list = [str(i).strip("/") for i in dir_list]
print(dir_list)
comm.Barrier()
for s in range(len(bb_edges_start)):
    selection = f"{bb_edges_start[s]}-{bb_edges_stop[s]}"
    print(f"Running selection {selection}")
    if selection in dir_list:
        pass
    else:
        trigreader = TrigReader(trigdat, fine=True)
        trigreader.set_active_time_interval(selection)
        trigreader.set_background_selections(
            tsbb.background_time_neg, tsbb.background_time_pos
        )

        # balrog_plugin = trigreader.to_plugin(*lu)
        # for i,d in enumerate(balrog_plugin):
        #    balrog_plugin[i] = d.use_effective_area_correction(0.7,1.3)
        # datalist = DataList(*balrog_plugin)
        datalist = {}
        for d in lu:
            print(d)
            speclike = trigreader._time_series[d].to_spectrumlike()
            # speclike.set_active_measurements("c1-c6")
            time = 0.5 * (
                trigreader._time_series[d].tstart + trigreader._time_series[d].tstop
            )

            balrog_like = BALROGLike.from_spectrumlike(
                speclike, time=time, free_position=False
            )

            balrog_like.set_active_measurements("c1-c6")
            datalist[d] = balrog_like
            if d not in ("b0", "b1"):
                datalist[d].use_effective_area_correction(0.7, 1.3)
            else:
                datalist[d].fix_effective_area_correction(1)

        cpl = Cutoff_powerlaw()
        cpl.K.prior = Log_uniform_prior(lower_bound=0.0001, upper_bound=500)
        cpl.xc.prior = Log_uniform_prior(lower_bound=10, upper_bound=1000)
        cpl.index.set_uninformative_prior(Uniform_prior)
        ps = PointSource(
            "grb",
            ra=float(swift_position.ra.deg),
            dec=float(swift_position.dec.deg),
            spectral_shape=cpl,
        )
        ps.position.fix = True
        ps.position.free = False
        model = Model(ps)
        bayes = BayesianAnalysis(model, datalist)
        bayes.set_sampler("multinest", share_spectrum=True)
        wrap = [0] * len(
            model.free_parameters
        )  # not working properlyViel Erfolg und Spaß in den ersten Wochen!
        wrap[0] = 1

        fit_path = os.path.join(
            os.environ.get("GBMDATA"), f"localizing/{GRB}/{selection}"
        )
        try:
            os.makedirs(fit_path)
        except FileExistsError:
            pass

        bayes.sampler.setup(
            n_live_points=800,
            wrapped_params=wrap,
            chain_name=os.path.join(fit_path, "fit_"),
            verbose=True,
        )
        bayes.sample()

        result_path = fit_path
        trigger = GRB
        spectrum = "cpl"
        bayes.results.write_to(
            result_path + "/grb_" + trigger + "_" + spectrum + ".fits", overwrite=True
        )

        results = bayes.results

        # spectrum and residuals
        try:
            spectrum_plot = display_spectrum_model_counts(bayes)
            spectrum_plot.savefig(
                result_path + "/grb_" + trigger + "_spectrum_" + spectrum + ".pdf"
            )
        except Exception as e:
            print(f"Spectrum plot not possible due to {e}")
        # corner plot

        cc_plot = results.corner_plot()
        cc_plot.savefig(
            result_path + "/grb_" + trigger + "_cornerplot_" + spectrum + ".pdf"
        )
        comm.Barrier()
        if rank == 0:
            opt_model = results.optimized_model
            result_dict = {}
            result_dict["K"] = float(opt_model["grb.spectrum.main.Cutoff_powerlaw.K"].value)
            result_dict["index"] = float(opt_model["grb.spectrum.main.Cutoff_powerlaw.index"].value)
            result_dict["xc"] = float(opt_model["grb.spectrum.main.Cutoff_powerlaw.xc"].value)
            for d in lu:
                result_dict[d] = float(opt_model[f"cons_{d}"].value)
            final_dict[selection] = result_dict
            with open(os.path.join(result_path,"res.yml"),"w+") as f:
                yaml.dump(final_dict,f)
        plt.close("all")
        bayes = None
        results = None
        opt_model = None

final_dict["separation"] = {}
for d in lu:
    final_dict["separation"][d] = float(sep[d])

if os.path.isfile(results_yml) and os.path.exists(result_yml):
    temp = {}
    temp[GRB] = final_dict
    with open(results_yml, "a") as f:
        yaml.dump(temp, f)
else:
    temp = {}
    temp[GRB] = final_dict
    with open(results_yml, "w+") as f:
        yaml.dump(temp, f)
