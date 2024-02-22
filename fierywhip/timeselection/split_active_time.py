#!/usr/bin/python

# from astropy.stats.bayesian_blocks import bayesian_blocks
from morgoth.utils.trig_reader import TrigReader
from fierywhip.utils.detector_utils import name2id

# import logging
import numpy as np
import os
import ruptures as rpt


def calculate_active_time_splits(
    trigdat_file: str,
    active_time: str,
    bkg_fit_files: list,
    use_dets: list,
    grb: str,
    max_drm_time=11,
    min_drm_time=1.024,
    max_nr_responses=3,
    min_bin_width=0.064,
):
    split = []
    success_restore = False
    i = 0
    while not success_restore:
        try:
            trig_reader = TrigReader(
                trigdat_file,
                fine=True,
                verbose=False,
                restore_poly_fit=bkg_fit_files,
            )
            success_restore = True
            i = 0
        except Exception:
            import time

            time.sleep(1)
            pass
        i += 1
        if i == 50:
            raise AssertionError(f"Can not restore background fit...\n{bkg_fit_files}")

    trig_reader.set_active_time_interval(active_time)
    cps, bkg = trig_reader.observed_and_background()
    cps_new = np.zeros_like(cps[0])
    # adding up all the dets we will use
    for d in use_dets:
        i = name2id(d)
        cps_new += cps[i]
    start, stop = trig_reader.tstart_tstop()

    def rebinning_changepoints(start, stop, vals, bin_width=min_bin_width):
        """
        Rebinns CPS into smaller bins with the min time resolution for the
        changepoint selection to use
        """
        new_times = np.zeros(np.sum((stop - start) // bin_width).astype(int) + 1)
        new_vals = np.zeros(np.sum((stop - start) // bin_width).astype(int) + 1)
        new_times_counter = 0
        for x, y, z in zip(start, stop, vals):
            if y - x > bin_width:
                fine_bin_number = int((y - x) // bin_width)
                for i in range(fine_bin_number):
                    nt = x + i * bin_width
                    new_times[new_times_counter] = nt
                    new_vals[new_times_counter] = z
                    new_times_counter += 1

            else:
                new_times[new_times_counter] = x
                new_vals[new_times_counter] = z
                new_times_counter += 1
        return new_times, new_vals

    at_start, at_stop = time_splitter(active_time)
    mask = np.zeros_like(start).astype(bool)
    mask[np.argwhere(start >= at_start)[0, 0] : np.argwhere(start > at_stop)[0, 0]] = (
        True
    )
    t, v = rebinning_changepoints(start[mask], stop[mask], cps_new[mask])
    res = rpt.Dynp(model="l2", min_size=min_drm_time // min_bin_width).fit(v)
    flag_rpt = True
    failed = 0
    while flag_rpt:
        try:
            r = res.predict(max_nr_responses)
            flag_rpt = False
        except rpt.exceptions.BadSegmentationParameters:
            failed += 1
    faulty = []
    for i in range(t.shape[0]):
        if t[-i] == 0:
            faulty.append(t.shape[0] - 1 - i)

    if len(v) in r:
        r = r[:-1]

    split.append(at_start)
    split.extend(r)
    split.append(at_stop)
    durations = split[1:] - split[:-1]
    for start_split, stop_split, d in (split[:-1], split[1:], durations):
        if d > max_drm_time and len(split) - 1 <= max_nr_responses:
            split.append(start_split + d / 2)
    split = sorted(split)
    return split


def save_lightcurves(trigreader, splits, grb, path=None):
    # TODO set x_lim
    if path is None:
        path = os.path.join(
            os.environ.get("GBM_TRIGGER_DATA_DIR"), grb, "trigdat/v00/lc"
        )
    if not os.path.exists(path):
        os.path.makedirs(path)

    bkg_intervals = trigreader.time_series["n0"].time_series.bkg_intervals
    prev = bkg_intervals[0].start_time, bkg_intervals[0].stop_time
    after = bkg_intervals[1].start_time, bkg_intervals[1].stop_time

    figs = trigreader.view_lightcurve(
        start=prev[-1] - 20, stop=after[0] + 20, return_plots=True
    )
    for f in figs:
        fig = f[1]
        axes = fig.get_axes()
        ylim = axes[0].get_ylim()
        for x in splits:
            axes[0].vlines(x, 0, 10e5, color="magenta")
        axes[0].set_ylim(ylim)
        fig.savefig(
            os.path.join(path, f"{grb}_lightcurve_trigdat_detector_{f[0]}_plot_v00.png")
        )


def rebinning(start, stop, obs, time_bounds):
    times_binned = list(time_bounds)
    # find the correspondingin indices
    indices = [0]
    for t in time_bounds:
        indices.append(np.argwhere(stop > t)[0, 0])
    indices.append(len(start) - 1)
    weights = stop - start / (stop[-1] - start[0])
    if np.sum(weights) == 0:
        weights = np.ones_like(len(indices) - 1)
    obs_binned = []
    for i in range(len(indices) - 1):
        obs_binned.append(np.average(obs[i : i + 1], weights=weights[i : i + 1]))
    width_binned = time_bounds[1:] - time_bounds[:-1]
    times_binned.append(stop[-1])
    return times_binned, obs_binned, width_binned


def time_splitter(time: str):
    splitted_time = time.split("-")
    if len(splitted_time) == 2:
        return float(splitted_time[0]), float(splitted_time[1])
    elif len(splitted_time) == 3:
        return -float(splitted_time[1]), float(splitted_time[-1])
    elif len(splitted_time) == 4:
        return -float(splitted_time[1]), -float(splitted_time[-1])
