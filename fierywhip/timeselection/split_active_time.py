#!/usr/bin/python

from astropy.stats.bayesian_blocks import bayesian_blocks
from morgoth.utils.trig_reader import TrigReader
from fierywhip.utils.detector_utils import name2id
import logging
import numpy as np


def calculate_active_time_splits(
    trigdat_file: str,
    active_time: str,
    bkg_fit_files: list,
    use_dets: list,
    grb: str,
    max_drm_time=11,
    min_drm_time=1.024,
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
        except Exception as e:
            import time

            time.sleep(1)
            pass
        i += 1
        if i == 50:
            raise AssertionError(f"Can not restore background fit...\n{bkg_fit_files}")

    trig_reader.set_active_time_interval(active_time)
    cps, bkg = trig_reader.observed_and_background()
    start, stop = trig_reader.tstart_tstop()
    assert len(start) == len(cps[0]), "len of times and count rates notmatching"
    mask = np.zeros_like(cps[0], dtype=int)
    trigger_start = 0
    trigger_stop = 0
    at = active_time.split("-")
    if len(at) == 2:
        trigger_start = float(at[0])
        trigger_stop = float(at[-1])
    elif len(at) == 3:
        trigger_start = -float(at[1])
        trigger_stop = float(at[-1])
    elif len(at) == 4:
        trigger_start = -float(at[1])
        trigger_stop = -float(at[-1])
    else:
        raise ValueError
    mask[
        np.argwhere(start >= trigger_start)[0, 0] : np.argwhere(start > trigger_stop)[
            0, 0
        ]
    ] = 1
    mask = mask.astype(bool)
    cps_tmp = np.empty(len(cps[0]), dtype=float)
    for d in use_dets:
        i = name2id(d)
        cps_tmp += cps[i]
    cps = cps_tmp
    res = bayesian_blocks(start[mask], cps[mask].astype(int), fitness="events")
    bayesian_blocks_res = rebinning(start[mask], stop[mask], cps[mask], res)
    # calculate the jumps between the return
    jumps = {}
    for i in range(len(bayesian_blocks_res[0])):
        if i != 0:
            jumps[str(i)] = bayesian_blocks_res[1][i] / bayesian_blocks_res[1][i - 1]
        else:
            jumps[str(i)] = 1

    jumps_sorted = sorted(jumps.items(), key=lambda item: item[-1])

    splits = [trigger_start, trigger_stop]

    def get_corresponding_index(s, sl):
        for x, e in enumerate(sl):
            if e < s:
                pass
            else:
                return x
        return len(sl)

    jump_index = -1
    flag = True
    directions = 0
    while flag:
        add = False
        if np.absolute(jump_index) < len(jumps_sorted):
            ji = jumps_sorted[jump_index][0]
        else:
            raise IndexError
        st = bayesian_blocks_res[0][int(ji)]
        si = get_corresponding_index(st, splits)
        if directions == 0:
            add = True
        elif directions == 1:
            if st > buffer_st:
                if (
                    st - splits[si - 1] >= min_drm_time
                    and splits[si] - st >= min_drm_time
                ):
                    add = True
        elif directions == -1:
            if st < buffer_st:
                if (
                    st - splits[si - 1] >= min_drm_time
                    and splits[si] - st >= min_drm_time
                ):
                    add = True

        if add:
            if st < splits[-1] and st > splits[0]:
                splits.insert(si, st)
                if (
                    st - splits[si - 1] > max_drm_time
                    and splits[si + 1] - st <= max_drm_time
                ):
                    directions = -1
                    buffer_st = st
                elif (
                    splits[si + 1] - st > max_drm_time
                    and st - splits[si - 1] <= max_drm_time
                ):
                    directions = 1
                    buffer_st = st
                elif (
                    st - splits[si - 1] > max_drm_time
                    and splits[si - 1] - st > max_drm_time
                ):
                    directions = 0
                else:
                    flag = False
        jump_index -= 1
    save_lightcurves(trig_reader, splits, grb)
    return splits


def save_lightcurves(trigreader, splits, grb, path=None):
    if path is None:
        path = os.path.join(os.environ.get("GBM_TRIGGER_DATA_DIR"), "tridgat/v00/lc")
    figs = trigreader.view_lightcurve(return_plots=True)
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
    obs_binned = []
    for i in range(len(indices) - 1):
        obs_binned.append(np.average(obs[i : i + 1], weights=weights[i : i + 1]))
    width_binned = time_bounds[1:] - time_bounds[:-1]
    times_binned.append(stop[-1])
    return times_binned, obs_binned, width_binned
