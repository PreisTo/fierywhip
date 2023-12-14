#!/usr/bin/env python3

from morgoth.auto_loc.time_selection import TimeSelection
import numpy as np
from astropy.stats import bayesian_blocks
from morgoth.utils.trig_reader import TrigReader
from threeML.utils.statistics.stats_tools import Significance
from fierywhip.utils.detector_utils import name_to_id

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


class TimeSelectionNew(TimeSelection):
    def __init__(self, **kwargs):
        self._trigdat_file = kwargs.get("trigdat_file", None)
        if self._trigdat_file is None:
            raise TypeError("trigdat_file must be supplied")
        self._fine = kwargs.get("fine", True)
        self._name = kwargs.get("name", "grb")
        self._p0 = kwargs.get("p0", 0.1)
        self._min_bkg_time = kwargs.get("min_bkg_time", 30)
        self._min_bb_block_bkg_duration = kwargs.get("min_bb_block_bkg_duration", 8)
        self._trigger_zone_background_start = kwargs.get(
            "trigger_zone_background_start", -5
        )
        self._trigger_zone_background_stop = kwargs.get(
            "trigger_zone_background_stop", 5
        )
        self._trigger_zone_active_start = kwargs.get("trigger_zone_active_start", -10)
        self._trigger_zone_active_stop = kwargs.get("trigger_zone_active_stop", 60)
        self._max_factor = kwargs.get("max_factor", 1.2)

        self._tr = TrigReader(self._trigdat_file, self._fine)

        obs, _ = self._tr.observed_and_background()
        self._tstart, self._tstop = self._tr.tstart_tstop()
        self._times = self._tstart + (self._tstop - self._tstart) / 2
        self._obs = np.sum(obs, axis=0).astype(int)
        self._bayesian_blocks = bayesian_blocks(self._times, self._obs, p0=self._p0)
        bb_indices = np.empty(len(self._bayesian_blocks), dtype=int)
        for i, r in enumerate(self._bayesian_blocks):
            if i != 0 and r > self._tstop[0]:
                index = np.argwhere(self._tstop <= r)[-1, -1]
                if self._tstop[index] < r:
                    if index + 1 < len(self._tstop):
                        if self._tstop[index + 1] > r:
                            bb_indices[i] = index + 1
                elif self._tstop[index] == r:
                    bb_indices[i] = index
            else:
                if i == 0:
                    bb_indices[i] = 0
                elif i == 1:
                    bb_indices[i] = 1
        m = np.zeros_like(bb_indices, dtype=bool)
        m[np.unique(bb_indices, return_index=True)[1]] = True
        bb_indices = bb_indices[m]
        self._bb_indices = bb_indices.copy()
        self._bb_times = self._tstart[self._bb_indices]
        self._bb_width = self._bb_times[1:] - self._bb_times[:-1]
        self._bb_cps = np.zeros_like(self._bb_width)
        for i in range(len(self._bb_indices) - 1):
            cps = np.average(
                self._obs[self._bb_indices[i] : self._bb_indices[i + 1]],
                weights=(self._tstop - self._tstart)[
                    self._bb_indices[i] : self._bb_indices[i + 1]
                ]
                / self._bb_width[i],
            )
            self._bb_cps[i] = cps

        self._bb_indices_self = np.arange(0, len(self._bb_width), 1)
        self._select_background()
        self._tr.set_background_selections(
            f"{self._bkg_neg_start}-{self._bkg_neg_stop}",
            f"{self._bkg_pos_start}-{self._bkg_pos_stop}",
        )
        self._select_active_time()

    def _select_background(self):
        mask = self._bb_width > self._min_bb_block_bkg_duration
        self._neg_bins = self._bb_times[:-1][mask] < self._trigger_zone_background_start
        # TODO make sure the neg background does not range into the trigger zone
        self._pos_bins = self._bb_times[:-1][mask] > self._trigger_zone_background_stop

        # neg_bkg
        bkg_neg = []
        start_flag = False
        for index in np.flip(self._bb_indices_self[mask][self._neg_bins]):
            if start_flag:
                if self._bb_cps[index] / bkg_neg[-1] > self._max_factor:
                    print("Too large of a jump")
                    if (
                        self._bb_times[np.max(bkg_neg)]
                        + self._bb_width[np.max(bkg_neg)]
                        - self._bb_times[np.min(bkg_neg)]
                        >= self._min_bkg_time
                    ):
                        print(
                            f"Breaking with a duration of {self._bb_times[np.max(bkg_neg)]+self._bb_width[np.max(bkg_neg)] - self._bb_times[np.min(bkg_neg)]}"
                        )
                        break
                    else:
                        print("not yet fulfilling min bkg time so adding anyways")
                        bkg_neg.append(index)
                else:
                    bkg_neg.append(index)
            else:
                bkg_neg.append(index)
                start_flag = True
        print("\n")
        # pos_bkg
        bkg_pos = []
        start_flag = False
        for index in self._bb_indices_self[mask][self._pos_bins]:
            print(self._bb_times[index])
            if start_flag:
                if self._bb_cps[index] / bkg_pos[-1] > self._max_factor:
                    print("Too large of a jump")
                    if (
                        self._bb_times[np.max(bkg_pos)]
                        + self._bb_width[np.max(bkg_pos)]
                        - self._bb_times[np.min(bkg_pos)]
                        >= self._min_bkg_time
                    ):
                        print(
                            f"Breaking with a duration of {self._bb_times[np.max(bkg_pos)]+self._bb_width[np.max(bkg_pos)] - self._bb_times[np.min(bkg_pos)]}"
                        )
                        break
                    else:
                        print("not yet fulfilling min bkg time so adding anyways")
                        bkg_pos.append(index)
                else:
                    bkg_pos.append(index)
            else:
                bkg_pos.append(index)
                start_flag = True
        self._bkg_neg_start = self._bb_times[np.min(bkg_neg)]
        self._bkg_neg_stop = self._bb_times[np.max(bkg_neg) + 1]
        self._bkg_pos_start = self._bb_times[np.min(bkg_pos)]
        self._bkg_pos_stop = self._bb_times[np.max(bkg_pos) + 1]
        self._background_time_neg = f"{self._bkg_neg_start}-{self._bkg_neg_stop}"
        self._background_time_pos = f"{self._bkg_pos_start}-{self._bkg_pos_stop}"
        self._max_time = self._bkg_pos_stop
        if self._bkg_pos_start < self._trigger_zone_active_stop:
            self._trigger_zone_active_stop = self._bkg_pos_start
        if self._bkg_neg_stop > self._trigger_zone_active_start:
            self._trigger_zone_active_start = self._bkg_neg_stop
        self._poly_order = -1

    def _select_active_time(self):
        avgs = {}

        for l in lu:
            temp = self._tr.time_series[l].significance_per_interval.copy()
            mask = np.ones_like(self._tstart).astype(int)
            mask[self._tstart < self._trigger_zone_background_start] = 0
            mask[self._tstart > self._trigger_zone_background_stop] = 0
            mask = mask.astype(bool)
            avgs[l] = np.average(
                temp[mask],
                weights=(self._tstop[mask] - self._tstart[mask])
                / (self._tstop[mask][-1] - self._tstart[mask][0]),
            )
        avgs_sorted = sorted(avgs.items(), key=lambda x: x[1])
        obs_significance = np.zeros_like(self._tstart)
        bkg_significance = np.zeros_like(self._tstart)
        obs_full, bkg_full = self._tr.observed_and_background()
        for k, v in avgs_sorted[-3:]:
            print(f"Using {k}")
            obs_significance += obs_full[name_to_id(k)]
            bkg_significance += bkg_full[name_to_id(k)]
        significance_object = Significance(obs_significance, bkg_significance)
        sig = significance_object.li_and_ma()
        sig[self._tstart < self._trigger_zone_active_start] = 0
        sig[self._tstart > self._trigger_zone_active_stop] = 0
        obs_significance[self._tstart < self._trigger_zone_active_start] = 0
        obs_significance[self._tstart > self._trigger_zone_active_stop] = 0
        at_start, at_stop, reason = self._select_active_time_algorithm(
            sig, obs_significance
        )
        print(f"Active Time eneded because of {reason}")
        self._active_time_start = at_start
        self._active_time_stop = at_stop
        self._active_time = f"{self._active_time_start}-{self._active_time_stop}"

    def _select_active_time_algorithm(
        self, sig, obs, max_trigger_duration=10.24, min_sig=None
    ):
        if min_sig is None:
            maxs = np.max(sig)
            means = np.mean(sig)
            min_sig = means + (maxs - means) * 0.2
        tstart = self._tstart
        tstop = self._tstop
        flag = True
        ts_start = np.argmax(obs)
        ts_stop = np.argmax(obs)
        print(f"Centering around {self._tstart[ts_start]}")
        duration = tstop[ts_stop] - tstart[ts_start]
        reasons = ["max_duration", "min_significance"]
        while duration <= max_trigger_duration:
            if sig[ts_start - 1] > sig[ts_stop + 1]:
                # 1 low
                if sig[ts_start - 2] > sig[ts_stop + 2]:
                    # 2 low
                    if sig[ts_start - 2] >= min_sig:
                        # 2 low min sig
                        if (
                            tstop[ts_stop] - tstart[ts_start - 2]
                            <= max_trigger_duration
                        ):
                            # 2 low max dur
                            ts_start -= 1
                            duration = tstop[ts_stop] - tstart[ts_start]
                        else:
                            reason = reasons[0]
                            duration = max_trigger_duration + 1
                    else:
                        duration = max_trigger_duration + 1
                        reason = reasons[1]
                else:
                    # 1 low 2 high
                    if sig[ts_stop + 2] >= min_sig:
                        # 1 low 2 high min sig
                        if (
                            tstop[ts_stop + 2] - tstart[ts_start]
                            <= max_trigger_duration
                        ):
                            # 1 low 2 high max dur
                            ts_stop += 1
                            duration = tstop[ts_stop] - tstart[ts_start]
                        else:
                            reason = reasons[0]
                            duration = max_trigger_duration + 1
                    else:
                        duration = max_trigger_duration + 1
                        reason = reasons[1]
            else:
                if sig[ts_start - 2] <= sig[ts_stop + 2]:
                    if sig[ts_stop + 2] >= min_sig:
                        if (
                            tstop[ts_stop + 2] - tstart[ts_start]
                            <= max_trigger_duration
                        ):
                            ts_stop += 1
                            duration = tstop[ts_stop] - tstart[ts_start]
                        else:
                            reason = reasons[0]
                            duration = max_trigger_duration + 1
                    else:
                        duration = max_trigger_duration + 1
                        reason = reasons[1]
                else:
                    if sig[ts_start - 2] >= min_sig:
                        if (
                            tstop[ts_stop] - tstart[ts_start - 2]
                            <= max_trigger_duration
                        ):
                            ts_start -= 1
                            duration = tstop[ts_stop] - tstart[ts_start]
                        else:
                            reason = reasons[0]
                            duration = max_trigger_duration + 1
                    else:
                        duration = max_trigger_duration + 1
                        reason = reasons[1]
        return tstart[ts_start], tstop[ts_stop], reason
