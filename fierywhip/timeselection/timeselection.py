#!/usr/bin/env python3

from morgoth.auto_loc.time_selection import TimeSelection
import numpy as np
from astropy.stats import bayesian_blocks
from morgoth.utils.trig_reader import TrigReader
from threeML.utils.statistics.stats_tools import Significance
from fierywhip.utils.detector_utils import name2id
from fierywhip.config.configuration import fierywhip_config
from gbm_drm_gen import DRMGenTrig
from gbm_drm_gen.io.balrog_drm import BALROG_DRM
from gbm_drm_gen.io.balrog_like import BALROGLike
from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike
from threeML.utils.spectrum.binned_spectrum import BinnedSpectrumWithDispersion
from threeML.utils.spectrum.binned_spectrum_set import BinnedSpectrumSet
from threeML.utils.time_series.binned_spectrum_series import BinnedSpectrumSeries
from threeML.utils.time_interval import TimeIntervalSet
from threeML.utils.data_builders.time_series_builder import TimeSeriesBuilder
import matplotlib.pyplot as plt
import logging

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
        # setting up the relevant parameters from kwargs
        self._trigdat_file = kwargs.get("trigdat_file", None)
        if self._trigdat_file is None:
            raise TypeError("trigdat_file must be supplied")
        self._fine = kwargs.get("fine", True)
        self._name = kwargs.get("name", "grb")
        self._p0 = kwargs.get("p0", 0.1)
        self._min_bkg_time = kwargs.get(
            "min_bkg_time", fierywhip_config.config.timeselection.min_bkg_time
        )
        self._min_bb_block_bkg_duration = kwargs.get(
            "min_bb_block_bkg_duration",
            fierywhip_config.config.timeselection.min_bb_block_bkg_duration,
        )
        self._trigger_zone_background_start = kwargs.get(
            "trigger_zone_background_start",
            fierywhip_config.config.timeselection.trigger_zone_bkg_start,
        )
        self._trigger_zone_background_stop = kwargs.get(
            "trigger_zone_background_stop",
            fierywhip_config.config.timeselection.trigger_zone_bkg_stop,
        )
        self._trigger_zone_active_start = kwargs.get(
            "trigger_zone_active_start",
            fierywhip_config.config.timeselection.trigger_zone_active_start,
        )
        self._trigger_zone_active_stop = kwargs.get(
            "trigger_zone_active_stop",
            fierywhip_config.config.timeselection.trigger_zone_active_stop,
        )
        self._max_factor = kwargs.get(
            "max_factor", fierywhip_config.config.timeselection.max_factor
        )
        self._sig_reduce_factor = kwargs.get(
            "sig_reduce_factor", fierywhip_config.config.timeselection.sig_reduce_factor
        )
        self._min_trigger_duration = kwargs.get(
            "min_trigger_duartion",
            fierywhip_config.config.timeselection.min_trigger_duration,
        )
        self._max_trigger_duration = kwargs.get(
            "max_trigger_duration",
            fierywhip_config.config.timeselection.max_trigger_duration,
        )

        logging.info(f"Starting the TimeSelectionNew with following kwargs: {kwargs}")

        # creating trigreader object
        self._tr = TrigReader(self._trigdat_file, self._fine)

        # get the observed rates
        obs, _ = self._tr.observed_and_background()
        self._tstart, self._tstop = self._tr.tstart_tstop()
        self._time_intervals = TimeIntervalSet.from_starts_and_stops(
            self._tstart, self._tstop
        )
        self._times = self._tstart + (self._tstop - self._tstart) / 2
        self._obs = np.sum(obs, axis=0).astype(int)

        self._create_bayesian_blocks()

        self._select_background()
        self._select_active_time()

        while self._bkg_pos_start - self._active_time_stop < 20:
            logging.info(f"Background pos start too close to trigger time, rerunning")
            self._trigger_zone_background_stop = self._active_time_stop + 20
            self._select_background()
            self._select_active_time()

    def _create_bayesian_blocks(self):
        # create the bayesian blocks
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

    def _select_background(self):
        flag = True
        while flag:
            mask = self._bb_width > self._min_bb_block_bkg_duration
            self._neg_bins = (
                self._bb_times[:-1][mask] < self._trigger_zone_background_start
            )
            # TODO make sure the neg background does not range into the trigger zone
            self._pos_bins = (
                self._bb_times[:-1][mask] > self._trigger_zone_background_stop
            )
            if (
                len(self._bb_indices_self[mask][self._pos_bins]) > 0
                and len(self._bb_indices_self[mask][self._neg_bins]) > 0
            ):
                flag = False
            else:
                if self._min_bb_block_bkg_duration >1.024:
                    self._min_bb_block_bkg_duration -= 0.5
                else:
                    raise ValueError("TimeSelection failed because min bkg bin duration is too small")
        # neg_bkg
        bkg_neg = []
        start_flag = False
        for index in np.flip(self._bb_indices_self[mask][self._neg_bins]):
            if start_flag:
                if self._bb_cps[index] / bkg_neg[-1] > self._max_factor:
                    if (
                        self._bb_times[np.max(bkg_neg)]
                        + self._bb_width[np.max(bkg_neg)]
                        - self._bb_times[np.min(bkg_neg)]
                        >= self._min_bkg_time
                    ):
                        logging.debug(
                            f"Breaking with a duration of {self._bb_times[np.max(bkg_neg)]+self._bb_width[np.max(bkg_neg)] - self._bb_times[np.min(bkg_neg)]}"
                        )
                        break
                    else:
                        bkg_neg.append(index)
                else:
                    bkg_neg.append(index)
            else:
                bkg_neg.append(index)
                start_flag = True
        # pos_bkg
        bkg_pos = []
        start_flag = False
        assert (
            len(self._bb_indices_self[mask][self._pos_bins]) > 0
        ), "yeah well that sucks"
        for index in self._bb_indices_self[mask][self._pos_bins]:
            if start_flag:
                if self._bb_cps[index] / bkg_pos[-1] > self._max_factor:
                    if (
                        self._bb_times[np.max(bkg_pos)]
                        + self._bb_width[np.max(bkg_pos)]
                        - self._bb_times[np.min(bkg_pos)]
                        >= self._min_bkg_time
                    ) and len(bkg_pos) > 0:
                        logging.debug(
                            f"Breaking with a duration of {self._bb_times[np.max(bkg_pos)]+self._bb_width[np.max(bkg_pos)] - self._bb_times[np.min(bkg_pos)]}"
                        )
                        break
                    else:
                        logging.debug(
                            "not yet fulfilling min bkg time so adding anyways"
                        )
                        bkg_pos.append(index)
                else:
                    bkg_pos.append(index)
            else:
                bkg_pos.append(index)
                start_flag = True
        assert len(bkg_neg) > 0, "no neg background selected"
        assert len(bkg_pos) > 0, "no pos background selected"
        self._bkg_neg_start = float(self._bb_times[np.min(bkg_neg)])
        self._bkg_neg_stop = float(self._bb_times[np.max(bkg_neg) + 1])
        self._bkg_pos_start = float(self._bb_times[np.min(bkg_pos)])
        self._bkg_pos_stop = float(self._bb_times[np.max(bkg_pos) + 1])
        self._background_time_neg = f"{self._bkg_neg_start}-{self._bkg_neg_stop}"
        self._background_time_pos = f"{self._bkg_pos_start}-{self._bkg_pos_stop}"
        self._max_time = float(self._bkg_pos_stop)
        if self._bkg_pos_start < self._trigger_zone_active_stop:
            self._trigger_zone_active_stop = self._bkg_pos_start
        if self._bkg_neg_stop > self._trigger_zone_active_start:
            self._trigger_zone_active_start = self._bkg_neg_stop
        self._poly_order = int(-1)
        self._tr.set_background_selections(
            f"{self._bkg_neg_start}-{self._bkg_neg_stop}",
            f"{self._bkg_pos_start}-{self._bkg_pos_stop}",
        )

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
        obs_significance = np.zeros((len(self._tstart), 8))
        obs_full, bkg_full = self._tr.observed_and_background()
        n5_skipped = False
        if fierywhip_config.config.timeselection.replace_n5:
            logging.info("Checking if n5 is used for timeselection, if so replacing it")
            for k, v in avgs_sorted[-3:]:
                if k != "n5":
                    logging.debug(f"Using {k}")
                    obs_significance += self._tr._rates[:, name2id(k), :].reshape(
                        len(self._tstart), 8
                    )
                else:
                    logging.debug("skipping n5")
                    n5_skipped = True
            if n5_skipped:
                k, v = avgs_sorted[-4]
                logging.info(f"Using {k} instead of n5")
                obs_significance += self._tr._rates[:, name2id(k), :].reshape(
                    len(self._tstart), 8
                )
        else:
            for k, v in avgs_sorted[-3:]:
                logging.debug(f"Using {k}")
                obs_significance += self._tr._rates[:, name2id(k), :].reshape(
                    len(self._tstart), 8
                )

        rates, bkg, sig = self._create_additional_timeseries(obs_significance)

        obs_significance = rates
        bkg_significance = bkg

        sig[self._tstart < self._trigger_zone_active_start] = 0
        sig[self._tstart > self._trigger_zone_active_stop] = 0
        obs_significance[self._tstart < self._trigger_zone_active_start] = 0
        obs_significance[self._tstart > self._trigger_zone_active_stop] = 0
        at_start, at_stop, reason, min_sig = self._select_active_time_algorithm(
            sig, obs_significance
        )
        while reason == "min_significance" and min_sig * self._sig_reduce_factor > 2:
            logging.debug(
                f"Trying if reducing the required significance by {round(1-self._sig_reduce_factor,2)} changes the selection"
            )
            (
                at_start_new,
                at_stop_new,
                reason_new,
                min_sig_new,
            ) = self._select_active_time_algorithm(
                sig,
                obs_significance,
                min_sig=min_sig * self._sig_reduce_factor,
                max_trigger_duration=self._max_trigger_duration,
            )
            if at_stop_new - at_start_new > at_stop - at_start:
                at_start = at_start_new
                at_stop = at_stop_new
                min_sig = min_sig_new
                reason = reason_new
            else:
                reason = "no_improvement"

        logging.debug(f"Active Time eneded because of {reason}")
        while at_stop - at_start < self._min_trigger_duration:
            logging.debug(
                f"Min trigger duration is not met - we will add another bin to the trigger end to fulfill it"
            )
            stopping_index = np.argwhere(self._tstop == at_stop)[0, 0]
            at_stop = self._tstop[stopping_index + 1]
        self._active_time_start = float(at_start)
        self._active_time_stop = float(at_stop)
        self._active_time = f"{self._active_time_start}-{self._active_time_stop}"
        self._tr.set_active_time_interval(self._active_time)

    def _select_active_time_algorithm(
        self, sig, obs, max_trigger_duration=11, min_sig=None
    ):
        if min_sig is None:
            maxs = np.max(sig)
            means = np.mean(sig)
            min_sig = means + (maxs - means) * 0.2
        tstart = self._tstart
        tstop = self._tstop
        ts_start = np.argmax(obs)
        ts_stop = np.argmax(obs)
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
        return tstart[ts_start], tstop[ts_stop], reason, min_sig

    def _create_additional_timeseries(self, cps):
        # just a dummy response gen used for getting the energy bins
        drm_gen = DRMGenTrig(
            trigdat_file=self._trigdat_file,
            det=0,  # det number
            tstart=self._tstart,
            tstop=self._tstop,
            mat_type=2,
            time=0,
            occult=True,
        )

        # we will use a single response for each detector

        tmp_drm = BALROG_DRM(drm_gen, 0, 0)

        # extract the counts

        counts = cps * (self._tstop - self._tstart).reshape(len(self._tstart), 1)
        # now create a binned spectrum for each interval

        binned_spectrum_list = []
        for c, start, stop in zip(counts, self._tstart, self._tstop):
            binned_spectrum_list.append(
                BinnedSpectrumWithDispersion(
                    counts=c,
                    exposure=stop - start,
                    response=tmp_drm,
                    tstart=start,
                    tstop=stop,
                )
            )

        # make a binned spectrum set

        bss = BinnedSpectrumSet(
            binned_spectrum_list,
            reference_time=0.0,
            time_intervals=self._time_intervals,
        )

        # convert that set to a series

        bss2 = BinnedSpectrumSeries(bss, first_channel=0)

        # create a time series builder which can produce plugins

        tsb = TimeSeriesBuilder(
            name="max_sig_combined",
            time_series=bss2,
            response=tmp_drm,
            verbose=False,
            poly_order=-1,
        )

        # attach that to the full list

        self._max_sig_tsb = tsb
        self._max_sig_tsb.set_background_interval(
            self._background_time_neg, self._background_time_pos
        )
        # dummy active time interval - needed for correct significance
        start = -1000
        stop = 1000
        counts = []
        width = []
        bins = bss.time_intervals.containing_interval(start, stop)
        for bin in bins:
            counts.append(
                self._max_sig_tsb.time_series.counts_over_interval(
                    bin.start_time, bin.stop_time
                )
            )
            width.append(bin.duration)
        counts = np.array(counts)
        width = np.array(width)
        rates_observed = counts / width

        polynomials = self._max_sig_tsb.time_series.polynomials

        bkg = []
        for j, tb in enumerate(bins):
            tmpbkg = 0.0
            for poly in polynomials:
                tmpbkg += poly.integral(tb.start_time, tb.stop_time)

            bkg.append(tmpbkg / width[j])

        rates_observed = np.array(rates_observed)
        bkg = np.array(bkg)

        sig = self._max_sig_tsb.significance_per_interval.copy()
        logging.debug(f"This is the significance {sig}")
        return rates_observed, bkg, sig

    @property
    def trigreader(self):
        return self._tr
