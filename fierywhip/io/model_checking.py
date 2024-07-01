#!/usr/bin/python
import numpy as np
from threeML.analysis_results import load_analysis_results
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
import os
import logging


def create_model_checking_plots(
    grb_list: list,
    save_path=None,
    sample_size=10000,
    base_dir=os.environ.get("GBM_TRIGGER_DATA_DIR"),
):
    """
    :param grb_list: list with GRB objects
    :param save_path: path to save the fig, defaults to None = not saving
    :param sample_size: how many samples will be drawn from the posterior
    :param base_dir: path of fit results, defaults to $GBM_TRIGGER_DATA_DIR
    """

    log_prob_samples = np.zeros(sample_size)
    log_prob_truth = 0.0
    for g in grb_list:
        try:
            ar = load_analysis_results(
                os.path.join(
                    base_dir, g.name, "trigdat/v00/trigdat_v00_loc_results.fits"
                )
            )
            s = ar.log_probability[
                np.random.choice(np.arange(ar.log_probability.shape[0]), sample_size)
            ]
            log_prob_samples += s
            sample_posis = SkyCoord(ra=ar.samples[0] * u.deg, dec=ar.samples[1] * u.deg)
            p = np.argmin(g.position.separation(sample_posis))
            log_prob_truth += ar.log_probability[p]

        except Exception as e:
            logging.info(e)
    fig, ax = plt.subplots(1)
    ax.hist(log_prob_samples, bins=100)
    ylims = ax.get_ylim()
    ax.vlines(log_prob_truth, 0, ylims[1])
    ax.set_ylim(ylims)
    if save_path is not None:
        fig.savefig(save_path)

    return fig, ax
