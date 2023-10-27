#!/usr/bin/env python3

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


class GRBModel:

    def __init__(self,grb):
