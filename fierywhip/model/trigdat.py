#!/usr/bin/python
from astromodels.functions import Cutoff_powerlaw_Ep, Log_uniform_prior, Uniform_prior
from astromodels.core.model import Model
from astromodels.sources.point_source import PointSource
from morgoth.utils.trig_reader import TrigReader
from morgoth.auto_loc.time_selection import TimeSelectionBB
from gbm_drm_gen.io.balrog_like import BALROGLike
from threeML.utils.spectrum.binned_spectrum import BinnedSpectrumWithDispersion
from threeML.utils.time_series.binned_spectrum_series import BinnedSpectrumSeries
from threeML.utils.spectrum.binned_spectrum_set import BinnedSpectrumSet
from threeML.utils.data_builders.time_series_builder import TimeSeriesBuilder
from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike
from threeML.data_list import DataList
from threeML.bayesian.bayesian_analysis import BayesianAnalysis
from threeML.io.plotting.post_process_data_plots import display_spectrum_model_counts
from threeML.utils.time_interval import TimeIntervalSet
import os
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
lu = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb"]


class GRBModel:
    def __init__(self, grb):
        self._base_dir = os.path.join(os.environ.get("GBM_TRIGGER_DATA_DIR"))
        self.grb = grb
        self._grb_name = self.grb.name
        self.grb.run_timeselection()
        self.trigreader = TrigReader(self.grb.trigdat, fine=True)
        self.trigreader.set_active_time_interval(self.grb.active_time)
        self.trigreader.set_background_selections(*self.grb.bkg_time)
        self._setup_model()
        self._to_plugin()

    def _setup_model(self):
        cpl = Cutoff_powerlaw_Ep()
        cpl.index.value = -1.1
        cpl.K.value = 1
        cpl.xp.value = 200
        cpl.index.prior = Uniform_prior(lower_bound=-2.5, upper_bound=1)
        cpl.K.prior = Log_uniform_prior(lower_bound=1e-4, upper_bound=1000)
        cpl.xp.prior = Uniform_prior(lower_bound=10, upper_bound=10000)

        self._model = Model(
            PointSource(
                "GRB",
                self.grb.position.ra.deg,
                self.grb.position.dec.deg,
                spectral_shape=cpl,
            )
        )

    def _to_plugin(self):
        """

        convert the series to a BALROGLike plugin

        :param detectors: detectors to use
        :return:
        """

        data = []

        for det in self.grb.detector_selection.good_dets:
            # first create a DSL

            speclike = self.trigreader._time_series[det].to_spectrumlike()

            # then we convert to BL

            time = 0.5 * (
                self.trigreader._time_series[det].tstart
                + self.trigreader._time_series[det].tstop
            )

            balrog_like = BALROGLike.from_spectrumlike(speclike, time=time)

            balrog_like.set_active_measurements("c1-c6")
            balrog_like.fix_effective_area_correction(
                self.grb.effective_area_correction(det)
            )
            data.append(balrog_like)
        self._data_list = DataList(*data)

    def fit(self):
        """
        Fit the model to data using multinest
        :return:
        """

        # define bayes object with model and data_list
        self._bayes = BayesianAnalysis(self._model, self._data_list)
        # wrap for ra angle
        wrap = [0] * len(self._model.free_parameters)
        wrap[0] = 1

        # define temp chain save path
        self._temp_chains_dir = os.path.join(self._base_dir, self._grb_name)
        chain_path = os.path.join(self._temp_chains_dir, f"chain_")

        # Make temp chains folder if it does not exists already
        if rank == 0:
            if not os.path.exists(self._temp_chains_dir):
                os.makedirs(os.path.join(self._temp_chains_dir))

        # use multinest to sample the posterior
        # set main_path+trigger to whatever you want to use

        self._bayes.set_sampler("multinest", share_spectrum=True)
        self._bayes.sampler.setup(
            n_live_points=800, chain_name=chain_path, wrapped_params=wrap, verbose=True
        )
        comm.Barrier()
        self._bayes.sample()
        comm.Barrier()
        self.results = self._bayes.results
        self.results.data_list = self._bayes.data_list

    def export_results(self):
        if rank == 0:
            self.results.write_to(
                os.path.join(self._temp_chains_dir, "result.fits"), overwrite=True
            )
            fig = self.results.corner_plot()
            fig.savefig(os.path.join(self._temp_chains_dir, "cplot.pdf"))
            plt.close("all")
            spectrum_plot = display_spectrum_model_counts(self.results)
            ca = spectrum_plot.get_axes()[0]
            y_lims = ca.get_ylim()
            if y_lims[0] < 10e-6:
                ca.set_ylim(bottom=10e-6)
            if y_lims[1] > 10e4:
                ca.set_ylim(top=10e4)
            spectrum_plot.tight_layout()
            spectrum_plot.savefig(
                os.path.join(self._temp_chains_dir, "splot.pdf"),
                bbox_inches="tight",
            )
            plt.close("all")
