#!/usr/bin/python

from fierywhip.frameworks.grbs import GRB
from fierywhip.model.model import GRBModel
from fierywhip.config.configuration import fierywhip_config
from threeML.analysis_results import load_analysis_results
from mpi4py import MPI
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
import logging
from gbm_drm_gen.io.balrog_like import BALROGLike
from threeML.data_list import DataList


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class CustomEffAreaCorrections(GRBModel):

    def __init__(self,grb,fix_position=False,use_eff_area=True,
        base_dir=None,
        eff_area_dict=None,
        smart_ra_dec=False):
        self._eff_area_dict = eff_area_dict
        super().__init__(grb,fix_position=fix_position,use_eff_area=use_eff_area,base_dir=base_dir,smart_ra_dec=smart_ra_dec)


    def _to_plugin(self):
        if self._fix_position:
            free_position = False
        else:
            free_position = True
        active_time = self.grb.active_time
        active_time = active_time.split("-")
        if len(active_time) == 2:
            start = float(active_time[0])
            stop = float(active_time[-1])
        if len(active_time) == 3:
            start = -float(active_time[1])
            stop = float(active_time[-1])
        elif len(active_time) == 4:
            start = -float(active_time[1])
            stop = -float(active_time[-1])
        assert start < stop, "start is after stop"
        response_time = (float(start) + float(stop)) / 2
        spectrum_likes = []
        for d in self.grb.detector_selection.good_dets:
            if self._timeseries[d]._name not in ("b0", "b1"):
                spectrum_like = self._timeseries[d].to_spectrumlike()
                spectrum_like.set_active_measurements("10-500")
                spectrum_likes.append(spectrum_like)
            else:
                spectrum_like = self._timeseries[d].to_spectrumlike()
                spectrum_like.set_active_measurements("300-30000")
                spectrum_likes.append(spectrum_like)
        balrog_likes = []
        logging.info(f"We are going to use {self.grb.detector_selection.good_dets}")
        for i, d in enumerate(self.grb.detector_selection.good_dets):
            if d not in ("b0", "b1"):
                bl = BALROGLike.from_spectrumlike(
                    spectrum_likes[i],
                    response_time,
                    self._responses[d],
                    free_position=free_position,
                )
                if self._use_eff_area:
                    bl.fix_effective_area_correction(self._eff_area_dict[d])
                balrog_likes.append(bl)
        self._data_list = DataList(*balrog_likes)
        if self._save_lc:
            for d in self.grb.detector_selection.good_dets:
                fig = self._timeseries[d].view_lightcurve()
                plot_path = os.path.join(self._base_dir, self.grb.name, "lightcurves/")
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path)
                fig.savefig(os.path.join(plot_path, f"{d}.pdf"))


if __name__ == "__main__":
    name = "GRB130216790"  # "GRB190824616"
    ra = 58.875000  # 215.329
    dec = 2.033000  # -41.90
    eff_area_dict = {"n0":1,"n1": 0.9929363050000001,
             "n2": 0.9514813730000001,
              "n3": 1.0400485800000001,
               "n4": 0.9644063780000001,
                "n5": 1.06590465,
                 "n6": 0.979424989,
                  "n7": 1.1175595400000002,
                   "n8": 0.924959813,
                    "n9": 0.94208006,
                     "na": 0.9274396269999999,
                      "nb": 0.902805862}
    timeselection_path = os.path.join(
        "/data/tpeis/test_morgoth/triplets_standard/GBM_TRIGGER_DATA",
        name,
        "timeselection.yml",
    )

    grb = GRB(name=name, ra=ra, dec=dec)
    try:
        grb.timeselection_from_yaml(timeselection_path)
    except FileNotFoundError:
        pass

    model = GRBModel(
        grb,
        fix_position=False,
        use_eff_area=False,
        base_dir=os.path.join(
            os.environ.get("GBMDATA"), "localizing", "comparison", name, "no_eff"
        ),
        smart_ra_dec=False,
    )
    res_file = os.path.join(
        os.environ.get("GBMDATA"),
        "localizing",
        "comparison",
        name,
        "no_eff",
        name,
        f"{name}.fits",
    )

    #if not os.path.exists(res_file):
    model.fit()
    res = model.results
    #else:
    #    res = load_analysis_results(res_file)

    model_eff = CustomEffAreaCorrections(
        grb,
        fix_position=False,
        use_eff_area=True,
        base_dir=os.path.join(
            os.environ.get("GBMDATA"), "localizing", "comparison", name, "eff"
        ),
        eff_area_dict=eff_area_dict,
        smart_ra_dec=False,
    )
    res_file_eff = os.path.join(
        os.environ.get("GBMDATA"),
        "localizing",
        "comparison",
        name,
        "eff",
        name,
        f"{name}.fits",
    )
    #if not os.path.exists(res_file):
    model_eff.fit()
    res_eff = model_eff.results
    #else:
    #    res_eff = load_analysis_results(res_file_eff)
    comm.Barrier()
    if rank == 0:
        res.display()
        res_eff.display()

        true_grb = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
        for x in [res, res_eff]:
            ra_fit = x.get_data_frame().loc["GRB.position.ra"]["value"]
            dec_fit = x.get_data_frame().loc["GRB.position.dec"]["value"]
            print(
                true_grb.separation(
                    SkyCoord(ra=ra_fit, dec=dec_fit, unit=(u.deg, u.deg))
                ).deg
            )
