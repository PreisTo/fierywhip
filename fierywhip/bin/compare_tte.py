#!/usr/bin/python

from fierywhip.frameworks.grbs import GRB
from fierywhip.model.model import GRBModel
from threeML.analysis_results import load_analysis_results
from mpi4py import MPI
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
import logging

from fierywhip.model.custom_eff_area_tte import CustomEffAreaCorrections



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def compare_free_spectrum(grb, eff_area_dict):
    model_eff = CustomEffAreaCorrections(
        grb,
        fix_position=False,
        use_eff_area=True,
        base_dir=os.path.join(
            os.environ.get("GBMDATA"), "localizing", "comparison", grb.name, "eff"
        ),
        eff_area_dict=eff_area_dict,
        smart_ra_dec=False,
    )
    res_file_eff = os.path.join(
        os.environ.get("GBMDATA"),
        "localizing",
        "comparison",
        grb.name,
        "eff",
        grb.name,
        f"{grb.name}.fits",
    )
    if not os.path.exists(res_file_eff):
        model_eff.fit()
        if rank == 0:
            res_eff = model_eff.results
    else:
        res_eff = load_analysis_results(res_file_eff)

    comm.Barrier()
    model = GRBModel(
        grb,
        fix_position=False,
        use_eff_area=False,
        base_dir=os.path.join(
            os.environ.get("GBMDATA"), "localizing", "comparison", grb.name, "no_eff"
        ),
        smart_ra_dec=False,
    )
    res_file = os.path.join(
        os.environ.get("GBMDATA"),
        "localizing",
        "comparison",
        grb.name,
        "no_eff",
        grb.name,
        f"{grb.name}.fits",
    )

    if not os.path.exists(res_file):
        model.fit()
        if rank == 0:
            res = model.results
    else:
        res = load_analysis_results(res_file)
    if rank == 0:
        return res, res_eff
    else:
        return None,None


def compare_fix_spectrum(grb, eff_area_dict, spectrum):
    base_dir = os.path.join(
        os.environ.get("GBMDATA"), "localizing", "comparison", grb.name
    )
    model_no_eff = CustomEffAreaCorrections(
        grb,
        fix_position=False,
        use_eff_area=False,
        base_dir=os.path.join(base_dir, "no_eff"),
        smart_ra_dec=False,
        fix_spectrum=spectrum,
    )
    model_no_eff.fit()

    comm.Barrier()

    model_eff = CustomEffAreaCorrections(
        grb,
        fix_position=False,
        use_eff_area=True,
        base_dir=os.path.join(base_dir, "eff"),
        smart_ra_dec=False,
        fix_spectrum=spectrum,
        eff_area_dict=eff_area_dict,
    )
    model_eff.fit()
    if rank == 0:
        res_no = model_no_eff.results
        res = model_eff.results
        return res_no, res
    else:
        return None,None


if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")
    name = "GRB231215408" #"GRB230328621" # "GRB130216790"  # "GRB190824616"
    ra = 9.75 #290.986 #58.875000  # 215.329
    dec = 57.64 #80.016 #2.033000  # -41.90

    spectrum = {"index": -1.013, "K": 3.49e-2, "xc": 9879}
    eff_area_dict = {
        "n0": 1,
        "n1": 0.9929363050000001,
        "n2": 0.9514813730000001,
        "n3": 1.0400485800000001,
        "n4": 0.9644063780000001,
        "n5": 1.06590465,
        "n6": 0.979424989,
        "n7": 1.1175595400000002,
        "n8": 0.924959813,
        "n9": 0.94208006,
        "na": 0.9274396269999999,
        "nb": 0.902805862,
    }
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

    # res, res_eff = compare_free_spectrum(grb, eff_area_dict)
    res, res_eff = compare_fix_spectrum(grb, eff_area_dict, spectrum)
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
