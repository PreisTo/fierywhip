#!/usr/bin/python

from fierywhip.frameworks.grbs import GRB
from fierywhip.model.model import GRBModel
from threeML.analysis_results import load_analysis_results
from mpi4py import MPI
from astropy.coordinates import SkyCoord
import astropy.units as u
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class CustomEffAreaCorrections(GRBModel):

    def __init__(self):
        super().__init__()
        

if __name__ == "__main__":
    name = "GRB130216790" #"GRB190824616" 
    ra = 58.875000 #215.329
    dec = 2.033000 #-41.90

    timeselection_path = os.path.join("/data/tpeis/test_morgoth/triplets_standard/GBM_TRIGGER_DATA", name, "timeselection.yml"
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
    res_file = os.path.join(os.environ.get("GBMDATA"),"localizing","comparison",name,"no_eff",name,f"{name}.fits")

    #if not os.path.exists(res_file):
    #    model.fit()
    #    res = model.results
    #else:
    res = load_analysis_results(res_file) 


    model_eff = GRBModel(
        grb,
        fix_position=False,
        use_eff_area=True,
        base_dir=os.path.join(
            os.environ.get("GBMDATA"), "localizing", "comparison", name, "eff"
        ),
        smart_ra_dec=False,
    )
    res_file_eff = os.path.join(os.environ.get("GBMDATA"),"localizing","comparison",name,"eff",name,f"{name}.fits")
    #if not os.path.exists(res_file):
    #    model_eff.fit()
    #    res_eff = model_eff.results
    #else:
    res_eff = load_analysis_results(res_file_eff) 

    res.display()
    res_eff.display()

    true_grb = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
    for x in [res,res_eff]:
        ra_fit = x.get_data_frame().loc["GRB.position.ra"]["value"]
        dec_fit = x.get_data_frame().loc["GRB.position.dec"]["value"]
        print(
            true_grb.separation(
                SkyCoord(ra=ra_fit, dec=dec_fit, unit=(u.deg, u.deg))
            ).deg
        )
