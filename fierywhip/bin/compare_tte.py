#!/usr/bin/python

from fierywhip.frameworks.grbs import GRB
from fierywhip.model.model import GRBModel
from mpi4py import MPI
from astropy.coordinates import SkyCoord
import astropy.units as u
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if __name__ == "__main__":
    name = "GRB091010113"
    ra = 298.667
    dec = -22.533

    timeselection_path = os.path.join(
        os.environ.get("GBM_TRIGGER_DATA_DIR"), name, "timeselection.yml"
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
        smart_ra_dec=False,
        base_dir=os.path.join(
            os.environ.get("GBMDATA"), "localizing", "comparison", name, "no_eff"
        ),
    )
    comm.Barrier()
    model.fit()

    comm.Barrier()

    model_eff = GRBModel(
        grb,
        fix_position=False,
        use_eff_area=True,
        smart_ra_dec=False,
        base_dir=os.path.join(
            os.environ.get("GBMDATA"), "localizing", "comparison", name, "eff"
        ),
    )

    comm.Barrier()
    model_eff.fit()

    comm.Barrier()

    model.results.display()
    model_eff.results.display()

    true_grb = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
    for x in [model, model_eff]:
        ra_fit = x.results.get_data_frame().loc["GRB.position.ra"]["value"]
        dec_fit = x.results.get_data_frame().loc["GRB.position.dec"]["value"]
        print(
            true_grb.separation(
                SkyCoord(ra=ra_fit, dec=dec_fit, unit=(u.deg, u.deg))
            ).deg
        )
