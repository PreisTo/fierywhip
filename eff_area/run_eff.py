#!/usr/bin/env python3

from gbmbkgpy.io.downloading import download_gbm_file, download_trigdata_file
from gbmgeometry import PositionInterpolator, GBM
from astropy.coordinates import SkyCoord
from gbmgeometry.utils.gbm_time import GBMTime
import astropy.time as time
import astropy.units as u
import pandas as pd
from datetime import datetime,timedelta
from morgoth.utils.trig_reader import TrigReader
from morgoth.auto_loc.time_selection import TimeSelectionBB
from astromodels.functions import Powerlaw, Cutoff_powerlaw, Band
from astromodels.sources.point_source import PointSource
from astromodels.functions.priors import Log_uniform_prior,Uniform_prior
from astromodels.core.model import Model
from threeML.data_list import DataList
from threeML.bayesian.bayesian_analysis import BayesianAnalysis,BayesianResults
import os


lu = ["n0","n1","n2","n3","n4","n5","n6","n7","n8","n9","na","nb","b0","b1"]

GRB = "GRB230826814"
day_seconds = 24*60*60
grb_date = datetime(int(f"20{GRB.strip('GRB')[:2]}"),int(GRB.strip("GRB")[2:4]),int(GRB.strip("GRB")[4:6]))
grb_time = grb_date + timedelta(seconds=int(GRB[-3:])*day_seconds/1000)

# get swift grb table and look for coinciding
swift_table = pd.read_csv("swift_grbs.txt", sep = "\t", decimal = ".", encoding='latin-1', index_col=False)
swift_table.insert(1, "Date", [i[0:-1] for i in swift_table["GRB"]], True)
coinc = swift_table.loc[swift_table["Date"] == GRB.strip("GRB")[:-3]]

print(f"Total number of {len(coinc['Date'])} Swift trigger(s) found")

swift_grb = None
for c in coinc["Time [UT]"]:
    cd = datetime.strptime(c,"%H:%M:%S")
    cd = cd.replace(year = grb_date.year, month = grb_date.month, day = grb_date.day)
    if grb_time >= cd - timedelta(minutes = 2) and grb_time <= cd + timedelta(minutes = 2):
        swift_grb = coinc.loc[coinc["Time [UT]"] == c]
    else:
        print(cd)
        print(grb_time)
        print((grb_time - cd).total_seconds())

if swift_grb is not None:
    swift_grb = swift_grb.to_dict()

    poshist = os.path.join(os.environ.get("GBMDATA"), "poshist",GRB.strip("GRB")[:-3],f"glg_poshist_all_{GRB.strip('GRB')[:-3]}_v00.fit")
    trigdat = os.path.join(os.environ.get("GBMDATA"),"trigdat",str(grb_time.year),f"glg_trigdat_all_bn{GRB.strip('GRB')}_v00.fit")
    if not os.path.exists(poshist):
        download_gbm_file(date = GRB.strip("GRB")[:-3],data_type="poshist")
        print("Done downloading poshist")
    if not os.path.exists(trigdat):
        download_trigdata_file(f"bn{GRB.strip('GRB')}")
        print("Done downloading trigdat")
    interpolator = PositionInterpolator.from_poshist(poshist)

    if swift_grb["XRT RA (J2000)"] != "n/a":
        sgd = list(swift_grb["Date"].keys())
        swift_position = SkyCoord(ra = swift_grb["XRT RA (J2000)"][sgd[0]], dec= swift_grb["XRT Dec (J2000)"][sgd[0]],unit = (u.hourangle,u.deg))
        print(swift_position)
    else:
        print("Only BAT localization available")
    interp_trigdat = PositionInterpolator.from_trigdat(trigdat)

t0 = time.Time(grb_time, format="datetime", scale="utc")
gbm_time = GBMTime(t0)
gbm = GBM(
    interpolator.quaternion(gbm_time.met),
    sc_pos=interpolator.sc_pos(gbm_time.met) * u.km,
)
sep = gbm.get_separation(swift_position)
for d in lu:
    print(d,sep[d])

tsbb=TimeSelectionBB(GRB,trigdat,fine=True)

trigreader = tsbb.trigreader_object
balrog_plugin = trigreader.to_plugin(*lu)
#for i,d in enumerate(balrog_plugin):
#    balrog_plugin[i] = d.use_effective_area_correction(0.7,1.3)
datalist = DataList(*balrog_plugin)
for d in lu:
    datalist[d].use_effective_area_correction(0.7,1.3)
cpl = Cutoff_powerlaw()
cpl.K.prior = Log_uniform_prior(lower_bound=0.0001, upper_bound=500)
cpl.xc.prior = Log_uniform_prior(lower_bound=10, upper_bound=1000)
cpl.index.set_uninformative_prior(Uniform_prior)
ps = PointSource("grb",ra = float(swift_position.ra.deg), dec = float(swift_position.dec.deg),spectral_shape=cpl)
ps.position.fix = True
ps.position.free = False
model = Model(ps)
bayes = BayesianAnalysis(model,datalist)
bayes.set_sampler("multinest",share_spectrum = True)
wrap = [0] * len(model.free_parameters)  # not working properlyViel Erfolg und Spaß in den ersten Wochen!
wrap[0] = 1

bayes.sampler.setup(n_live_points=400, wrapped_params=wrap, chain_name="fit_", verbose=True)
bayes.sample()

bayes.results.write_to(
        result_path + "/grb_" + trigger + "_" + spectrum + ".fits", overwrite=True
    )

results = bayes.results

result_path = "."
# spectrum and residuals
try:
    spectrum_plot = display_spectrum_model_counts(bayes)
    spectrum_plot.savefig(
        result_path + "/grb_" + trigger + "_spectrum_" + spectrum + ".pdf"
    )
except Exception as e:
    print(f"Spectrum plot not possible due to {e}")
# corner plot

cc_plot = results.corner_plot_cc()
cc_plot.savefig(
    result_path + "/grb_" + trigger + "_cornerplot_" + spectrum + ".pdf"
)
