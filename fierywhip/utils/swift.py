#!/usr/bin/env python3

from datetime import datetime, timedelta
import pandas as pd
import os
import pkg_resources
from astropy.coordinates import SkyCoord
import astropy.time as time
import astropy.units as u
import numpy as np
import logging


def check_swift_old(GRB, grb_time):
    file_path = pkg_resources.resource_filename("fierywhip", "data/swift_grbs.txt")
    day_seconds = 24 * 60 * 60
    grb_date = datetime(
        int(f"20{GRB.strip('GRB')[:2]}"),
        int(GRB.strip("GRB")[2:4]),
        int(GRB.strip("GRB")[4:6]),
    )
    grb_time = grb_date + timedelta(seconds=int(GRB[-3:]) * day_seconds / 1000)

    # get swift grb table and look for coinciding
    swift_table = pd.read_csv(
        file_path, sep="\t", decimal=".", encoding="latin-1", index_col=False
    )
    swift_table.insert(1, "Date", [i[0:-1] for i in swift_table["GRB"]], True)
    coinc = swift_table.loc[swift_table["Date"] == GRB.strip("GRB")[:-3]]

    logging.info(f"Total number of {len(coinc['Date'])} Swift trigger(s) found")

    swift_grb = None
    swift_position = None
    swift_candidates = []
    for c in coinc["Time [UT]"]:
        Flag = True
        while Flag:
            try:
                cd = datetime.strptime(c, "%H:%M:%S")
                Flag = False
            except ValueError:
                c = c[:-1]
        cd = cd.replace(year=grb_date.year, month=grb_date.month, day=grb_date.day)
        if grb_time >= cd - timedelta(minutes=2) and grb_time <= cd + timedelta(
            minutes=2
        ):
            swift_candidates.append(
                [
                    coinc.loc[coinc["Time [UT]"] == c],
                    float((grb_time - cd).total_seconds()),
                ]
            )
        else:
            logging.debug(cd)
            logging.debug(grb_time)
            logging.debug((grb_time - cd).total_seconds())
    time_distance = 100
    for i in range(len(swift_candidates)):
        if np.abs(swift_candidates[i][1]) < time_distance:
            swift_grb = swift_candidates[i][0]
    if swift_grb is not None:
        swift_grb = swift_grb.to_dict()

        sgd = list(swift_grb["Date"].keys())
        if len(sgd) == 0:
            return None, None
        logging.debug(f"This is sgd {sgd}")
        # if str(swift_grb["XRT RA (J2000)"][sgd[0]]) != "nan":
        #    ra = swift_grb["XRT RA (J2000)"]
        #    dec = swift_grb["XRT Dec (J2000)"]
        #    swift_position = SkyCoord(
        #        ra=ra[sgd[0]],
        #        dec=dec[sgd[0]],
        #        unit=(u.hourangle, u.deg),
        #    )
        try:
            logging.warning("Only BAT localization available")
            ra = swift_grb["BAT RA (J2000)"]
            dec = swift_grb["BAT Dec (J2000)"]
            swift_position = SkyCoord(
                ra=ra[sgd[0]], dec=dec[sgd[0]], unit=(u.hourangle, u.deg), frame="icrs"
            )
        except Exception as e:
            logging.error(e)
            return None, None
        logging.debug(swift_position)
        return swift_grb, swift_position


def check_swift(GRB, grb_time):
    file_path = pkg_resources.resource_filename("fierywhip", "data/Fermi_Swift.lis")
    csv = pd.read_csv(file_path, header=None, index_col=None, sep=" ")
    g = int(GRB.strip("GRB"))
    sel = csv[csv.iloc[:, 0] == g].index[0]
    ra = str(csv.iloc[sel, 5])
    dec = str(csv.iloc[sel, 6])
    swift_position = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg), frame="icrs")
    swift_grb = sel
    return swift_grb, swift_position
