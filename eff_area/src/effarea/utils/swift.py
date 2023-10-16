#!/usr/bin/env python3

from datetime import datetime, timedelta
import pandas as pd
import os
import pkg_resources
from astropy.coordinates import SkyCoord
import astropy.time as time
import astropy.units as u


def check_swift(GRB, grb_time):
    file_path = pkg_resources.resource_filename("effarea", "data/swift_grbs.txt")
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

    print(f"Total number of {len(coinc['Date'])} Swift trigger(s) found")

    swift_grb = None
    swift_position = None
    for c in coinc["Time [UT]"]:
        cd = datetime.strptime(c, "%H:%M:%S")
        cd = cd.replace(year=grb_date.year, month=grb_date.month, day=grb_date.day)
        if grb_time >= cd - timedelta(minutes=2) and grb_time <= cd + timedelta(
            minutes=2
        ):
            swift_grb = coinc.loc[coinc["Time [UT]"] == c]
        else:
            print(cd)
            print(grb_time)
            print((grb_time - cd).total_seconds())
    if swift_grb is not None:
        swift_grb = swift_grb.to_dict()

        if swift_grb["XRT RA (J2000)"] != "nan":
            sgd = list(swift_grb["Date"].keys())
            ra = float(swift_grb["XRT RA (J2000)"])
            dec = float(swift_grb["XRT Dec (J2000)"])
            swift_position = SkyCoord(
                ra=ra[sgd[0]],
                dec=dec[sgd[0]],
                unit=(u.hourangle, u.deg),
            )
            print(swift_position)
        else:
            print("Only BAT localization available")
            ra = float(swift_grb["XRT RA (J2000)"])
            dec = float(swift_grb["XRT Dec (J2000)"])
            swift_position = SkyCoord(
                ra=ra[sgd[0]],
                dec=dec[sgd[0]],
                unit=(u.hourangle, u.deg),
            )
        return swift_grb, swift_position
