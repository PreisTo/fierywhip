#!/usr/bin/env python3

from gbm_localize.trigdat_fit import read_mpe_json_file, fit_trigdat
import json
import urllib.request
import os
from gbmbkgpy.io.downloading import download_trigdata_file
from glob import glob
import pandas as pd


def run_trigger(trigger):
    url = f"https://grb.mpe.mpg.de/grb/{trigger}/json"
    with urllib.request.urlopen(url) as json_url:
        data = json.load(json_url)  # [0]["grb_params"][0]
    with open("temp.json", "w+") as f:
        json.dump(data, f)

    trigdat_file = os.path.join(
        os.environ.get("GBMDATA"),
        f"trigdat/20{trigger.strip('GRB')[:2]}",
        f"glg_trigdat_all_bn{trigger.strip('GRB')}_v00.fit",
    )
    if not os.path.isfile(trigdat_file):
        download_trigdata_file(f"bn{trigger.strip('GRB')}")
    trigdat_file = glob(
        os.path.join(
            os.environ.get("GBMDATA"),
            f"trigdat/20{trigger.strip('GRB')[:2]}",
            f"glg_trigdat_all_bn{trigger.strip('GRB')}_v0*.fit",
        )
    )[0]
    print("Done with data loading")
    fit_trigdat(
        str(int(data[0]["grb_params"][0]["trigger_number"])),
        json_file="temp.json",
        trigdat_file=trigdat_file,
    )
    os.remove("temp.json")


def table_prep():
    table = pd.read_csv("table.csv", sep="\t", decimal=".")
    return list(table["Trigger"])


def triggerID2trigger(triggerid):
    url = f"https://gcn.gsfc.nasa.gov/other/{triggerid}.fermi"
    with urllib.request.urlopen(url) as u:
        content = u.read()
        content = content.decode()
    grb_date_id = content.find("GRB_DATE")
    grb_time_id = content.find("GRB_TIME")
    sod_id = content.find("SOD")
    grb_date = content[grb_date_id:grb_time_id].split("\n")
    grb_date = grb_date[0][-8::]
    grb_date = grb_date[0:2] + grb_date[3:5] + grb_date[6:]
    grb_time = content[grb_time_id:sod_id]
    grb_time = grb_time.strip("GRB_TIME:").strip(" ").strip("\t")
    tot_seconds = 24 * 3600
    grb_time = float(grb_time)
    t = grb_time / tot_seconds
    t = str(t).split(".")
    trigger = f"GRB{grb_date}{t[1][:3]}"
    print(trigger)
    return trigger


if __name__ == "__main__":
    # run_trigger("GRB230913408")
    triggers = table_prep()
    for t in triggers:
        trigger = triggerID2trigger(t)
        run_trigger(trigger)
