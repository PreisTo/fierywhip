#!/usr/bin/env python3

from gbm_localize.trigdat_fit import read_mpe_json_file, fit_trigdat
import json
import urllib.request
import os
from gbmbkgpy.io.downloading import download_trigdata_file
from glob import glob


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


if __name__ == "__main__":
    run_trigger("GRB230913408")
