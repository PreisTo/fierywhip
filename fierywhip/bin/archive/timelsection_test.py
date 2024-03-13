#!/usr/bin/env python3
from fierywhip.frameworks.grbs import GRBList, GRB
from fierywhip.config.configuration import fierywhip_config
import yaml
import os

if __name__ == "__main__":
    fierywhip_config.config.timeselection.store_and_reload = False
    yaml_path = os.path.join(os.environ.get("HOME"), "ts.yml")
    grblist = GRBList(run_det_sel=False, check_finished=False)
    for g in grblist.grbs:
        g.run_timeselection(max_trigger_duration=30)
        active_time = g.active_time
        bkg_time = g.bkg_time
        res = {}
        res["active_time"] = active_time
        res["bkg_neg"] = bkg_time[0]
        res["bkg_pos"] = bkg_time[1]
        try:
            with open(yaml_path, "r") as f:
                file_content = yaml.safe_load(f)
                file_content[g.name] = res
        except FileNotFoundError:
            file_content = {g.name: res}
            with open(yaml_path, "w+") as f:
                yaml.safe_dump(file_content, f)
