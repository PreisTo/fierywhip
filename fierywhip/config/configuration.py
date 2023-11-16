#!/usr/bin/python
from omegaconf import OmegaConf
import yaml
import pkg_resources

external_config = False
try:
    with open(
        pkg_resources.resource_filename("fierywhip", "config/config.yml"), "r"
    ) as f:
        structure = yaml.safe_load(f)
        external_config = True

except FileNotFoundError:
    pass
if not external_config:
    structure = {}
    structure["eff_corr_lim_low"] = 0.8
    structure["eff_corr_lim_high"] = 1.2
    structure["eff_corr_gaussian"] = True
    structure["live_points"] = 800
    structure["live_points_trigdat"] = 1200
    structure["max_sep"] = 60
    structure["max_sep_norm"] = 40
    structure["max_number_det"] = 3
    structure["min_number_det"] = 3
    structure["ipn"] = True
    structure["swift"] = True
    structure["det_sel"] = {}
    structure["det_sel"]["mode"] = "max_sig"
    structure["timeselection"] = {}
    structure["timeselection"]["save"] = True
    structure["default_plot_path"] = None
fierywhip_config = OmegaConf.create(structure)
