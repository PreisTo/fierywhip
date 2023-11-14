#!/usr/bin/python
from omegaconf import OmegaConf

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

fierywhip_config = OmegaConf.create(structure)
