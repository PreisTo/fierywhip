#!/usr/bin/python
from omegaconf import OmegaConf

structure = {}
structure["eff_corr_lims"] = (0.8, 1.2)
structure["live_points"] = 800
structure["live_points_trigdat"] = 1200
structure["max_sep"] = 60
structure["max_sep_norm"] = 40
structure["max_number_det"] = 4
structure["min_number_det"] = 3
fierywhip_config = OmegaConf.create(structure)
