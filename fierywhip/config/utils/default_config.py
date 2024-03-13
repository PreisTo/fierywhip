#!/usr/bin/env python3


def default_timeselection(structure: dict) -> dict:
    structure["timeselection"] = {}
    structure["timeselection"]["save"] = False
    structure["timeselection"]["store_and_reload"] = False
    return structure


def default_fit_settings(structure: dict) -> dict:
    structure["live_points"] = 800
    structure["live_points_trigdat"] = 1200
    return structure


def default_data_loading(structure: dict) -> dict:
    structure["ipn"] = {}
    structure["ipn"]["small"] = False
    structure["ipn"]["full"] = False
    structure["swift"] = True
    structure["grb_list"] = {"create_objects": True}
    return structure


def default_det_sel(structure: dict) -> dict:
    structure["det_sel"] = {}
    structure["det_sel"]["mode"] = "max_sig"
    structure["det_sel"]["exclude_blocked_dets"] = False
    structure["max_sep"] = 60
    structure["max_sep_norm"] = 40
    structure["max_number_det"] = 3
    structure["min_number_det"] = 3
    return structure


def default_eff_correction(structure: dict) -> dict:
    structure["eff_corr_lim_low"] = 0.8
    structure["eff_corr_lim_high"] = 1.2
    structure["eff_corr_gaussian"] = True
    return structure


def default_exporting(structure: dict) -> dict:
    structure["default_plot_path"] = None
    structure["comparison"] = {}
    structure["comparison"]["csv_path"] = None
    structure["comparison"]["csv_name"] = None
    return structure


def default_complete() -> dict:
    """
    Creates and returns the full default dict
    """

    structure = {}
    structure = default_timeselection(structure)
    structure = default_det_sel(structure)
    structure = default_data_loading(structure)
    structure = default_fit_settings(structure)
    structure = default_exporting(structure)
    return structure
