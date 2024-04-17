#!/usr/bin/env python3


def default_timeselection(structure: dict) -> dict:
    structure["timeselection"] = {}
    structure["timeselection"]["save"] = False
    structure["timeselection"]["store_and_reload"] = False
    structure["timeselection"]["max_trigger_duration"] = 11
    structure["timeselection"]["min_trigger_duration"] = 0.064
    structure["timeselection"]["min_bkg_time"] = 45
    structure["timeselection"]["min_bb_block_bkg_duration"] = 8
    structure["timeselection"]["trigger_zone_bkg_start"] = -5
    structure["timeselection"]["trigger_zone_bkg_stop"] = 10
    structure["timeselection"]["trigger_zone_active_start"] = -10
    structure["timeselection"]["trigger_zone_active_stop"] = 60
    structure["timeselection"]["max_factor"] = 1.2
    structure["timeselection"]["sig_reduce_factor"] = 0.8
    structure["timeselection"]["replace_n5"] = True
    return structure


def default_fit_settings(structure: dict) -> dict:
    structure["live_points"] = 800
    structure["live_points_trigdat"] = 1200
    structure["mpiexec_path"] = "/usr/bin/mpiexec"
    structure["multinest_nr_cores"] = 8
    structure["long_grb_duration"] = 11
    return structure


def default_data_loading(structure: dict) -> dict:
    structure["ipn"] = {}
    structure["ipn"]["small"] = False
    structure["ipn"]["full"] = False
    structure["swift"] = True
    structure["full_list"] = True
    structure["grb_list"] = {"create_objects": True}
    structure["grb_list"]["check_finished"] = True
    structure["grb_list"]["run_det_sel"] = True
    structure["grb_list"]["testing"] = False
    structure["grb_list"]["reverse"] = False
    structure["grb_list"]["create_objects"] = True
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
    structure["eff_area_correction"] = {}
    structure["eff_area_correction"]["use_eff_corr"] = False
    structure["eff_area_correction"]["eff_corr_lim_low"] = 0.8
    structure["eff_area_correction"]["eff_corr_lim_high"] = 1.2
    structure["eff_area_correction"]["eff_corr_gaussian"] = True
    return structure


def default_exporting(structure: dict) -> dict:
    structure["default_plot_path"] = None
    structure["comparison"] = {}
    structure["comparison"]["csv_path"] = None
    structure["comparison"]["csv_name"] = None
    return structure


def default_tte_stuff(structure: dict) -> dict:
    structure["tte"] = {}
    structure["tte"]["fix_position"] = False
    return structure


def default_trigdat_modelling(structure):
    structure["trigdat"] = {}
    cpl = {}
    cpl["index_prior_bounds"] = None
    cpl["xc_prior_bounds"] = [1, 10000]
    cpl["k_prior_bounds"] = [1e-3, 10**4]
    cpl["smart_ra_dec_init"] = False
    structure["trigdat"]["cpl"] = cpl
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
    structure = default_tte_stuff(structure)
    structure = default_eff_correction(structure)
    structure = default_trigdat_modelling(structure)
    return structure
