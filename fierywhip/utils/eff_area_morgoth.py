#!/usr/bin/env python3

from morgoth.auto_loc.utils.fit import MultinestFitTrigdat


class MultinestFitTrigdatEffArea(MultinestFitTrigdat):
    def __init__(
        self,
        grb_name,
        version,
        trigdat_file,
        bkg_fit_yaml_file,
        time_selection_yaml_file,
    ):
        super().__init__(
            grb_name, version, trigdat_file, bkg_fit_yaml_file, time_selection_yaml_file
        )
