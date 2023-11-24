#!/usr/bin/env python3

from fierywhip.data.grbs import GRB
import json
from urllib.request import urlopen


class BalrogLocalization:
    """
    Class to retrieve Balrog locations from website
    """

    def __init__(self, grb: GRB):
        assert isinstance(
            grb, GRB
        ), "grb has to be an instance of fierywhip.data.grbs GRB"
        self._grb = grb

    def _check_if_grb_exists(self):
        url = f"https://grb.mpe.mpg.de/grb/{self.grb.name}/json"
        response = urlopen(url)
        json_website = json.loads(response.read().decode("utf-8"))
        version_dict = json_website[0]["grb_params"]
        error = 360
        version = list(version_dict.keys())[0]
        for v in version_dict.keys():
            if error >= version_dict[v]["balrog_one_sig_err_circle"]:
                version = v
                error = version_dict[v]["balrog_one_sig_err_circle"]
        print(f"Using {version} out of {version_dict.keys()}")

        self._grb_dict = version_dict[version]

        @property
        def grb_dict(self):
            return self._grb_dict
