#!/usr/bin/env python

from fierywhip.data.grbs import GRB
import json
from urllib.request import urlopen
from astropy.coordinates import SkyCoord
import astropy.units as u


class BalrogLocalization:
    """
    Class to retrieve Balrog locations from website
    """

    def __init__(self, grb: GRB):
        assert isinstance(
            grb, GRB
        ), "grb has to be an instance of fierywhip.data.grbs GRB"
        self._grb = grb
        self._exists = self.check_balrog_exists()
        if self._exists:
            self.load_json()
            self.balrog_separation()
        else:

    def check_balrog_exists(self):
        url = f"https://grb.mpe.mpg.de/grb/{self._grb.name}/json"
        response = urlopen(url)
        self._json_website = json.loads(response.read().decode("utf-8"))
        if len(self._json_website) > 0:
            return True
        else:
            return False

    def load_json(self):
        version_dict = self._json_website[0]["grb_params"]
        error = 360
        version = 0
        for v in range(len(version_dict)):
            if error >= version_dict[v]["balrog_one_sig_err_circle"]:
                if "tte" not in version_dict[v]["version"]:
                    version = v
                    error = version_dict[v]["balrog_one_sig_err_circle"]
        print(
            f"Using {version_dict[version]['version']} out of {[version_dict[i]['version'] for i in range(len(version_dict))]}"
        )

        self._grb_dict = version_dict[version]
        self._balrog_position = SkyCoord(
            ra=self._grb_dict["balrog_ra"],
            dec=self._grb_dict["balrog_dec"],
            unit=[u.deg, u.deg],
            frame="icrs",
        )

    def balrog_separation(self):
        self._separation = self._grb.position.separation(self._balrog_position)

    @property
    def grb_dict(self):
        return self._grb_dict

    @property
    def balrog_position(self):
        return self._balrog_position

    @property
    def separation(self):
        return self._separation

    @property
    def balrog_exists(self):
        return self._exists
    def save_to_csv(self):
        with open(self._csv_path,"r") as f:
            raise NotImplementedError
