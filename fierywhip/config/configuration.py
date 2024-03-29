#!/usr/bin/python
from omegaconf import OmegaConf
import yaml
import pkg_resources
from fierywhip.config.utils.dict_iterator import recursive_key_finder
from fierywhip.config.utils.default_config import *
from fierywhip.config.utils.update_dict import *
import logging


class FierywhipConfig:
    """
    The Config for the whole package
    """

    def __init__(self, structure):
        self._config = OmegaConf.create(structure)

    def update_config(self, update_vals: dict, correct_structure=True):
        """
        Update given config with a dict of new vals. Key of new vals only needs to
        be the last key, assuming unique key names

        :param update_vals: the vals for updating
        :type update_vals: dict

        :returns: config
        """
        if not correct_structure:
            new = self._config.copy()
            for key in update_vals.keys():
                flag, path = recursive_key_finder(self._config, key)
                if flag:
                    path = path.split("&")
                    logging.debug(f"This is the len of the path {len(path)}")
                    if len(path) == 0:
                        new = level1(new.copy(), path, update_vals[key])
                    elif len(path) == 1:
                        new = level2(new.copy(), path, update_vals[key])
                    elif len(path) == 2:
                        new = level3(new.copy(), path, update_vals[key])
                    elif len(path) == 3:
                        new = level4(new.copy(), path, update_vals[key])

                    logging.info(f"Updated {path} with {update_vals[key]}")
                else:
                    logging.info(
                        "The key %s was not found in the config, creating it", key
                    )
                    new[key] = update_vals[key]
            self._config = new
        else:
            new = self._config.copy()
            for k1 in update_vals.keys():
                if isinstance(update_vals[k1], dict):
                    for k2 in update_vals[k1].keys():
                        if isinstance(update_vals[k1][k2], dict):
                            for k3 in update_vals[k1][k2].keys():
                                new[k1][k2][k3] = update_vals[k1][k2][k3]
                        else:
                            new[k1][k2] = update_vals[k1][k2]
                else:
                    new[k1] = update_vals[k1]
            self._config = new

    @property
    def config(self):
        return self._config

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, "r") as f:
            structure = yaml.safe_load(f)
        return cls(structure)

    @classmethod
    def from_default(cls):
        structure = default_complete()
        return cls(structure)


yaml_path = None
try:
    yaml_path = pkg_resources.resource_filename("fierywhip", "config/config.yml")
except FileNotFoundError:
    logging.info("No config.yml found")
    yaml_path = None
if yaml_path is not None:
    try:
        fierywhip_config = FierywhipConfig.from_yaml(yaml_path)
    except FileNotFoundError:
        fierywhip_config = FierywhipConfig.from_default()
else:
    fierywhip_config = FierywhipConfig.from_default()
