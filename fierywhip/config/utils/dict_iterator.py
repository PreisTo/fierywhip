#!/usr/bin/env python3

import logging
from omegaconf import OmegaConf


def recursive_key_finder(dictionary: dict, key: str) -> (bool, str):
    """
    Recurively finds the way to a given key. This does ONLY
    work for unique keys! Otherwise the first appearance will be returned

    The individual keys are concatenated by the & symbol

    :param dictionary: the input dict
    :type dictionary: dict
    :param key: the key which should be found:
    :type key: str

    :returns: (bool,str)
    """

    found = False
    path = ""
    if key in list(dictionary.keys()):
        found = True
        path = key
        return found, path
    for k in dictionary.keys():
        if isinstance(dictionary[k], dict):
            found, p = recursive_key_finder(dictionary[k], key)
        if found:
            if path != "":
                path = p + "&" + path

            else:
                path = p
            path = k + "&" + path
            break

    return found, path


def update_config(config: OmegaConf, update_vals: dict):
    """
    Update given config with a dict of new vals. Key of new vals only needs to
    be the last key, assuming unique key names

    :param config: the config that needs updating
    :type config: OmegaConf
    :param update_vals: the vals for updating
    :type update_vals: dict

    :returns: config
    """
    for key in update_vals.keys():
        flag, path = recursive_key_finder(config, key)
        if flag:
            path = path.split("&")
            temp = config
            assert key == path[-1], "Path to key does not match key"
            for p in path:
                # this works thanks to omegaconf assignig by
                temp = temp[p]
            temp = update_vals[key]
            logging.info(f"Updated {path} with {update_vals[key]}")
        else:
            logging.info("The key %s was not found in the config, skipping", key)

    return config
