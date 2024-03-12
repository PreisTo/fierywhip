#!/usr/bin/env python3


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
