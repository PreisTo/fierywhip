#!/usr/bin/env python3


def level1(config: dict, path: list, val) -> dict:
    config[path[0]] = val
    return config


def level2(config: dict, path: list, val) -> dict:
    config[path[0]][path[1]] = val
    return config


def level3(config: dict, path: list, val) -> dict:
    config[path[0]][path[1]][path[2]] = val
    return config


def level4(config: dict, path: list, val) -> dict:
    config[path[0]][path[1]][path[2]][path[3]] = val
    return config
