#!/usr/bin/env python3


def detector_list():
    lu = [
        "n0",
        "n1",
        "n2",
        "n3",
        "n4",
        "n5",
        "n6",
        "n7",
        "n8",
        "n9",
        "na",
        "nb",
        "b0",
        "b1",
    ]
    return lu


def nai_list():
    lu = detector_list()[:-2]
    return lu


def name_to_id(det):
    lu = {
        "n0": 0,
        "n1": 1,
        "n2": 2,
        "n3": 3,
        "n4": 4,
        "n5": 5,
        "n6": 6,
        "n7": 7,
        "n8": 8,
        "n9": 9,
        "na": 10,
        "nb": 11,
    }
    return lu[det]
