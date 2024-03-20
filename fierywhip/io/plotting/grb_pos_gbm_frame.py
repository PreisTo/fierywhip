#!/usr/bin/env python3

from fierywhip.frameworks.grbs import GRB, GRBList
from gbmgeometry.position_interpolator import PositionInterpolator
from gbmgeometry.gbm_frame import GBMFrame
import yaml
import numpy as np
import matplotlib.pyplot as plt
from gbmgeometry.utils.plotting import skyplot, SphericalCircle
import astropy.units as u
import os

"""
Small Script creating a mollweide plot showing the detectors fov (60 deg)
as well as all the grbs in the sample

Creates also an histogram displaying the distribution of grbs seen by a
given number of dets
"""

dets = {
    "n0": {
        "lon": 45.89,
        "lat": 90 - 20.58,
    },
    "n1": {
        "lon": 45.11,
        "lat": 90 - 45.31,
    },
    "n2": {
        "lon": 58.44,
        "lat": 90 - 90.21,
    },
    "n3": {
        "lon": 314.87,
        "lat": 90 - 45.24,
    },
    "n4": {
        "lon": 303.15,
        "lat": 90 - 90.27,
    },
    "n5": {
        "lon": 3.35,
        "lat": 90 - 89.97,
    },
    "n6": {
        "lon": 224.93,
        "lat": 90 - 20.43,
    },
    "n7": {
        "lon": 224.62,
        "lat": 90 - 46.18,
    },
    "n8": {
        "lon": 236.61,
        "lat": 90 - 89.97,
    },
    "n9": {
        "lon": 135 - 19,
        "lat": 90 - 45.55,
    },
    "nb": {
        "lon": 183.74,
        "lat": 90 - 90.32,
    },
    "na": {
        "lon": 123.73,
        "lat": 90 - 90.42,
    },
}


if __name__ == "__main__":
    if not os.path.exists(os.path.join(os.environ.get("GBMDATA"), "grb_gbm_frame.yml")):
        gl = GRBList(run_det_sel=False, check_finished=False)
        res = {}

        for g in gl.grbs:
            print(g.name)
            pi = PositionInterpolator.from_trigdat(g.trigdat)
            gbm_posi = g.position.transform_to(GBMFrame(**pi.quaternion_dict(0)))
            print(gbm_posi)
            res[g.name] = {
                "lon": float(gbm_posi.lon.deg),
                "lat": float(gbm_posi.lat.deg),
            }

        with open(
            os.path.join(os.environ.get("GBMDATA"), "grb_gbm_frame.yml"), "w+"
        ) as f:
            yaml.dump(res, f)
    with open(os.path.join(os.environ.get("GBMDATA"), "grb_gbm_frame.yml"), "r") as f:
        res = yaml.safe_load(f)

    fig, ax = plt.subplots(1, subplot_kw={"projection": "hammer"})

    for d in dets.keys():
        lon = np.deg2rad(dets[d]["lon"])
        lat = np.deg2rad(dets[d]["lat"])
        if lon > np.pi:
            lon -= 2 * np.pi
        if lat > np.pi / 2:
            lat -= np.pi

        # circle1 = plt.Circle(
        #     (lon, lat), np.radians(30), color="r", fill=True, alpha=0.2
        # )
        # ax.add_artist(circle1)
        phi = np.linspace(0, 2.0 * np.pi, 100)  # 36 points
        r = np.radians(60)
        x = lon + r * np.cos(phi)
        y = lat + r * np.sin(phi)
        ax.plot(x, y, color="r")
        ax.text(lon, lat, d, color="red")
    seen_dets = {}
    for r in res.keys():
        lon = np.deg2rad(res[r]["lon"])
        if lon > np.pi:
            lon -= 2 * np.pi
        lat = np.deg2rad(res[r]["lat"])
        if lat > np.pi / 2:
            lat -= np.pi

        ax.scatter(lon, lat, color="blue", marker=".")

        seen_dets[r] = 0
        for d in dets.keys():
            lon_d = np.deg2rad(dets[d]["lon"])
            lat_d = np.deg2rad(dets[d]["lat"])
            if lon_d > np.pi:
                lon_d -= 2 * np.pi
            if lat_d > np.pi / 2:
                lat_d -= np.pi

            ang_sep = np.arccos(
                np.sin(lat) * np.sin(lat_d)
                + np.cos(lat) * np.cos(lat_d) * np.cos(lon - lon_d)
            )
            ang_sep = np.rad2deg(ang_sep)
            if np.abs(ang_sep) <= 60:
                seen_dets[r] += 1
    ax.grid()
    fig.savefig(os.path.join(os.environ.get("GBMDATA"), "grb_gbm_frame.png"), dpi=600)
    hist_data = []
    for g in seen_dets.keys():
        hist_data.append(seen_dets[g])
    plt.close("all")
    plt.hist(hist_data, bins=np.arange(0, 11, 1))
    plt.title("Nr. of Dets with angular separation <= 60 deg")
    plt.savefig(os.path.join(os.environ.get("GBMDATA"), "dets_seeing_grb.png"), dpi=600)
