#!/usr/bin/env python3

from astropy.coordinates import SkyCoord
from gbmgeometry.gbm_frame import GBMFrame
from gbmgeometry.gbm import GBM
from gbmgeometry.utils.gbm_time import GBMTime
from astropy import units as u
from gbmgeometry.position_interpolator import PositionInterpolator


def calc_angular_incident(grb_position, gbm, gbm_time, interpolator):
    assert type(grb_position) is SkyCoord, "grb_position has to be SkyCoord"
    assert type(gbm) is GBM, "gbm_frame has to be GBM"
    assert type(gbm_time) is GBMTime, "gbm_time has to be GBMTime"
    assert (
        type(interpolator) is PositionInterpolator
    ), "interpolator has to be PositionInterpolator"
    quats = interpolator.quaternion(gbm_time.met)
    sc_pos = interpolator.sc_pos(gbm_time.met)
    gbm_frame = GBMFrame(
        quaternion_1=quats[0],
        quaternion_2=quats[1],
        quaternion_3=quats[2],
        quaternion_4=quats[3],
        sc_pos_X=sc_pos[0],
        sc_pos_Y=sc_pos[1],
        sc_pos_Z=sc_pos[2],
    )

    use_dets = gbm.get_good_fov(grb_position, 60, fermi_frame=False)[1]

    return_dict = {}
    for det_name, det in gbm.detectors.items():
        return_dict[det_name] = {}
        return_dict["grb"] = {}
        return_dict["grb"]["lon"] = float(grb_position.transform_to(gbm_frame).lon.deg)
        return_dict["grb"]["lat"] = float(grb_position.transform_to(gbm_frame).lat.deg)
        lon = float(
            grb_position.transform_to(gbm_frame).lon.deg
            - det.get_center().transform_to(gbm_frame).lon.deg
        )
        lat = float(
            grb_position.transform_to(gbm_frame).lat.deg
            - det.get_center().transform_to(gbm_frame).lat.deg
        )
        return_dict[det_name]["lon"] = lon
        return_dict[det_name]["lat"] = lat
    return return_dict, use_dets


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
