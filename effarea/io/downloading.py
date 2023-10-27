#!/usr/bin/env python3

from gbmbkgpy.io.downloading import (
    download_gbm_file,
    download_trigdata_file,
    download_files,
)
import os
from mpi4py import MPI
from astropy.utils.data import download_file
import shutil
from urllib.error import HTTPError

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if size > 1:
    using_mpi = True
else:
    using_mpi = False
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


def download_tte_file(trigger, detector="none"):
    """
    Downloads a tte file
    :param trigger: string like bn230903824
    :param detector: string like n0, n1, ...
    :return: returns path to file
    """
    d = detector
    date = trigger.strip("GRB")[:6]
    year = "20%s" % date[:2]
    month = date[2:-2]
    day = date[-2:]
    data_type = "tte"

    data_path = os.environ.get("GBMDATA")
    file_path = os.path.join(data_path, data_type, trigger)

    final_path = os.path.join(
        file_path, f"glg_tte_{d}_bn{trigger.strip('GRB')}_v00.fit"
    )

    if rank == 0:
        try:
            os.makedirs(file_path)
        except FileExistsError:
            pass

        if not os.path.exists(final_path):
            base_url = (
                f"https://heasarc.gsfc.nasa.gov/FTP/fermi/data/"
                f"gbm/triggers/{year}/bn{trigger.strip('GRB')}/current/"
                f"glg_tte_{d}_bn{trigger.strip('GRB')}_v0"
            )

            path_to_file = None
            for version in ["0", "1", "2", "3", "4"]:
                try:
                    path_to_file = download_file(f"{base_url}{version}.fit")
                except HTTPError:
                    pass
                if path_to_file is not None:
                    break

            if path_to_file is None:
                print(f"No version found for the url {base_url}?.fit")
            try:
                shutil.move(path_to_file, final_path)
            except TypeError:
                pass

    if using_mpi:
        comm.Barrier()

    return final_path


def download_cspec_file(trigger, detector="none"):
    """
    Downloads a tte file
    :param trigger: string like bn230903824
    :param detector: string like n0, n1, ...
    :return: returns path to file
    """
    d = detector
    date = trigger.strip("GRB")[:6]
    year = "20%s" % date[:2]
    month = date[2:-2]
    day = date[-2:]
    data_type = "cspec"

    data_path = os.environ.get("GBMDATA")
    file_path = os.path.join(data_path, data_type, trigger)

    final_path = os.path.join(
        file_path, f"glg_cspec_{d}_bn{trigger.strip('GRB')}_v00.pha"
    )

    if rank == 0:
        try:
            os.makedirs(file_path)
        except FileExistsError:
            pass

        if not os.path.exists(final_path):
            base_url = (
                f"https://heasarc.gsfc.nasa.gov/FTP/fermi/data/"
                f"gbm/triggers/{year}/bn{trigger.strip('GRB')}/current/"
                f"glg_cspec_{d}_bn{trigger.strip('GRB')}_v0"
            )

            path_to_file = None
            for version in ["0", "1", "2", "3", "4"]:
                try:
                    path_to_file = download_file(f"{base_url}{version}.pha")
                except HTTPError:
                    pass
                if path_to_file is not None:
                    break

            if path_to_file is None:
                print(f"No version found for the url {base_url}?.pha")
            try:
                shutil.move(path_to_file, final_path)
            except TypeError:
                pass

    if using_mpi:
        comm.Barrier()

    return final_path
