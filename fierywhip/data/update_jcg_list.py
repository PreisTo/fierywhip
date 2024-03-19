#!/usr/bin/env python3

import pandas as pd
from astropy.coordinates import SkyCoord
import numpy as np
import urllib.request
from urllib.error import HTTPError
from datetime import datetime
import pkg_resources
import logging
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def update_full_jcg_list(
    path=pkg_resources.resource_filename("fierywhip", "data/full_jcg_list.csv"),
    testing=False,
):
    if rank == 0:
        logging.info(
            "Updating Full JCG List from https://www.mpe.mpg.de/~jcg/grbgen.html"
        )
        logging.info("This may take several minutes")

        df = pd.read_html(
            "https://www.mpe.mpg.de/~jcg/grbgen.html", header=0, index_col=0
        )[0]
        if testing:
            df = df.iloc[0:200]
        logging.debug("Done downloading")
        locs = np.empty(len(df), dtype=SkyCoord)
        for i, g in enumerate(df["GRB X-ray position"]):
            temp = g.replace("°", "d").replace("'", "m")
            temp = temp.split(" ")
            temp_new = temp[0] + " " + temp[1] + temp[2]
            try:
                locs[i] = SkyCoord(temp_new)
            except ValueError:
                temp_new = temp[0] + " " + temp[1] + str(int(temp[2][:-1]) - 10) + "m"
                locs[i] = SkyCoord(temp_new)
        df["position"] = locs
        df["ra"] = np.zeros(len(df))
        df["dec"] = np.zeros(len(df))
        logging.debug("Calculated the positions")
        for x in df.index:
            try:
                df.at[x, "ra"] = df.loc[x]["position"].ra.deg
            except Exception as e:
                pass
            try:
                df.at[x, "dec"] = df.loc[x]["position"].dec.deg
            except Exception as e:
                pass
                logging.debug("Get the necessary part from the website")
    else:
        df = None

    comm.Barrier()
    df = comm.bcast(df, root=0)
    size_per_rank = int(len(df) / size)
    rank_start = size_per_rank * rank
    rank_stop = rank_start + size_per_rank
    if rank == size - 1:
        rank_stop = len(df)
    counter = 0
    df["sod"] = np.zeros(len(df), dtype=float)
    df["seen_by_fermi"] = np.zeros(len(df), dtype=bool)
    day = 24 * 3.6
    logging.debug(f"Rank {rank} will run from {rank_start} to {rank_stop}")
    for g in df.index[rank_start:rank_stop]:
        fermi = False
        sod = 0
        if df.loc[g]["Instrument"] != "Fermi":
            try:
                page = urllib.request.urlopen(
                    f"https://www.mpe.mpg.de/~jcg/grb{g}.html"
                )
                cont = page.read().decode(errors="ignore")
                if "fermi" in cont.lower():
                    if "gbm" in cont.lower():
                        fermi = True
                        if "GRB_TIME" in cont:
                            ind = cont.strip("\t").find("GRB_TIME")
                            sod_float = float(
                                cont[ind + 10 : ind + cont[ind:].find("SOD")].strip(" ")
                            )
                            sod = sod_float
                        counter += 1
                        if counter % 10 == 0:
                            logging.debug(f"{rank}: {counter}")
            except HTTPError:
                pass
            if sod > 0:
                df.at[g, "sod"] = sod
            df.at[g, "seen_by_fermi"] = fermi
    comm.Barrier()
    dfl = comm.gather(df, root=0)
    if rank == 0:
        df = dfl[0]
        for r in range(1, size - 1, 1):
            df.iloc[r * size_per_rank : (r + 1) * size_per_rank] = dfl[r].iloc[
                r * size_per_rank : (r + 1) * size_per_rank
            ]
        df.iloc[(size - 1) * size_per_rank :] = dfl[-1].iloc[
            (size - 1) * size_per_rank :
        ]
    df = comm.bcast(df, root=0)
    logging.debug(type(df))
    logging.debug(df)
    names = np.empty(len(df), dtype=str)
    df["name"] = names
    logging.debug("Done with preparation")
    comm.Barrier()
    if rank == 0:
        logging.debug("Starting to check for availability")
    for x in df.index[rank_start:rank_stop]:
        if df.loc[x]["seen_by_fermi"]:
            n = x[:6]
            sod = df.loc[x]["sod"]
            day = 24 * 3.6
            perc = str(int(round(sod / day, 0))).zfill(3)
            date = datetime.strptime(n, "%y%m%d")
            if date < datetime(2008, 9, 1):
                df.at[x, "seen_by_fermi"] = False
            else:
                found = False
                counter = 0
                adds = [0, -1, 1,-2,2]
                while not found and counter < len(adds):
                    perc = str(int(round(sod / day, 0) + adds[counter])).zfill(3)
                    name = f"bn{date.strftime('%y%m%d')}{perc}"
                    url = f"https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/{date.strftime('%Y')}/{name}/"
                    try:
                        urllib.request.urlopen(url)
                        found = True
                    except HTTPError:
                        counter += 1
                        pass
                if not found:
                    df.at[x, "seen_by_fermi"] = False
                if rank == 0:
                    if found:
                        logging.debug(f"Found {x}")
                df.at[x, "name"] = name
    comm.Barrier()
    dfl = comm.gather(df, root=0)
    logging.debug(dfl)
    if rank == 0:
        df = dfl[0]
        for r in range(1, size - 1, 1):
            df.iloc[r * size_per_rank : (r + 1) * size_per_rank] = dfl[r].iloc[
                r * size_per_rank : (r + 1) * size_per_rank
            ]
        df.iloc[(size - 1) * size_per_rank :] = dfl[-1].iloc[
            (size - 1) * size_per_rank :
        ]
        df.drop(columns="position", inplace=True)
        logging.debug(path)
        logging.debug(df)
        logging.debug(type(df))
        df.to_csv(path)


if __name__ == "__main__":
    logging.getLogger().setLevel("DEBUG")
    update_full_jcg_list()
    if rank == 0:
        path = pkg_resources.resource_filename("fierywhip", "data/full_jcg_list.csv")
        df = pd.read_csv(path, index_col=0)
        df[df["seen_by_fermi"]].to_csv(path)
