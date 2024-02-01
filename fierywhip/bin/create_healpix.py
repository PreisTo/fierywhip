#!/usr/bin/env python3

import os
from fierywhip.healpix.healpix import MorgothHealpix
import re
import logging


def run(path, output_path):
    ls = os.listdir(path)
    for l in ls:
        if re.search("GRB[0-9]{9}", l) is not None and os.path.isdir(
            os.path.join(path, l)
        ):
            logging.info(f"Found match in {l}")
            grb = l
            result_file = os.path.join(
                path, l, "trigdat", "v00", "trigdat_v00_loc_results.fits"
            )
            hp = MorgothHealpix.from_result_fits_file(result_file=result_file)
            hp.save_healpix_map(path=os.path.join(output_path, f"grb.fits"))


if __name__ == "__main__":
    path = "/data/tpreis/test_morgoth/website_version/trigger_data_dir/GBM_TRIGGER_DATA_DIR"
    output_path = "/data/tpreis/test_morgoth/website_version/healpix/"
    run(path, output_path)
