"""Spectra retriever

Scrape site for new spectras that include wavelength and intensity
"""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from .utils import check_folder, PERIODIC_TABLE, element_downloader


def spectra_retriever(
    output_folder: Path = Path("spectras2"),
    force_redownload: bool = False,
):
    """Downloads spectra.

    Downloads atomic spectra from nist.gov.

    Keyword Arguments:
        force_redownload {bool}
        output_folder {Path} -- saves the downloaded spectra in given folder.
            (default: {Path("spectras2")})
    """

    check_folder(output_folder)

    # Returns spectra, energies
    for elem in tqdm(PERIODIC_TABLE):

        filepath = output_folder / f"{elem[2]}.dat"

        if filepath.exists() and not force_redownload:
            tqdm.write(f"{filepath} already exist.")
            continue

        values = element_downloader(elem[2], return_fmt="dict")

        # Sorts the elements into an array
        elements = np.empty((len(values), 2), dtype=float)
        for i, val in enumerate(values):
            elements[i][0] = val["spectra"]
            elements[i][1] = val["intensity"]

        np.savetxt(filepath, elements)
        tqdm.write(f"Downloaded {elem[0]} and wrote to file: {filepath}")
