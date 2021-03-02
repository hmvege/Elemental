# Scrape site for new spectras that include wavelength and intensity
import sys
import os
from tqdm import tqdm
import lib.utils as utils
import numpy as np


def main():
    output_folder = "spectras2"

    if not os.path.isdir(output_folder):
        print("> mkdir %s" % output_folder)
        os.mkdir(output_folder)

    # # values = utils.element_downloader("U", "scratch/uranium_lines.html")
    # values = utils.element_downloader("U", "scratch/hydrogen_lines.html",
    #                                   return_fmt="dict")

    # elements = np.empty((len(values), 2), dtype=float)
    # for i, val in enumerate(values):
    #     elements[i][0] = val["spectra"]
    #     elements[i][1] = val["intensity"]
    # print(elements)
    # filepath = os.path.join(output_folder, "H") + ".dat"
    # np.savetxt(filepath, elements)
    # exit(1)

    # Returns spectra, energies
    for elem in tqdm(utils.PERIODIC_TABLE):
        values = utils.element_downloader(elem[2], return_fmt="dict")

        # Sorts the elemenets into an array
        elements = np.empty((len(values), 2), dtype=float)
        for i, val in enumerate(values):
            elements[i][0] = val["spectra"]
            elements[i][1] = val["intensity"]

        filepath = os.path.join(output_folder, elem[2]) + ".dat"
        np.savetxt(filepath, elements)
        tqdm.write("Downloaded %s and wrote to file: %s" % (
            elem[0], filepath))

    #     if elem[1] == 2:
    #         exit("Exits after test")
if __name__ == '__main__':
    main()
