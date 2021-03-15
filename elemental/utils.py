import re
import warnings
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import urlopen

import numpy as np

# Periodic table
PERIODIC_TABLE = [
    ("Hydrogen", "1", "H"),
    ("Helium", "2", "He"),
    ("Lithium", "3", "Li"),
    ("Beryllium", "4", "Be"),
    ("Boron", "5", "B"),
    ("Carbon", "6", "C"),
    ("Nitrogen", "7", "N"),
    ("Oxygen", "8", "O"),
    ("Fluorine", "9", "F"),
    ("Neon", "10", "Ne"),
    ("Sodium", "11", "Na"),
    ("Magnesium", "12", "Mg"),
    ("Aluminium", "13", "Al"),
    ("Silicon", "14", "Si"),
    ("Phosphorus", "15", "P"),
    ("Sulfur", "16", "S"),
    ("Chlorine", "17", "Cl"),
    ("Argon", "18", "Ar"),
    ("Potassium", "19", "K"),
    ("Calcium", "20", "Ca"),
    ("Scandium", "21", "Sc"),
    ("Titanium", "22", "Ti"),
    ("Vanadium", "23", "V"),
    ("Chromium", "24", "Cr"),
    ("Manganese", "25", "Mn"),
    ("Iron", "26", "Fe"),
    ("Cobalt", "27", "Co"),
    ("Nickel", "28", "Ni"),
    ("Copper", "29", "Cu"),
    ("Zinc", "30", "Zn"),
    ("Gallium", "31", "Ga"),
    ("Germanium", "32", "Ge"),
    ("Arsenic", "33", "As"),
    ("Selenium", "34", "Se"),
    ("Bromine", "35", "Br"),
    ("Krypton", "36", "Kr"),
    ("Rubidium", "37", "Rb"),
    ("Strontium", "38", "Sr"),
    ("Yttrium", "39", "Y"),
    ("Zirconium", "40", "Zr"),
    ("Niobium", "41", "Nb"),
    ("Molybdenum", "42", "Mo"),
    ("Technetium", "43", "Tc"),
    ("Ruthenium", "44", "Ru"),
    ("Rhodium", "45", "Rh"),
    ("Palladium", "46", "Pd"),
    ("Silver", "47", "Ag"),
    ("Cadmium", "48", "Cd"),
    ("Indium", "49", "In"),
    ("Tin", "50", "Sn"),
    ("Antimony", "51", "Sb"),
    ("Tellurium", "52", "Te"),
    ("Iodine", "53", "I"),
    ("Xenon", "54", "Xe"),
    ("Caesium", "55", "Cs"),
    ("Barium", "56", "Ba"),
    ("Lanthanum", "57", "La"),
    ("Cerium", "58", "Ce"),
    ("Praseodymium", "59", "Pr"),
    ("Neodymium", "60", "Nd"),
    ("Promethium", "61", "Pm"),
    ("Samarium", "62", "Sm"),
    ("Europium", "63", "Eu"),
    ("Gadolinium", "64", "Gd"),
    ("Terbium", "65", "Tb"),
    ("Dysprosium", "66", "Dy"),
    ("Holmium", "67", "Ho"),
    ("Erbium", "68", "Er"),
    ("Thulium", "69", "Tm"),
    ("Ytterbium", "70", "Yb"),
    ("Lutetium", "71", "Lu"),
    ("Hafnium", "72", "Hf"),
    ("Tantalum", "73", "Ta"),
    ("Tungsten", "74", "W"),
    ("Rhenium", "75", "Re"),
    ("Osmium", "76", "Os"),
    ("Iridium", "77", "Ir"),
    ("Platinum", "78", "Pt"),
    ("Gold", "79", "Au"),
    ("Mercury", "80", "Hg"),
    ("Thallium", "81", "Tl"),
    ("Lead", "82", "Pb"),
    ("Bismuth", "83", "Bi"),
    ("Polonium", "84", "Po"),
    ("Astatine", "85", "At"),
    ("Radon", "86", "Rn"),
    ("Francium", "87", "Fr"),
    ("Radium", "88", "Ra"),
    ("Actinium", "89", "Ac"),
    ("Thorium", "90", "Th"),
    ("Protactinium", "91", "Pa"),
    ("Uranium", "92", "U"),
    ("Neptunium", "93", "Np"),
    ("Plutonium", "94", "Pu"),
    ("Americium", "95", "Am"),
    ("Curium", "96", "Cm"),
    ("Berkelium", "97", "Bk"),
    ("Californium", "98", "Cf"),
    ("Einsteinium", "99", "Es"),
    ("Fermium", "100", "Fm"),
    ("Mendelevium", "101", "Md"),
    ("Nobelium", "102", "No"),
    ("Lawrencium", "103", "Lr"),
    ("Rutherfordium", "104", "Rf"),
    ("Dubnium", "105", "Db"),
    ("Seaborgium", "106", "Sg"),
    ("Bohrium", "107", "Bh"),
    ("Hassium", "108", "Hs"),
    ("Meitnerium", "109", "Mt"),
    ("Darmstadtium", "110", "Ds"),
    ("Roentgenium", "111", "Rg"),
    ("Copernicium", "112", "Cn"),
    ("Nihonium", "113", "Nh"),
    ("Flerovium", "114", "Fl"),
    ("Moscovium", "115", "Mc"),
    ("Livermorium", "116", "Lv"),
    ("Tennessine", "117", "Ts"),
    ("Oganesson", "118", "Og"),
]


def element_downloader(
    element: str, local_file: str = None, return_fmt: str = "array"
) -> np.ndarray:
    """Function to retrieve element wavelengths.

    Retrieves the element from either a local file or the NIST webpage.

    Arguments:
        element {str} -- period table element name

    Keyword Arguments:
        local_file {str} -- path to local file (default: {None})
        return_fmt {str} -- type of the return format.
            Options: 'array', 'dict' (default: {"array"})

    Returns:
        np.ndarray -- array containing the element we seek to download.
    """
    if local_file:
        if local_file.split(".")[-1] == "html":
            nist_webpage = local_file
            html = open(nist_webpage, "r")
        else:
            # Trying to get the file as an array
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return np.loadtxt(local_file)
    else:
        nist_webpage = (
            "https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra="
            "%s&limits_type=0&low_w=&upp_w=&unit=1&de=0&format=0&lin"
            "e_out=0&en_unit=1&output=0&page_size=15&show_obs_wl=1&o"
            "rder_out=0&max_low_enrg=&show_av=3&max_upp_enrg=&tsb_va"
            "lue=0&min_str=&A_out=0&intens_out=on&max_str=&allowed_o"
            "ut=1&forbid_out=1&min_accur=&min_intens=&submit=Retriev"
            "e+Data" % element
        )

        html = urlopen(nist_webpage)

    # Strings to match for in html-file
    # The [\W]+ matches all whitespace characters leading up to the
    # wavelength number.
    pre_string_wl = r"\<td class\=\"fix\"\>[\W]+"
    post_string_wl = r"\&nbsp"  # Cut-off at "<".
    search_string_wl = r"%s([\d\W\.]+)%s" % (pre_string_wl, post_string_wl)

    # Search string for the line intensity
    search_string_intensity = r"([\d]+)"

    regex_sequence_wl = re.compile(search_string_wl)
    regex_sequence_intensity = re.compile(search_string_intensity)
    regex_blanks = re.compile(r"[ ]+")

    # List of dictionaries
    retreived_data = []

    # Retrieving spectral and intensity data
    for i, line in enumerate(html):

        if isinstance(line, bytes):
            line = line.decode("utf-8")

        match_wl = regex_sequence_wl.findall(line)
        # match_intensity = regex_sequence_intensity.findall(l)

        if len(match_wl) != 0:

            retreived_data.append(
                {
                    "counter": i,
                    "spectra": float(regex_blanks.sub("", match_wl[0])),
                    "intensity": 0,
                }
            )

        # Only required untill first element is found.
        if len(retreived_data) == 0:
            continue

        if retreived_data[-1]["counter"] + 1 == i:
            match_intensity = regex_sequence_intensity.findall(line)

            # In case there is no associated wavelength intensity, we
            # continue.
            if len(match_intensity) == 0:
                continue

            # Makes sure we only have found a single intensity.
            if len(match_intensity) > 1:
                warning_msg = (
                    "Found multiple intensities, using the first one: "
                    + "[%s]" % ", ".join(match_intensity)
                )
                warnings.warn(warning_msg)

            retreived_data[-1]["intensity"] = float(match_intensity[0])
            continue

    if return_fmt == "array":
        spectras = map(lambda i: i["spectra"], retreived_data)
        return np.array(list(spectras))

    elif return_fmt == "dict":
        return retreived_data

    else:
        raise KeyError(f"{return_fmt} not recognized. Use 'array' or 'dict'.")


def print_element(element: str):
    s = "{0:<15s} {1:<4s} {2:<3s}".format(*element)
    print(s)


def print_all_elements():
    print("{0:<15s} {1:<4s} {2:<3s}".format("Element", "Num", "Id"))
    for element in PERIODIC_TABLE:
        print_element(element)


def check_folder(folder: Path, verbose: bool = True):
    """Checks if folder exists, and if not, creates it.

    Arguments:
        folder {Path} -- folder to check for existence.
        verbose {bool} -- more verbose output.
    """
    if not folder.exists():
        if verbose:
            print(f"Creating output folder: {folder}")
        folder.mkdir()


def element_search(search_element: str) -> Optional[str]:
    """
    Searching the periodic table for elements. Returns false if desired
    element is not found.
    """
    for element in PERIODIC_TABLE:
        if search_element == element[-1]:
            return search_element
    return False


def get_element(search_element: str) -> Optional[Tuple[int, str, str]]:
    """
    Searching the periodic table for elements. Returns false if desired
    element is not found.
    """
    for element in PERIODIC_TABLE:
        if search_element == element[-1]:
            return element
    return False
