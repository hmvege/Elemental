from pathlib import Path
from typing import List, Tuple

import click
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import rcParams
from scipy.interpolate import interp1d
from tqdm import tqdm

from elemental.elemental import Elemental
from elemental.utils import element_search, check_folder
from get_viable_elements import retrieve_viable_elements


# rcParams['text.usetex'] = True
# rcParams['font.family'] = "serif"


CIE = np.array(
    [
        [0.0014, 0.0000, 0.0065],
        [0.0022, 0.0001, 0.0105],
        [0.0042, 0.0001, 0.0201],
        [0.0076, 0.0002, 0.0362],
        [0.0143, 0.0004, 0.0679],
        [0.0232, 0.0006, 0.1102],
        [0.0435, 0.0012, 0.2074],
        [0.0776, 0.0022, 0.3713],
        [0.1344, 0.0040, 0.6456],
        [0.2148, 0.0073, 1.0391],
        [0.2839, 0.0116, 1.3856],
        [0.3285, 0.0168, 1.6230],
        [0.3483, 0.0230, 1.7471],
        [0.3481, 0.0298, 1.7826],
        [0.3362, 0.0380, 1.7721],
        [0.3187, 0.0480, 1.7441],
        [0.2908, 0.0600, 1.6692],
        [0.2511, 0.0739, 1.5281],
        [0.1954, 0.0910, 1.2876],
        [0.1421, 0.1126, 1.0419],
        [0.0956, 0.1390, 0.8130],
        [0.0580, 0.1693, 0.6162],
        [0.0320, 0.2080, 0.4652],
        [0.0147, 0.2586, 0.3533],
        [0.0049, 0.3230, 0.2720],
        [0.0024, 0.4073, 0.2123],
        [0.0093, 0.5030, 0.1582],
        [0.0291, 0.6082, 0.1117],
        [0.0633, 0.7100, 0.0782],
        [0.1096, 0.7932, 0.0573],
        [0.1655, 0.8620, 0.0422],
        [0.2257, 0.9149, 0.0298],
        [0.2904, 0.9540, 0.0203],
        [0.3597, 0.9803, 0.0134],
        [0.4334, 0.9950, 0.0087],
        [0.5121, 1.0000, 0.0057],
        [0.5945, 0.9950, 0.0039],
        [0.6784, 0.9786, 0.0027],
        [0.7621, 0.9520, 0.0021],
        [0.8425, 0.9154, 0.0018],
        [0.9163, 0.8700, 0.0017],
        [0.9786, 0.8163, 0.0014],
        [1.0263, 0.7570, 0.0011],
        [1.0567, 0.6949, 0.0010],
        [1.0622, 0.6310, 0.0008],
        [1.0456, 0.5668, 0.0006],
        [1.0026, 0.5030, 0.0003],
        [0.9384, 0.4412, 0.0002],
        [0.8544, 0.3810, 0.0002],
        [0.7514, 0.3210, 0.0001],
        [0.6424, 0.2650, 0.0000],
        [0.5419, 0.2170, 0.0000],
        [0.4479, 0.1750, 0.0000],
        [0.3608, 0.1382, 0.0000],
        [0.2835, 0.1070, 0.0000],
        [0.2187, 0.0816, 0.0000],
        [0.1649, 0.0610, 0.0000],
        [0.1212, 0.0446, 0.0000],
        [0.0874, 0.0320, 0.0000],
        [0.0636, 0.0232, 0.0000],
        [0.0468, 0.0170, 0.0000],
        [0.0329, 0.0119, 0.0000],
        [0.0227, 0.0082, 0.0000],
        [0.0158, 0.0057, 0.0000],
        [0.0114, 0.0041, 0.0000],
        [0.0081, 0.0029, 0.0000],
        [0.0058, 0.0021, 0.0000],
        [0.0041, 0.0015, 0.0000],
        [0.0029, 0.0010, 0.0000],
        [0.0020, 0.0007, 0.0000],
        [0.0014, 0.0005, 0.0000],
        [0.0010, 0.0004, 0.0000],
        [0.0007, 0.0002, 0.0000],
        [0.0005, 0.0002, 0.0000],
        [0.0003, 0.0001, 0.0000],
        [0.0002, 0.0001, 0.0000],
        [0.0002, 0.0001, 0.0000],
        [0.0001, 0.0000, 0.0000],
        [0.0001, 0.0000, 0.0000],
        [0.0001, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000],
    ]
)


SPECTRA_GENERATION_OPTIONS = [
    "NTSC",
    "EBU",
    "SMPTE",
    "HDTV",
    "CIE",
    "Rec709",
    "AdobeRBG1998",
    "SRGB",
    "empirical",
]

# Plotting parameters
OUTPUT_DPI = 300
OUTPUT_SIZE = (6.6, 3.6)


class ColorSystem:
    """Class for the color systems."""

    def __init__(
        self,
        system_name: str,
        x_red: float,
        y_red: float,
        x_green: float,
        y_green: float,
        x_blue: float,
        y_blue: float,
        x_white: float,
        y_white: float,
        gamma: float,
    ):

        # Name of the color system
        self.name = system_name

        z_red = 1 - (x_red + y_red)
        z_green = 1 - (x_green + y_green)
        z_blue = 1 - (x_blue + y_blue)

        # Dimensions x Color. rgb -> xyz
        self.M = np.array(
            [
                [x_red, y_red, z_red],
                [x_green, y_green, z_green],
                [x_blue, y_blue, z_blue],
            ],
            dtype=float,
        ).T

        # Finds inverse, xyz -> rgb
        self.M_inv = np.array(
            [
                [
                    self.M[1, 1] * self.M[2, 2] - self.M[1, 2] * self.M[2, 1],
                    self.M[0, 2] * self.M[2, 1] - self.M[0, 1] * self.M[2, 2],
                    self.M[0, 1] * self.M[1, 2] - self.M[0, 2] * self.M[1, 1],
                ],
                [
                    self.M[1, 2] * self.M[2, 0] - self.M[1, 0] * self.M[2, 2],
                    self.M[0, 0] * self.M[2, 2] - self.M[0, 2] * self.M[2, 0],
                    self.M[0, 2] * self.M[1, 0] - self.M[0, 0] * self.M[1, 2],
                ],
                [
                    self.M[1, 0] * self.M[2, 1] - self.M[1, 1] * self.M[2, 0],
                    self.M[0, 1] * self.M[2, 0] - self.M[0, 0] * self.M[2, 1],
                    self.M[0, 0] * self.M[1, 1] - self.M[0, 1] * self.M[1, 0],
                ],
            ],
            dtype=float,
        )

        # White points
        z_white = 1 - (x_white + y_white)
        self.W = np.array([x_white, y_white, z_white], dtype=float)

        rw = (self.M_inv[0, :] @ self.W) / self.W[1]
        gw = (self.M_inv[1, :] @ self.W) / self.W[1]
        bw = (self.M_inv[2, :] @ self.W) / self.W[1]

        self.M_inv[0, :] /= rw
        self.M_inv[1, :] /= gw
        self.M_inv[2, :] /= bw

        # for i in range(3):
        #     self.M_inv[i] /= self.W[i]

        self.gamma = gamma

    def __str__(self):
        msg = "Color System: %s" % self.name
        return msg


def generate_color_systems():
    """Generates the different color systems that using matrix color
    mixing."""

    # For NTSC television
    IlluminantC = (0.3101, 0.3162)
    # For EBU and SMPTE
    IlluminantD65 = (0.3127, 0.3291)
    # CIE equal-energy illuminant
    IlluminantE = (0.33333333, 0.33333333)

    GAMMA_REC709 = 0.0

    # Different color system setups
    NTSCsystem = ColorSystem(
        "NTSC", 0.67, 0.33, 0.21, 0.71, 0.14, 0.08, *IlluminantC, GAMMA_REC709
    )
    EBUsystem = ColorSystem(
        "EBU (PAL/SECAM)",
        0.64,
        0.33,
        0.29,
        0.60,
        0.15,
        0.06,
        *IlluminantD65,
        GAMMA_REC709,
    )
    SMPTEsystem = ColorSystem(
        "SMPTE",
        0.630,
        0.340,
        0.310,
        0.595,
        0.155,
        0.070,
        *IlluminantD65,
        GAMMA_REC709,
    )
    HDTVsystem = ColorSystem(
        "HDTV",
        0.670,
        0.330,
        0.210,
        0.710,
        0.150,
        0.060,
        *IlluminantD65,
        GAMMA_REC709,
    )
    CIEsystem = ColorSystem(
        "CIE",
        0.7355,
        0.2645,
        0.2658,
        0.7243,
        0.1669,
        0.0085,
        *IlluminantE,
        GAMMA_REC709,
    )
    Rec709system = ColorSystem(
        "CIE REC 709",
        0.64,
        0.33,
        0.30,
        0.60,
        0.15,
        0.06,
        *IlluminantD65,
        GAMMA_REC709,
    )
    AdobeRBG1998System = ColorSystem(
        "Adobe RBG 1998",
        0.64,
        0.33,
        0.21,
        0.71,
        0.15,
        0.06,
        *IlluminantD65,
        GAMMA_REC709,
    )
    SRGBSystem = ColorSystem(
        "sRGB", 0.64, 0.33, 0.3, 0.6, 0.15, 0.06, *IlluminantD65, GAMMA_REC709
    )

    return {
        "NTSC": NTSCsystem,
        "EBU": EBUsystem,
        "SMPTE": SMPTEsystem,
        "HDTV": HDTVsystem,
        "CIE": CIEsystem,
        "Rec709": Rec709system,
        "AdobeRBG1998": AdobeRBG1998System,
        "SRGB": SRGBSystem,
    }


COLOR_SYSTEMS = generate_color_systems()


def planc_rad_law(lmbda: float, T: float = 5000) -> float:
    """Planc's Radiation Law.

    Returns the intensity for a wavelength as given by Planc's Radiation Law,

    P(lmbda) dLmbda = c1 lmbda^-1 / (exp(c2 / (T * lmbda)) - 1) dLmbda

    Arguments:
        lmbda {float} -- wavelength in nm.

    Keyword Arguments:
        T {float} -- temperature of spectrum (default: {5000})

    Returns:
        float or np.ndarray -- the intensity P(lmbda).
    """

    # assert np.all(lmbda >= 380.0) and np.all(lmbda <= 780), (
    #     "Wavelength outside visible spectrum: %f" % lmbda)

    # Converts from nm to m
    lmbd = lmbda * 1e-9

    c1 = 3.74183e-16  # W m^2, 2pi*h*c^2
    c2 = 1.4388e-2  # m^2 K, h*c / k

    return c1 / (lmbd ** 5 * (np.exp(c2 / (T * lmbd)) - 1.0))


def _CIE_color_matching(
    T: float = 5000,
    wavelengths: np.ndarray = np.arange(380, 780.1, 5.0),
    _CIE: np.ndarray = CIE,
):
    """Converts wavelengths to xyz."""

    # X, Y, Z = 0, 0, 0

    # Return array setup
    xyz = np.empty((wavelengths.shape[0], 3), dtype=float)

    for i, lmbda in enumerate(wavelengths):

        xyz[i] = planc_rad_law(lmbda, T=T)
        xyz[i] *= _CIE[i]
    #     X += Me * cie[i, 0]
    #     Y += Me * cie[i, 1]
    #     Z += Me * cie[i, 2]

    # XYZ = (X + Y + Z);
    # return np.array([X / XYZ, Y / XYZ, Z / XYZ])

    return xyz / np.sum(xyz)


def spectrum_to_xyz(T=5000):
    """Converts wavelengths to xyz."""

    xyz = _CIE_color_matching(T=T)

    X = xyz[:, 0].sum()
    Y = xyz[:, 1].sum()
    Z = xyz[:, 2].sum()

    # XYZ = X + Y + Z
    # return np.array([X / XYZ, Y / XYZ, Z / XYZ])

    return np.array([X, Y, Z])


def xyz_to_rgb(cs: ColorSystem, xyz: np.ndarray):
    """Converts the xyz representation to RGB."""
    if len(xyz.shape) == 2:
        rgb = np.empty(xyz.shape, dtype=float)
        for i in range(xyz.shape[0]):
            rgb[i] = cs.M_inv @ xyz[i]

        return rgb
    else:
        return cs.M_inv @ xyz


def norm_rgb(r, g, b):
    """Normalizes the RGB representation."""
    greatest = max(r, max(g, b))
    if greatest > 0:
        r /= greatest
        g /= greatest
        b /= greatest
    return np.array([r, g, b])


def constrain_rgb(r, g, b):
    """Constrains the RGB representation, in order to remove negative values."""
    w = -min(0, r, g, b)
    if w > 0:
        r += w
        g += w
        b += w
        return True, np.array([r, g, b])
    return False, np.array([r, g, b])


def _create_expanded_spectrum_input(N_new):
    """Expands the color mixing spectrum.

    Uses the CIE mixing value and interpolates them to a new size defined
    by the input.

    Arguments:
        N_new {int} -- new CIE mixing size.

    Returns:
        np.ndarray, np.ndarray, np.ndarray -- x array, x new array, expanded
            CIE 3xN_new array.
    """

    x = np.linspace(0, 1, CIE.shape[0])
    x_dense = np.linspace(0, 1, N_new)
    CIE_expanded = np.vstack(
        [
            interp1d(x, CIE[:, 0], kind="cubic")(x_dense),
            interp1d(x, CIE[:, 1], kind="cubic")(x_dense),
            interp1d(x, CIE[:, 2], kind="cubic")(x_dense),
        ]
    ).T

    return x, x_dense, CIE_expanded


@nb.njit(cache=True)
def wavelength_to_rgb(
    wavelength: float, gamma: float = 0.8
) -> Tuple[float, float, float]:

    """This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html

    Retrieved from
    http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    """

    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0

    return R, G, B


@nb.njit(cache=True)
def logistic(x: np.ndarray) -> np.ndarray:
    """Simple logistic function."""
    return 1 / (1 + np.exp(-x))


def create_emission_image(
    spectra_rgb_image: np.ndarray,
    x_ticks: np.ndarray,
    wl_labels: List[str],
    element_dict: dict,
    element_watermark: bool,
    output_folder: str,
):
    """Create the emission spectra image."""

    element_name, element_ids, element_short = element_dict.values()

    fig1 = plt.figure(figsize=OUTPUT_SIZE, dpi=OUTPUT_DPI)
    ax1 = fig1.add_axes([0, 0, 1, 1])
    fig1.frameon = False
    ax1.set_frame_on(False)
    ax1.imshow(spectra_rgb_image)

    # Fixes the wavelength labels slightly inside the plot
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(wl_labels, fontsize=5, color="white")
    ax1.tick_params(axis="x", direction="in", pad=-15)
    ax1.set_xticks(ax1.get_xticks()[1:-1])

    if element_watermark:
        ax1.text(
            0.01,
            0.95,
            r"%s, $_{%d}$%s" % (element_name, element_ids, element_short),
            verticalalignment="center",
            horizontalalignment="left",
            transform=ax1.transAxes,
            color="white",
            fontsize=15,
        )

    figpath = output_folder / f"{element_name}_{'%03d'%element_ids}.png"
    fig1.canvas.print_png(figpath)
    tqdm.write(f"Saved {figpath}.")

    plt.close(fig1)


def _setup_visible_spectrum(wavelengths, color_system="CIE", temperature=3000):
    """Helper function for seting up a visible rainbow spectrum."""

    # Expanding
    x, x_dense, CIE_expanded = _create_expanded_spectrum_input(
        wavelengths.shape[0]
    )

    xyz = _CIE_color_matching(
        T=temperature, wavelengths=wavelengths, _CIE=CIE_expanded
    )
    rgb = xyz_to_rgb(COLOR_SYSTEMS[color_system], xyz)

    rgb_array = np.empty(rgb.shape, dtype=float)
    for i in range(rgb.shape[0]):
        is_constrained, _rgb = constrain_rgb(*rgb[i])
        if is_constrained:
            rgb_array[i] = norm_rgb(*_rgb)
        else:
            rgb_array[i] = norm_rgb(*rgb[i])

    return x, x_dense, CIE_expanded, xyz, rgb, rgb_array


def _expand_rgb(rgb_array: np.ndarray, Ny: int) -> np.ndarray:
    """Expands the RBG so it can be fit into an image/full matrix.

    Performs a constant extrapolation in the new axis direction in a new
    matrix [np.newaxis, ...].

    Arguments:
        rgb_array {np.ndarray} -- the image/matrix to expand into full image
            along Ny axis.
        Ny {int} -- the number of pixels in the y direction.

    Returns:
        np.ndarray -- the expanded image.
    """
    Nx, N_channels = rgb_array.shape
    rgb_m = np.empty((Ny, Nx, N_channels), dtype=float)
    for i in range(Nx):
        for j in range(Ny):
            rgb_m[j, i] = rgb_array[i]

    return rgb_m


def test_color_system(cs):
    """Creates the color spectrum mixing.

    Using a ColorSystem input to create the spectrum mixing of it.

    Arguments:
        cs {ColorSystem} -- color system mixing.
    """

    print("Tests color system: %s" % str(cs))

    temperatures = np.asarray(list(range(1000, 10001, 500)))
    bb_radiations = np.empty((temperatures.shape[0], 3), dtype=float)

    for i, t in enumerate(temperatures):
        xyz = spectrum_to_xyz(T=t)
        x, y, z = xyz
        r, g, b = xyz_to_rgb(cs, xyz)
        s = "  %5.0f K      %.4f %.4f %.4f   " % (t, x, y, z)
        is_constrained, _rgb = constrain_rgb(r, g, b)
        if is_constrained:
            r, g, b = norm_rgb(*_rgb)
            s += "%.3f %.3f %.3f (Approximation)" % (r, g, b)
        else:
            r, g, b = norm_rgb(r, g, b)
            s += "%.3f %.3f %.3f" % (r, g, b)
        print(s)
        bb_radiations[i] = np.array([r, g, b])

    plt.figure()
    plt.plot(temperatures, bb_radiations[:, 0], "--", color="r", label="Red")
    plt.plot(temperatures, bb_radiations[:, 1], "--", color="g", label="Green")
    plt.plot(temperatures, bb_radiations[:, 2], "--", color="b", label="Blue")
    plt.xlabel(r"Temperature $[K]$")
    plt.legend()
    plt.show()


def test_create_color_mixing():
    """Creates the color mixing spectrum.

    Performs a extrapolation of the color mixing spectrum and creates a
    figure of it.
    """
    N_wls = 1500

    # Expanding
    x, x_dense, CIE_expanded = _create_expanded_spectrum_input(N_wls)

    print("Shape before interpolation:", CIE.shape)
    print("Shape after interpolation:", CIE_expanded.shape)
    plt.plot(x, CIE[:, 0], color="r")
    plt.plot(x_dense, CIE_expanded[:, 0], "--o", color="r")
    plt.plot(x, CIE[:, 1], color="b")
    plt.plot(x_dense, CIE_expanded[:, 1], "--o", color="b")
    plt.plot(x, CIE[:, 2], color="g")
    plt.plot(x_dense, CIE_expanded[:, 2], "--o", color="g")
    plt.title("CIE color mixing")
    plt.show()


def test_create_visible_spectrum():
    """Creates the visible spectrum.

    Creates the visible spectrum(rainbow) from 380 nm to 780 nm.
    """

    # Parameters
    Ny = 100  # number of y pixels
    N_wls = 1500
    wl_start = 380
    wl_stop = 780
    wavelengths = np.linspace(wl_start, wl_stop, N_wls)

    x, x_dense, CIE_expanded, xyz, rgb, rgb_array = _setup_visible_spectrum(
        wavelengths
    )

    # Mixing plot
    fig0, ax0 = plt.subplots(1, 1)
    ax0.plot(xyz)
    ax0.plot(rgb, "--")
    ax0.legend(["x", "y", "z", "r", "b", "g"])

    # Expands rgb to full image and smooths
    rgb_m = _expand_rgb(rgb_array, Ny)

    # X-direction scaling
    x = np.linspace(0, 1, rgb_array.shape[0])
    smoothing_x = 4 * (-((x - 0.5) ** 2) + 0.25)
    for i in range(rgb_array.shape[0]):
        rgb_m[:, i, :] *= smoothing_x[i]

    # Rainbow spectra plot
    wl_labels = ["%.1f" % i for i in wavelengths[::100]]
    fig1, ax1 = plt.subplots(1, 1)
    ax1.imshow(rgb_m)
    ax1.set_xticks(np.linspace(0, rgb_m.shape[1] - 1, len(wl_labels)))
    ax1.set_xticklabels(wl_labels)
    plt.show()


@click.command()
@click.option(
    "-s",
    "--spectra",
    type=click.Path(exists=True),
    default=Path("spectras2"),
    show_default=True,
    help="The spectra data folder.",
)
@click.option(
    "-o",
    "--output_folder",
    default=Path("generated_emission_spectras"),
    show_default=True,
    type=click.Path(exists=True),
    help="Path to save spectra at.",
)
@click.option(
    "--element_watermark",
    default=False,
    show_default=True,
    is_flag=True,
    type=bool,
    help=(
        "Will add element name and symbol in top left corner of output spectra"
        " image."
    ),
)
@click.option(
    "--spectra_option",
    default="empirical",
    type=click.Choice(SPECTRA_GENERATION_OPTIONS),
    show_default=True,
    help=("Method to use for generating spectra."),
)
def generate_emission_spectra(
    spectra="spectras2",
    output_folder="generated_emission_spectras",
    element_watermark=False,
    spectra_option="empirical",
):
    """Generates emission spectra.

    Generates emission spectra based on observed emission spectra. Attempts to
    moderate the strength of a line by using the intensity.

    Arguments:
        spectra {str} -- folder path to spectra sound files.
        output_folder {str} -- path to save spectra at.
        element_watermark {bool} -- if true, will include the element in the
            output upper left corner.
        spectra_option {str} -- method to use for generating spectra.
    """

    output_folder = Path(output_folder)
    spectra = Path(spectra)

    # Parameters
    Ny = 1080
    Nx = 1980
    wl_start = 380
    wl_stop = 750
    wavelengths = np.linspace(wl_start, wl_stop, Nx)

    if spectra_option == "empirical":
        rgb_array = np.zeros((len(wavelengths), 3), dtype=float)
        for i in range(len(wavelengths)):
            rgb_array[i, :] = wavelength_to_rgb(wavelengths[i], 0.8)
    else:
        rgb_array = _setup_visible_spectrum(
            wavelengths,
            color_system=spectra_option,
            temperature=3000,
        )[-1]

    # Sets up image arrays
    rgb_m_smoothed = _expand_rgb(rgb_array, Ny)
    rgb_m = np.copy(rgb_m_smoothed)

    # X-direction scaling
    x = np.linspace(0, 1, Nx)
    smoothing_x = 4 * (-((x - 0.5) ** 2) + 0.25)
    for i in range(Nx):
        rgb_m_smoothed[:, i, :] *= smoothing_x[i]

    # Y-direction scaling
    smoothing_y = np.sin(np.linspace(0, 1, Ny) * np.pi)
    for i in range(Ny):
        rgb_m_smoothed[i, :, :] *= smoothing_y[i]

    # Rainbow spectra plot
    wl_labels = ["%.0fnm" % i for i in wavelengths[::100]]
    x_ticks = np.linspace(0, Nx - 1, len(wl_labels)).astype(int)

    # Sets up interpolator for wavelength to index
    wl_to_index = interp1d(wavelengths, np.arange(len(wavelengths)))

    check_folder(output_folder)

    for element_dict in tqdm(retrieve_viable_elements(spectra)):

        # Sets up spectra data path
        element_fpath = spectra / (element_dict["short"] + ".dat")

        # Returns periodic table notation
        element = element_search(element_dict["short"])
        if not element:
            tqdm.write(f"Element not found: {element_fpath}.")

        Sound = Elemental(
            element, local_file=str(element_fpath), verbose=False
        )

        if not Sound.has_spectra:
            tqdm.write(f"No spectra exist for {element}.")
            continue

        # Sound.remove_beat(1e-2)
        spectra_wl = Sound.spectra[:, 0]
        spectra_intensity = Sound.spectra[:, 1]

        # Filters out non-visible lines
        spectra_wl = spectra_wl[wl_start < spectra_wl]
        spectra_wl = spectra_wl[spectra_wl < wl_stop]

        # Interpolates to get spectra indexes. We use the interpolator to
        # retrieve the spectra nearest indexes in floats.
        spectra_ids = wl_to_index(spectra_wl)

        # Sets up interpolation points
        spectra_ids_lower = list(map(int, np.floor(spectra_ids)))
        spectra_ids_upper = list(map(int, np.ceil(spectra_ids)))

        # Removes lines which is too close to each other to be seen.
        ids_to_pop = []
        spectra_wl_updated = []
        spectra_intensity_updated = []
        for i in range(spectra_wl.shape[0]):
            if spectra_ids_lower[i] == spectra_ids_lower[i - 1]:
                ids_to_pop.append(i)
            else:
                spectra_wl_updated.append(spectra_wl[i])
                spectra_intensity_updated.append(spectra_intensity[i])

        # Removes all weights for wl's no longer in use
        for i in reversed(ids_to_pop):
            del spectra_ids_lower[i]
            del spectra_ids_upper[i]

        # Updates the spectra wavelengths for the element
        spectra_wl = np.array(spectra_wl_updated)
        spectra_intensity = np.array(spectra_intensity_updated)

        if len(spectra_wl) == 0:
            tqdm.write(f"No spectra found for {element}. Saving empty image.")

            create_emission_image(
                rgb_m_smoothed * 0.25,
                x_ticks,
                wl_labels,
                element_dict,
                element_watermark,
                output_folder,
            )

            continue

        # Softly normalizes spectra intensity
        spectra_intensity = logistic(spectra_intensity)

        # Broadcasting the intensity
        spectra_intensity_weights = np.empty(
            (rgb_m.shape[0], len(spectra_ids_lower), rgb_m.shape[2]),
            dtype=float,
        )
        for i in range(len(spectra_ids_lower)):
            spectra_intensity_weights[:, i, :] = spectra_intensity[i]

        # Scales the background
        spectra_rgb_image = rgb_m_smoothed * 0.25

        # Places the spectra RBG in the output image.
        spectra_rgb_image[:, spectra_ids_lower, :] = np.maximum(
            rgb_m[:, spectra_ids_lower, :] * spectra_intensity_weights,
            spectra_rgb_image[:, spectra_ids_lower, :],
        )
        spectra_rgb_image[:, spectra_ids_upper, :] = np.maximum(
            rgb_m[:, spectra_ids_upper, :] * spectra_intensity_weights,
            spectra_rgb_image[:, spectra_ids_upper, :],
        )

        create_emission_image(
            spectra_rgb_image,
            x_ticks,
            wl_labels,
            element_dict,
            element_watermark,
            output_folder,
        )


if __name__ == "__main__":
    generate_emission_spectra()

    # test_color_system(COLOR_SYSTEMS["SMPTE"])
    # test_create_visible_spectrum()
    # test_create_color_mixing()
