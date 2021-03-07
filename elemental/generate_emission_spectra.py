import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy.interpolate import interp1d
from lib.utils import element_search
import ElementSound as es


CIE = np.array([
    [0.0014, 0.0000, 0.0065], [0.0022, 0.0001, 0.0105], [0.0042, 0.0001, 0.0201],
    [0.0076, 0.0002, 0.0362], [0.0143, 0.0004, 0.0679], [0.0232, 0.0006, 0.1102],
    [0.0435, 0.0012, 0.2074], [0.0776, 0.0022, 0.3713], [0.1344, 0.0040, 0.6456],
    [0.2148, 0.0073, 1.0391], [0.2839, 0.0116, 1.3856], [0.3285, 0.0168, 1.6230],
    [0.3483, 0.0230, 1.7471], [0.3481, 0.0298, 1.7826], [0.3362, 0.0380, 1.7721],
    [0.3187, 0.0480, 1.7441], [0.2908, 0.0600, 1.6692], [0.2511, 0.0739, 1.5281],
    [0.1954, 0.0910, 1.2876], [0.1421, 0.1126, 1.0419], [0.0956, 0.1390, 0.8130],
    [0.0580, 0.1693, 0.6162], [0.0320, 0.2080, 0.4652], [0.0147, 0.2586, 0.3533],
    [0.0049, 0.3230, 0.2720], [0.0024, 0.4073, 0.2123], [0.0093, 0.5030, 0.1582],
    [0.0291, 0.6082, 0.1117], [0.0633, 0.7100, 0.0782], [0.1096, 0.7932, 0.0573],
    [0.1655, 0.8620, 0.0422], [0.2257, 0.9149, 0.0298], [0.2904, 0.9540, 0.0203],
    [0.3597, 0.9803, 0.0134], [0.4334, 0.9950, 0.0087], [0.5121, 1.0000, 0.0057],
    [0.5945, 0.9950, 0.0039], [0.6784, 0.9786, 0.0027], [0.7621, 0.9520, 0.0021],
    [0.8425, 0.9154, 0.0018], [0.9163, 0.8700, 0.0017], [0.9786, 0.8163, 0.0014],
    [1.0263, 0.7570, 0.0011], [1.0567, 0.6949, 0.0010], [1.0622, 0.6310, 0.0008],
    [1.0456, 0.5668, 0.0006], [1.0026, 0.5030, 0.0003], [0.9384, 0.4412, 0.0002],
    [0.8544, 0.3810, 0.0002], [0.7514, 0.3210, 0.0001], [0.6424, 0.2650, 0.0000],
    [0.5419, 0.2170, 0.0000], [0.4479, 0.1750, 0.0000], [0.3608, 0.1382, 0.0000],
    [0.2835, 0.1070, 0.0000], [0.2187, 0.0816, 0.0000], [0.1649, 0.0610, 0.0000],
    [0.1212, 0.0446, 0.0000], [0.0874, 0.0320, 0.0000], [0.0636, 0.0232, 0.0000],
    [0.0468, 0.0170, 0.0000], [0.0329, 0.0119, 0.0000], [0.0227, 0.0082, 0.0000],
    [0.0158, 0.0057, 0.0000], [0.0114, 0.0041, 0.0000], [0.0081, 0.0029, 0.0000],
    [0.0058, 0.0021, 0.0000], [0.0041, 0.0015, 0.0000], [0.0029, 0.0010, 0.0000],
    [0.0020, 0.0007, 0.0000], [0.0014, 0.0005, 0.0000], [0.0010, 0.0004, 0.0000],
    [0.0007, 0.0002, 0.0000], [0.0005, 0.0002, 0.0000], [0.0003, 0.0001, 0.0000],
    [0.0002, 0.0001, 0.0000], [0.0002, 0.0001, 0.0000], [0.0001, 0.0000, 0.0000],
    [0.0001, 0.0000, 0.0000], [0.0001, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000]
])


class ColorSystem:
    def __init__(self, system_name,
                 x_red, y_red, x_green, y_green, x_blue, y_blue,
                 x_white, y_white, gamma):

        # Name of the color system
        self.name = system_name

        z_red = 1 - (x_red + y_red)
        z_green = 1 - (x_green + y_green)
        z_blue = 1 - (x_blue + y_blue)

        # Dimensions x Color. rgb -> xyz
        self.M = np.array([
            [x_red,     y_red,      z_red],
            [x_green,   y_green,    z_green],
            [x_blue,    y_blue,     z_blue]],
            dtype=float).T

        # Finds inverse, xyz -> rgb
        # self.M_inv = np.linalg.inv(self.M)

        self.M_inv = np.array([
            [self.M[1, 1]*self.M[2, 2] - self.M[1, 2]*self.M[2, 1],
             self.M[0, 2]*self.M[2, 1] - self.M[0, 1]*self.M[2, 2],
             self.M[0, 1]*self.M[1, 2] - self.M[0, 2]*self.M[1, 1]],
            [self.M[1, 2]*self.M[2, 0] - self.M[1, 0]*self.M[2, 2],
             self.M[0, 0]*self.M[2, 2] - self.M[0, 2]*self.M[2, 0],
             self.M[0, 2]*self.M[1, 0] - self.M[0, 0]*self.M[1, 2]],
            [self.M[1, 0]*self.M[2, 1] - self.M[1, 1]*self.M[2, 0],
             self.M[0, 1]*self.M[2, 0] - self.M[0, 0]*self.M[2, 1],
             self.M[0, 0]*self.M[1, 1] - self.M[0, 1]*self.M[1, 0]],
        ], dtype=float)

        # print(self.M_inv)

        # White points
        z_white = 1 - (x_white + y_white)
        self.W = np.array([x_white, y_white, z_white], dtype=float)

        rw = (self.M_inv[0, :] @ self.W) / self.W[1]
        gw = (self.M_inv[1, :] @ self.W) / self.W[1]
        bw = (self.M_inv[2, :] @ self.W) / self.W[1]

        self.M_inv[0, :] /= rw
        self.M_inv[1, :] /= gw
        self.M_inv[2, :] /= bw

        # print(self.M)
        # print(self.M_inv)

        # for i in range(3):
        #     self.M_inv[i] /= self.W[i]

        self.gamma = gamma

    def __str__(self):
        msg = "Color System: %s" % self.name
        return msg


def generate_color_systems():

    # For NTSC television
    IlluminantC = (0.3101, 0.3162)
    # For EBU and SMPTE
    IlluminantD65 = (0.3127, 0.3291)
    # CIE equal-energy illuminant
    IlluminantE = (0.33333333, 0.33333333)

    GAMMA_REC709 = 0.0

    # Different color system setups
    NTSCsystem = ColorSystem("NTSC", 0.67, 0.33, 0.21, 0.71,
                             0.14, 0.08, *IlluminantC, GAMMA_REC709)
    EBUsystem = ColorSystem("EBU (PAL/SECAM)", 0.64, 0.33,
                            0.29, 0.60, 0.15, 0.06, *IlluminantD65,
                            GAMMA_REC709)
    SMPTEsystem = ColorSystem("SMPTE", 0.630, 0.340, 0.310,
                              0.595, 0.155, 0.070, *IlluminantD65,
                              GAMMA_REC709)
    HDTVsystem = ColorSystem("HDTV", 0.670, 0.330, 0.210,
                             0.710, 0.150, 0.060, *IlluminantD65,
                             GAMMA_REC709)
    CIEsystem = ColorSystem("CIE", 0.7355, 0.2645, 0.2658,
                            0.7243, 0.1669, 0.0085, *IlluminantE,
                            GAMMA_REC709)
    Rec709system = ColorSystem("CIE REC 709", 0.64, 0.33,
                               0.30, 0.60, 0.15, 0.06, *IlluminantD65,
                               GAMMA_REC709)
    AdobeRBG1998System = ColorSystem("Adobe RBG 1998", 0.64, 0.33,
                                     0.21, 0.71, 0.15, 0.06, *IlluminantD65,
                                     GAMMA_REC709)
    SRGBSystem = ColorSystem("sRGB", 0.64, 0.33, 0.3, 0.6, 0.15, 0.06,
                             *IlluminantD65, GAMMA_REC709)

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


def planc_rad_law(lmbda: float, T: float = 5000):
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

    return c1 / (lmbd**5 * (np.exp(c2 / (T * lmbd)) - 1.0))


def CIE_color_matching(T=5000, wavelengths=np.arange(380, 780.1, 5.0),
                       _CIE=CIE):

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

    xyz = CIE_color_matching(T=T)

    X = xyz[:, 0].sum()
    Y = xyz[:, 1].sum()
    Z = xyz[:, 2].sum()

    XYZ = X + Y + Z

    # return np.array([X / XYZ, Y / XYZ, Z / XYZ])
    return np.array([X, Y, Z])


def xyz_to_rgb(cs: ColorSystem, xyz: np.ndarray):
    # rgb to xyz mixing matrix
    M = cs.M

    # White points
    W = cs.W

    if len(xyz.shape) == 2:
        rgb = np.empty(xyz.shape, dtype=float)
        for i in range(xyz.shape[0]):
            rgb[i] = cs.M_inv @ xyz[i]

        return rgb
    else:
        return cs.M_inv @ xyz


def norm_rgb(r, g, b):
    greatest = max(r, max(g, b))
    if greatest > 0:
        r /= greatest
        g /= greatest
        b /= greatest
    return np.array([r, g, b])


def constrain_rgb(r, g, b):
    w = - min(0, r, g, b)
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
        np.ndarray, np.ndarray, np.ndarray -- x array, x new array, expaned CIE 3xN_new array
    """

    x = np.linspace(0, 1, CIE.shape[0])
    x_dense = np.linspace(0, 1, N_new)
    CIE_expanded = np.vstack(
        [interp1d(x, CIE[:, 0], kind="cubic")(x_dense),
         interp1d(x, CIE[:, 1], kind="cubic")(x_dense),
         interp1d(x, CIE[:, 2], kind="cubic")(x_dense)]).T

    return x, x_dense, CIE_expanded


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
    wl_start = 380
    wl_stop = 780

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


def _setup_visible_spectrum(wavelengths, color_system="SRGB", temperature=3000):
    """Helper function for seting up a visible rainbow spectrum."""

    # Expanding
    x, x_dense, CIE_expanded = _create_expanded_spectrum_input(wavelengths.shape[0])

    xyz = CIE_color_matching(T=temperature, wavelengths=wavelengths,
                             _CIE=CIE_expanded)
    rgb = xyz_to_rgb(COLOR_SYSTEMS[color_system], xyz)

    rgb_array = np.empty(rgb.shape, dtype=float)
    for i in range(rgb.shape[0]):
        is_constrained, _rgb = constrain_rgb(*rgb[i])
        if is_constrained:
            rgb_array[i] = norm_rgb(*_rgb)
        else:
            rgb_array[i] = norm_rgb(*rgb[i])

    return x, x_dense, CIE_expanded, xyz, rgb, rgb_array


def _expand_rgb(rgb_array, smoothing=None, N=100):
    """Expands the RBG so it can be fit into an image."""
    x = np.linspace(0, 1, rgb_array.shape[0])

    # Expands the RGB value to cover the full matrix.
    rgb_m = np.empty((N, *rgb_array.shape), dtype=float)
    for i in range(rgb_array.shape[0]):
        for j in range(rgb_m.shape[0]):
            if not isinstance(smoothing, type(None)):
                rgb_m[j, i] = rgb_array[i] * smoothing[i]
            else:
                rgb_m[j, i] = rgb_array[i]

    return rgb_m


def test_create_visible_spectrum():
    """Creates the visible spectrum.

    Creates the visible spectrum(rainbow) from 380 nm to 780 nm.
    """

    # Parameters
    N = 100 # number of y pixels
    N_wls = 1500
    wl_start = 380
    wl_stop = 780
    wavelengths = np.linspace(wl_start, wl_stop, N_wls)

    x, x_dense, CIE_expanded, xyz, rgb, rgb_array = \
        _setup_visible_spectrum(wavelengths)

    # Mixing plot
    fig0, ax0 = plt.subplots(1, 1)
    ax0.plot(xyz)
    ax0.plot(rgb, "--")
    ax0.legend(["x", "y", "z", "r", "b", "g"])

    # Expands rgb to full image and smooths
    x = np.linspace(0, 1, rgb_array.shape[0])
    smoothing = 4*(-(x-0.5)**2 + 0.25)
    rgb_m = _expand_rgb(rgb_array, smoothing, N)

    # Rainbow spectra plot
    wl_labels = ["%.1f" % i for i in wavelengths[::100]]
    fig1, ax1 = plt.subplots(1, 1)
    ax1.imshow(rgb_m*0.75)
    ax1.set_xticks(np.linspace(0, rgb_m.shape[1] - 1, len(wl_labels)),
                   wl_labels)

    plt.show()


def generateEmissionSpectra(folder_path, output_folder="emission_spectras"):
    """Generates emission spectra.

    Generates emission spectra based on observed emission spectra. Attempts to moderate the strength of a line by using the intensity.

    Arguments:
        folder_path {str} -- folder path to spectras.
    """

    # Parameters
    N = 100
    N_wls = 1500
    wl_start = 380
    wl_stop = 780
    wavelengths = np.linspace(wl_start, wl_stop, N_wls)

    x, x_dense, CIE_expanded, xyz, rgb, rgb_array = \
        _setup_visible_spectrum(wavelengths)

    x = np.linspace(0, 1, rgb_array.shape[0])
    smoothing = 4*(-(x-0.5)**2 + 0.25)
    rgb_m = _expand_rgb(rgb_array, smoothing=smoothing, N=N)

    fig0, ax0 = plt.subplots(1, 1)
    ax0.plot(rgb_array[:,0])
    ax0.plot(rgb_array[:,1])
    ax0.plot(rgb_array[:,2])
    ax0.set_title("Wavelength to RGB")
    plt.show()

    exit(1)

    # Rainbow spectra plot
    wl_labels = ["%.1f" % i for i in wavelengths[::100]]

    wl_to_index = interp1d(wavelengths, np.arange(len(wavelengths)))

    if not os.path.isdir(output_folder):
        print("> mkdir %s" % output_folder)
        os.mkdir(output_folder)

    for element_file in os.listdir(folder_path):

        if element_file.startswith("."):
            continue

        element = element_search(element_file.split(".")[0])
        assert element != False, "element not found: %s" % element
        fpath = os.path.join(folder_path, element_file)

        # TEMP
        if element != "H":
            continue

        Sound = es.ElementSound(element,
                                local_file=fpath,
                                filename=element,
                                parallel=True,
                                num_processors=8)

        if not Sound.has_spectra:
            print("No spectras exist for %s" % element)
            continue

        Sound.remove_beat(1e-2)
        # Sound.create_sound(length=10, Hz=440, amplitude=0.01,
        #                    sampling_rate=44100, convertion_factor=100,
        #                    wavelength_cutoff=2.5e-1, output_folder=output_folder)
        # print("Created sound for %s at %s" % (element, output_folder))

        spectra_wl, spectra_int = Sound.spectra[:,0], Sound.spectra[:,1]

        # Filters out non-visible lines
        spectra_wl = spectra_wl[wl_start<spectra_wl]
        spectra_wl = spectra_wl[spectra_wl<wl_stop]

        # Interpolates to get spectra indexes. We use the interpolator to
        # retrieve the spectras nearest indexes in floats.
        spectra_ids = wl_to_index(spectra_wl)
        print("spectra_wl", spectra_wl)
        print("spectra_ids", spectra_ids)
        print(spectra_ids.shape)
        exit(1)
        print(np.floor(spectra_ids[0]), np.ceil(spectra_ids[0]))

        # Sets up interpolation points
        weights_ids0 = list(map(int, np.floor(spectra_ids)))
        weights_ids1 = list(map(int, np.ceil(spectra_ids)))

        # Removes lines which is too close to each other to be seen.
        ids_to_pop = []
        spectra_wl_updated = []
        for i in range(spectra_wl.shape[0]):
            if weights_ids0[i] == weights_ids0[i-1]:
                ids_to_pop.append(i)
            else:
                spectra_wl_updated.append(spectra_wl[i])

        # Removes all weights for wl's no longer in use
        for i in reversed(ids_to_pop):
            del weights_ids0[i]
            del weights_ids1[i]

        # Updates the spectra wavelengths for the element
        spectra_wl = np.array(spectra_wl_updated)

        # Sets up the weights to use for spectra_wl
        weights = (spectra_ids - np.floor(spectra_ids)) / \
            (np.ceil(spectra_ids) - np.floor(spectra_ids))

        rbg_spectra = rgb_m * 0.5
        # rbg_spectra = np.zeros_like(rbg_spectra) # Temp, setting spectra to be dark

        print(spectra_wl.shape)
        exit("Jeppssss")

        res = _setup_visible_spectrum(spectra_wl)
        print("OK here??")
        print(res[-1])
        exit(1)
        ids = []
        for i_wl in range(res[-1].shape[0]):

            f, ax = plt.subplots(1, 1)
            for j in range(res[-1].shape[1]):
                # print(np.abs(rgb_array[:,j]-res[-1][i_wl, j]))
                i = np.where(np.abs(rgb_array[:,j]-res[-1][i_wl, j]))

            #     print(i)
                ax.plot(np.abs(rgb_array[:,j]-res[-1][i_wl, j]))
            plt.show()


            exit(1)

        print(res[-1].shape)
        print(_expand_rgb(res[-1]).shape)
        print(np.rollaxis(_expand_rgb(res[-1]),1).shape)

        for i, j, w, wl_color in zip(weights_ids0, weights_ids1, weights,
                                     np.rollaxis(_expand_rgb(res[-1]), 1)):
            rbg_spectra[:,i,:] = wl_color * w
            rbg_spectra[:,j,:] = wl_color * (1 - w)

        exit(1)


        # TODO: det som må interpoleres er at en må finne der res[-1] passer inn i rgb_array

        fig1, ax1 = plt.subplots(1, 1)
        ax1.imshow(rbg_spectra)
        ax1.set_xticks(np.linspace(0, rgb_m.shape[1] - 1, len(wl_labels)))
        ax1.set_xticklabels(wl_labels)

        # res = _setup_visible_spectrum(spectra_wl)
        # rgb_spectra = _expand_rgb(res[-1])

        # ax1.imshow(rgb_spectra)

        plt.show()

        exit("Exits after generating a single sound.")



# TODO: load spectras
# TODO: generate spectras


def main():
    """Runs basic tests."""

    # test_color_system(COLOR_SYSTEMS["SMPTE"])
    # test_create_visible_spectrum()
    # test_create_color_mixing()

    generateEmissionSpectra("spectras2")


if __name__ == '__main__':
    main()
