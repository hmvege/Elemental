import multiprocessing as mp
from pathlib import Path
from typing import Tuple

import numba as nb  # For converting the main function to machine code
import numpy as np
from scipy.io.wavfile import write as wavfile_write

from .utils import element_downloader, check_folder


@nb.njit("double[:](double[:], double[:], double[:], int64)", cache=True)
def create_tones(
    t: np.ndarray, spectra: np.ndarray, envelope: np.ndarray, Hz: int
) -> np.ndarray:
    """Creates the sound array.

    Creates sound waves from the wavelengths of the different provided spectra.

    Decorators:
        nb.njit

    Arguments:
        t {np.ndarray} -- time array in seconds. Length corresponds sampling
            rate times length in seconds.
        spectra {np.ndarray} -- the different wavelengths to create sine
            waves from.
        envelope {np.ndarray} -- envelope to limit extremes of the output
            sound.
        Hz {int} -- hertz of the sound array.

    Returns:
        np.ndarray -- sound array
    """
    t_modified = (2 * np.pi * Hz) * t

    tone_matrix = np.outer(spectra, t_modified)
    tone = np.zeros(tone_matrix.shape[1])
    for i in range(tone_matrix.shape[0]):
        tone += np.sin(tone_matrix[i])

    return envelope * tone


def _tone_parallel(
    _input_values: Tuple[np.ndarray, np.ndarray, np.ndarray, int]
) -> np.ndarray:
    """Helper function for running create_tones in parallel.

    Arguments:
        _input_values {Tuple[np.ndarray, np.ndarray, np.ndarray, int]} -- input
            values to be passed to create_tones.

    Returns:
        np.ndarray -- sound array.
    """
    t, spectra, envelope, hz = _input_values
    return create_tones(t, spectra, envelope, hz)


@nb.njit(cache=True)
def _calculate_rydeberg_spectra(_n1: int, _n2: int, _R: float) -> float:
    """Calculates the spectra from the Rydeberg formula.

    Decorators:
        nb.njit

    Arguments:
        _n1 {int} -- First principal quantum number.
        _n2 {int} -- Second principal quantum number.
        _R {float} -- Rydeberg constant.

    Returns:
        float -- Calculated wavelength.
    """
    return _R * (1.0 / _n1 ** 2 - 1.0 / _n2 ** 2)


@nb.njit(cache=True)
def rydeberg(n1: int, n2_max: int = 10) -> np.ndarray:
    """Retrieves the Rydeberg hydrogen spectra

    The Rydeberg formula for finding the spectra wavelength

    Arguments:
        n1 {int} -- the principal quantum number of the upper energy level

    Keyword Arguments:
        n2_max {int} -- principal quantum number upper limit of the
            lower energy level for the atomic electron transition.
            (default: {20})

    Returns:
        number -- atomic spectras, wavelength inverse
    """

    # Generates wavelength spectra
    R = 1.09677583e7  # [m−1]

    n2_list = np.linspace(n1 + 1, n1 + n2_max, n2_max)

    return _calculate_rydeberg_spectra(n1, n2_list, R)


class _Sound:
    def __init__(self, parallel: bool = False, num_processors: int = 4):
        # Checking if we are to run in parallel
        if parallel:
            self.parallel = True
            self.num_processors = num_processors
        else:
            self.parallel = False

    def remove_beat(self, eps: float = 1e-2):
        """
        Removes beat frequencies that cause destructive interference. Will
        incrementally remove spectra lines till the total number is
        below 1000 due to save time.
        """

        # Warning: changing this variable beyond 1200 may cause computer
        # slowdown.
        max_spectras = 1000

        spectra = self.spectra
        temp_spectra = spectra

        while len(spectra) > max_spectras:
            temp_spectra = [
                spectra[i]
                for i in range(0, len(spectra) - 1)
                if abs(spectra[i] - spectra[i + 1]) > eps
            ]
            spectra = temp_spectra
            eps *= 1.1

        self.spectra = spectra

    def _envelope(
        self, N: int, sampling_rate: int, length: float, amplitude: float
    ) -> np.array:
        """
        Envelope for 'smoothing' out the beginning and end of the signal.
        Also sets the sound amplitude.
        """
        envelope_time = 0.2  # seconds
        envelope_N = int(sampling_rate * envelope_time)
        envelope_array = np.ones(N)
        envelope = self._envelope_function(envelope_N)
        envelope_array[:envelope_N] *= envelope
        envelope_array[(N - envelope_N):] *= envelope[::-1]

        return amplitude * envelope_array

    @staticmethod
    @nb.njit(cache=True)
    def _envelope_function(envelope_N: int) -> np.ndarray:
        """
        Cosine envelope
        """
        x = np.linspace(0, np.pi, envelope_N)
        return (1 - np.cos(x)) / 2.0

    def create_sound(
        self,
        length: int = 10,
        hz: int = 440,
        amplitude: float = 0.01,
        sampling_rate: int = 44100,
        convertion_factor: int = 100,
        wavelength_cutoff: float = 2.5e-1,
        output_folder: Path = Path("sounds"),
    ):
        """
        Function for creating soundfile from experimental element spectra.
        """
        if amplitude >= 0.1:
            raise ValueError(f"Change amplitude! {amplitude} is way to high!")

        if length < 0.5:
            raise ValueError(
                "Cannot create a sample of length less than 0.5 "
                f"seconds({length} is provided)."
            )

        # Creating time vector and getting length of datapoints
        N = int(length * sampling_rate)
        t = np.linspace(0, length, N)

        # Setting up spectra and converting to audible spectrum
        spectra = (1.0 / np.asarray(self.spectra)) * convertion_factor
        if wavelength_cutoff:
            spectra = np.asarray([i for i in spectra if i > wavelength_cutoff])

        # Creating envelope
        envelope = self._envelope(N, sampling_rate, length, amplitude)

        if self.parallel:
            num_processors = self.num_processors
            # t_modified = (2 * np.pi * Hz) * t
            input_values = zip(
                np.split(t, num_processors),
                [spectra for i in range(num_processors)],
                np.split(envelope, num_processors),
                [hz for i in range(num_processors)],
            )

            with mp.Pool(processes=num_processors) as pool:
                output = pool.map_async(_tone_parallel, input_values).get()
                tone = np.concatenate(output)

        else:
            tone = create_tones(t, spectra, envelope, hz)

        check_folder(output_folder)

        filepath = output_folder / f"{self.filename}_{int(length)}sec.wav"

        wavfile_write(filepath, sampling_rate, tone)
        print(f"{filepath} written.")

        self.tone, self.length = tone, length


class Elemental(_Sound):
    # For storing if we find spectras.
    has_spectra = True

    def __init__(
        self, element, local_file=None, parallel=False, num_processors=4
    ):
        """
        Initialization of element class.
        """
        super().__init__(parallel=parallel, num_processors=num_processors)

        self.filename = element

        # Downloading or retrieving element spectra data
        if not local_file:
            self.spectra = element_downloader(element)
            if not self.check_spectras(element):
                self.has_spectra = False
                return
            print("Spectra retrieved from nist.org.")
        else:
            self.spectra = element_downloader(element, local_file)
            if not self.check_spectras(element):
                self.has_spectra = False
                return
            print(f"Spectra retrieved from local file {local_file}.")

    def check_spectras(self, element: str) -> bool:
        """
        Helper function for checking if we have retrieved any available
        spectra.
        """
        if len(self.spectra) == 0:
            print(f"No atomic spectra available for {element}")
            return False
        return True


class Rydeberg(_Sound):
    def __init__(self, n1_series, n2_max, parallel=False, num_processors=4):

        super().__init__(parallel=parallel, num_processors=num_processors)

        self.filename = f"H_Rydeberg_n{n1_series}"

        assert (
            n1_series >= 1
        ), "Principal quantum number n1 must be larger than 1."

        assert n1_series < n2_max, "n1 must be less than n2."

        if n2_max > 15:
            print(f"Warning: {n2_max} > 15 will give poor results!")

        self.spectra = [
            rydeberg(i, n2_max=n2_max) ** (-1) * 10 ** 9
            for i in range(1, n1_series + 1)
        ]
        self.spectra = np.array(self.spectra).flatten()
