import os
import sys
import numpy as np
import multiprocessing as mp
import argparse
import numba as nb  # For converting the main function to machine code
from scipy.io.wavfile import write as wavfile_write

from .utils import element_downloader, print_all_elements, element_search

import time  # for benchmarking purposes

"""
Program for converting atomic spectra to sound.
"""


@nb.njit(cache=True)
def create_tones(t, spectra, envelope, Hz):
    """
    External function to use when parallelizing.
    Returns tone spectra as a t-length array.
    Fully vectorized.
    """
    t_modified = 2*np.pi*Hz*t

    tone_matrix = np.outer(spectra, t_modified)
    tone = np.zeros(tone_matrix.shape[1])
    for i in range(tone_matrix.shape[0]):
        tone += np.sin(tone_matrix[i])
    return envelope*tone


@nb.njit(cache=True)
def _calculate_rydeberg_spectra(_n1, _n2, _R):
    return _R*(1.0/_n1**2 - 1.0/_n2**2)


def rydeberg(n1: int, n2_max: int = 10):
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

    n2_list = np.linspace(n1+1, n1+n2_max, n2_max)
    spectra = np.empty(n2_list.shape[0])

    for i, n2 in enumerate(n2_list):
        spectra[i] = _calculate_rydeberg_spectra(n1, n2, R)

    return spectra


class _Sound:
    def __init__(self, parallel=False, num_processors=4):
        # Checking if we are to run in parallel
        if parallel:
            self.parallel = True
            self.num_processors = num_processors
        else:
            self.parallel = False

    def remove_beat(self, eps=1e-2):
        """
        Removes beat frequencies that cause a beat destructively. Will
        incrementally remove spectra lines till the total number is
        below 1000 due to save time.
        """

        # Warning: changing this variable beyond 1200 may cause computer
        # slowdown.
        max_spectras = 1000

        spectra = self.spectra
        temp_spectra = spectra

        while len(spectra) > max_spectras:
            temp_spectra = [spectra[i] for i in range(
                0, len(spectra)-1) if abs(spectra[i] - spectra[i+1]) > eps]
            spectra = temp_spectra
            eps *= 1.1

        self.spectra = spectra

    def _envelope(self, N, sampling_rate, length, amplitude):
        """
        Envelope for 'smoothing' out the beginning and end of the signal.
        Also sets the amplitude
        """
        envelope_time = 0.2  # seconds
        envelope_N = int(sampling_rate*envelope_time)
        envelope_array = np.ones(N)
        envelope = self._envelope_function(envelope_N)
        envelope_array[:envelope_N] *= envelope
        envelope_array[(N - envelope_N):] *= envelope[::-1]

        return amplitude * envelope_array

    @staticmethod
    @nb.njit(cache=True)
    def _envelope_function(envelope_N):
        """
        Cosine envelope
        """
        x = np.linspace(0, np.pi, envelope_N)
        return (1 - np.cos(x))/2.

    def create_sound(self, length=10, Hz=440, amplitude=0.01,
                     sampling_rate=44100, convertion_factor=100,
                     wavelength_cutoff=2.5e-1, output_folder="sounds"):
        """
        Function for creating soundfile from experimental element spectra.
        """
        if amplitude >= 0.1:
            sys.exit("Change amplitude! %g is way to high!" % amplitude)

        if length < 0.5:
            sys.exit("Cannot create a sample of length less than 0.5 seconds.")

        # Creating time vector and getting length of datapoints
        N = int(length*sampling_rate)
        t = np.linspace(0, length, N)

        # Setting up spectra and converting to audible spectrum
        spectra = (1./np.asarray(self.spectra))*convertion_factor
        if wavelength_cutoff:
            spectra = np.asarray([i for i in spectra if i > wavelength_cutoff])

        # Creating envelope
        envelope = self._envelope(N, sampling_rate, length, amplitude)

        if self.parallel:
            num_processors = self.num_processors
            input_values = zip(np.split(t, num_processors),
                               [spectra for i in range(num_processors)],
                               np.split(envelope, num_processors),
                               [Hz for i in range(num_processors)])
            pool = mp.Pool(processes=num_processors)
            results = pool.starmap(create_tones, input_values)
            tone = np.concatenate(results)
        else:
            tone = create_tones(t, spectra, envelope, Hz)

        # Writing to file
        if not os.path.isdir(output_folder):
            print('Creating output folder %s' % output_folder)
            os.mkdir(output_folder)

        filename = os.path.join(
            output_folder, '%s_%dsec.wav' % (self.filename, int(length)))

        wavfile_write(filename, sampling_rate, tone)
        print('%s written.' % filename)

        self.tone, self.length = tone, length


class Elemental(_Sound):
    # For storing if we find spectras.
    has_spectra = True

    def __init__(self, element, local_file=None, filename=None,
                 parallel=False, num_processors=4):
        """
        Initialization of element class.
        """
        super().__init__(parallel=parallel, num_processors=num_processors)

        # Creating filename
        if not filename:
            self.filename = "%s" % element
        else:
            self.filename = filename

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
            print("Spectra retrieved from local file %s." % local_file)

    def check_spectras(self, element):
        """
        Helper function for checking if we have retrieved any available
        spectras.
        """
        if len(self.spectra) == 0:
            print('No atomic spectras available for %s' % element)
            return False
        return True


class Rydeberg(_Sound):
    def __init__(self, n1_series, n2_max, filename=None,
                 parallel=False, num_processors=4):

        super().__init__(parallel=parallel, num_processors=num_processors)

        # Creating filename
        if not filename:
            self.filename = "H_Rydeberg_n%d" % n1_series
        else:
            self.filename = filename

        assert n1_series >= 1, (
            "Principal quantm number n1 must be larger than 1.")

        assert n1_series < n2_max, "n1 must be less than n2."

        if n2_max > 15:
            print("Warning: %d > 15 will give poor results!" % n2_max)

        self.spectra = [rydeberg(i, n2_max=n2_max)**(-1) * 10**9
                        for i in range(1, n1_series+1)]
        self.spectra = np.array(self.spectra).flatten()


def main(args):
    parser = argparse.ArgumentParser(
        prog='Elemental',
        description=('Program for converting atomic spectra to '
                     'the audible spectrum'))

    # Prints program version if prompted
    parser.add_argument('--version', action='version',
                        version='%(prog)s 0.9.2')

    parser.add_argument('-ls', '--list', default=False, action='store_const',
                        const=True, help='list elements')

    subparser = parser.add_subparsers(dest='subparser')

    # ELEMENT SPECTRA CREATION
    element_parser = subparser.add_parser(
        "element", help="Retrieves and creates spectra from element.")

    element_parser.add_argument(
        'element', default=False, type=str,
        nargs=1, help='takes the type of element. E.g. He')

    # Possible choices
    element_parser.add_argument(
        '-lf', '--local_file', default=None, type=str,
        help=('takes a .html file for an atomic spectra '
              'from nist.org'))
    element_parser.add_argument(
        '-fn', '--filename', default=None,
        type=str, help='output filename')
    element_parser.add_argument(
        '-output_folder', default="sounds",
        type=str, help='output folder')
    element_parser.add_argument(
        '-p', '--parallel', default=False,
        action='store_const', const=True,
        help='enables running in parallel')
    element_parser.add_argument(
        '-np', '--num_processors', default=4,
        type=int, help='number of processors')
    element_parser.add_argument(
        '-ln', '--length', default=10,
        type=float, help='length in seconds')
    element_parser.add_argument(
        '-hz', '--hertz', default=440,
        type=int, help='frequency')
    element_parser.add_argument(
        '-amp', '--amplitude', default=0.01,
        type=float, help='amplitude of track')
    element_parser.add_argument(
        '-sr', '--sampling_rate', default=44100,
        type=int, help='sampling rate')
    element_parser.add_argument(
        '-cf', '--convertion_factor', default=100,
        type=float,
        help='factor to pitch-shift spectra by')
    element_parser.add_argument(
        '-wlc', '--wavelength_cutoff', default=2.5e-1,
        type=float,
        help=('inverse wavelength to cutoff lower tones, '
              'default=2.5e-1.'))
    element_parser.add_argument(
        '-bt', '--beat_cutoff', default=1e-2, type=float,
        help=('removes one wavelength if two wavelengths '
              'have |lambda-lambda_0| > beat_cutoff'))

    # RYDEBERG SPECTRA CREATION
    rydeberg_parser = subparser.add_parser(
        "rydeberg", help=('Creates a Hydrogen spectra using '
                          'the Rydeberg formula. The number provide, '
                          'corresponds the number of exitation levels '
                          'that will be included.'))
    rydeberg_parser.add_argument(
        'n1', default=None, type=int,
        help=('Principal quantum number'))
    rydeberg_parser.add_argument(
        '-n2_max', default=10, type=int,
        help=('Maximal quantum number that we will exite to.'))
    rydeberg_parser.add_argument(
        '-fn', '--filename', default=None,
        type=str, help='output filename')
    rydeberg_parser.add_argument(
        '-output_folder', default="sounds",
        type=str, help='output folder')
    rydeberg_parser.add_argument(
        '-p', '--parallel', default=False,
        action='store_const', const=True,
        help='enables running in parallel')
    rydeberg_parser.add_argument(
        '-np', '--num_processors', default=4,
        type=int, help='number of processors')
    rydeberg_parser.add_argument(
        '-ln', '--length', default=10,
        type=float, help='length in seconds')
    rydeberg_parser.add_argument(
        '-hz', '--hertz', default=440,
        type=int, help='frequency')
    rydeberg_parser.add_argument(
        '-amp', '--amplitude', default=0.01,
        type=float, help='amplitude of track')
    rydeberg_parser.add_argument(
        '-sr', '--sampling_rate', default=44100,
        type=int, help='sampling rate')
    rydeberg_parser.add_argument(
        '-cf', '--convertion_factor', default=100,
        type=float,
        help='factor to pitch-shift spectra by')
    rydeberg_parser.add_argument(
        '-wlc', '--wavelength_cutoff', default=2.5e-1,
        type=float,
        help=('inverse wavelength to cutoff lower tones, '
              'default=2.5e-1.'))
    rydeberg_parser.add_argument(
        '-bt', '--beat_cutoff', default=1e-2, type=float,
        help=('removes one wavelength if two wavelengths '
              'have |lambda-lambda_0| > beat_cutoff'))

    args = parser.parse_args()

    if args.list:
        print_all_elements()
        return

    if args.subparser == "element":
        element = args.element[0]
        assert element_search(
            element), ('Element %s not found.' % element)
        Sound = Elemental(element, args.local_file,
                          args.filename, args.parallel, args.num_processors)
        if not Sound.has_spectra:
            exit("Exiting: no spectra found.")

    if args.subparser == "rydeberg":
        Sound = Rydeberg(args.n1, args.n2_max, args.filename,
                         args.parallel, args.num_processors)

    Sound.remove_beat(args.beat_cutoff)
    Sound.create_sound(args.length, args.hertz, args.amplitude,
                       args.sampling_rate, args.convertion_factor,
                       args.wavelength_cutoff,
                       output_folder=args.output_folder)


if __name__ == '__main__':
    pre_time = time.time()
    main(sys.argv[1:])
    post_time = time.time()
    print('Time used on program: %.2f seconds' % (post_time - pre_time))
