import time
from pathlib import Path

import click

from elemental.elemental import Rydeberg, Elemental
from elemental.spectra_retriever import spectra_retriever
from .utils import element_search


def default_sound_options(function):
    """Method containing common decorators for elemental and Rydeberg."""
    function = click.option(
        "-o",
        "--output_folder",
        default=Path("sounds"),
        type=click.Path(exists=True),
        help="Output folder",
    )(function)
    function = click.option(
        "-p", "--parallel", is_flag=True, help="Enables parallel computation."
    )(function)
    function = click.option(
        "-np",
        "--num_processors",
        type=int,
        default=4,
        help="Number of threads to start.",
    )(function)
    function = click.option(
        "-ln",
        "--length",
        type=float,
        default=10,
        show_default=True,
        help="Length of output sound file in seconds.",
    )(function)
    function = click.option(
        "-hz",
        "--hertz",
        type=int,
        default=440,
        show_default=True,
        help="Frequency to generate sound at.",
    )(function)
    function = click.option(
        "-a",
        "--amplitude",
        type=float,
        default=0.01,
        show_default=True,
        help="Amplitude of track.",
    )(function)
    function = click.option(
        "-sr",
        "--sampling_rate",
        type=int,
        default=44100,
        show_default=True,
        help="Sampling rate.",
    )(function)
    function = click.option(
        "-cf",
        "--convertion_factor",
        type=float,
        default=100,
        show_default=True,
        help="Factor to pitch-shift spectra by.",
    )(function)
    function = click.option(
        "-wc",
        "--wavelength_cutoff",
        type=float,
        default=2.5e-1,
        show_default=True,
        help="Inverse wavelength to cutoff lower tones.",
    )(function)
    function = click.option(
        "-bc",
        "--beat_cutoff",
        type=float,
        default=1e-2,
        show_default=True,
        help=(
            "Removes one wavelength if two wavelengths have "
            "|lambda-lambda_0| > beat_cutoff"
        ),
    )(function)
    function = click.option(
        "-ev",
        "--envelope_length",
        type=float,
        default=0.2,
        show_default=True,
        help="Length of the envelope",
    )(function)
    function = click.option(
        "-v",
        "--verbose",
        default=False,
        show_default=True,
        is_flag=True,
        help="More verbose run.",
    )(function)
    return function


@click.command()
@click.argument("element", nargs=1, type=str)
@click.option(
    "-lf",
    "--local_elem_file",
    default=False,
    type=click.Path(exists=True),
    help="Takes a file containing a list of emissions(in nanometers).",
)
@default_sound_options
def elemental(
    element: str,
    local_elem_file: click.Path,
    output_folder: click.Path,
    parallel: bool,
    num_processors: int,
    length: float,
    hertz: int,
    amplitude: float,
    sampling_rate: int,
    convertion_factor: float,
    wavelength_cutoff: float,
    beat_cutoff: float,
    envelope_length: float,
    verbose: bool,
):
    """Generates audible spectra for given element. To run, specify an element
    by its periodic table name, e.g. 'U' for Uranium.
    """

    if verbose:
        pre_time = time.time()

    assert element_search(element), f"Element {element} not found."

    Sound = Elemental(
        element, local_elem_file, parallel, num_processors, verbose
    )

    if not Sound.has_spectra:
        raise click.BadParameter("No spectra found.")

    Sound.remove_beat(beat_cutoff)
    Sound.create_sound(
        length,
        hertz,
        amplitude,
        sampling_rate,
        convertion_factor,
        wavelength_cutoff,
        output_folder=Path(output_folder),
        envelope_length=envelope_length,
    )

    if verbose:
        post_time = time.time()
        print("Time used on program: %.2f seconds" % (post_time - pre_time))


@click.command()
@click.argument("n1", nargs=1, type=int)
@click.option(
    "--n2",
    nargs=1,
    default=10,
    show_default=True,
    type=int,
    help="Maximal quantum number that we will generate exitation spectra for.",
)
@default_sound_options
def rydeberg(
    n1: int,
    n2: int,
    output_folder: click.Path,
    parallel: bool,
    num_processors: int,
    length: float,
    hertz: int,
    amplitude: float,
    sampling_rate: int,
    convertion_factor: float,
    wavelength_cutoff: float,
    beat_cutoff: float,
    envelope_length: float,
    verbose: bool,
):
    """Generates audible spectra based off the Rydeberg formula.

    Takes in the principal quantum numbers n1 and n2, and generates spectra
    from emission between those two.
    """

    if verbose:
        pre_time = time.time()

    Sound = Rydeberg(n1, n2, parallel, num_processors, verbose)

    Sound.remove_beat(beat_cutoff)
    Sound.create_sound(
        length,
        hertz,
        amplitude,
        sampling_rate,
        convertion_factor,
        wavelength_cutoff,
        output_folder=Path(output_folder),
        envelope_length=envelope_length,
    )

    if verbose:
        post_time = time.time()
        print("Time used on program: %.2f seconds" % (post_time - pre_time))


@click.command()
@click.argument(
    "output_folder",
    default=Path("downloaded_spectras"),
    type=click.Path(exists=True),
)
@click.option(
    "-f",
    "--force_redownload",
    default=False,
    show_default=True,
    is_flag=True,
    help=(
        "Will re-download spectra, even if they already exists in provided "
        "output folder."
    ),
)
def download_spectra(output_folder: click.Path, force_redownload: bool):
    """Downloads all of the available atom spectra from nist.gov.
    """
    spectra_retriever(Path(output_folder), force_redownload)
