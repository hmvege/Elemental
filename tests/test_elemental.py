import tempfile
from typing import List
from pathlib import Path

import numpy as np
import pytest
from scipy.io.wavfile import read as wavfile_read

from elemental.elemental import (
    create_tones,
    _tone_parallel,
    _calculate_rydeberg_spectra,
    rydeberg,
    Elemental,
    Rydeberg,
)


RYDEBERG_CONSTANT = 1.09677583e7


@pytest.mark.parametrize(
    "hz,spectra,N",
    [
        (1.0, np.array([1.0]), 100),
        (2.0, np.array([1.0, 2.0]), 1000),
    ],
)
def test_create_tones(hz: float, spectra: np.ndarray, N: int):
    """Test tone calculation method."""
    t = np.linspace(0, 1, N)
    envelope = np.ones(N)

    tone = create_tones(t, spectra, envelope, hz)

    spectrum = spectra[:, np.newaxis] * t[np.newaxis, :]
    test_tone = np.sin(2 * np.pi * hz * spectrum).sum(axis=0)

    assert np.allclose(tone, test_tone)


def test_tone_parallel():
    """Test for the parallel unpacking calculation method."""
    N = 100

    hz = 1.0

    spectra = np.array([1.0])

    t = np.linspace(0, 1, N)
    envelope = np.ones(N)

    tone = _tone_parallel([t, spectra, envelope, hz])

    spectrum = spectra[:, np.newaxis] * t[np.newaxis, :]
    test_tone = np.sin(2 * np.pi * hz * spectrum).sum(axis=0)

    assert np.allclose(tone, test_tone)


@pytest.mark.parametrize(
    "n1,n2,R",
    [
        (1.0, 2.0, 1.0),
        (1.0, 4.0, RYDEBERG_CONSTANT),
        (2.0, 4.0, RYDEBERG_CONSTANT),
        (3.0, 4.0, RYDEBERG_CONSTANT),
    ],
)
def test_calculate_rydeberg(n1: float, n2: float, R: float):
    """Basic test for the Rydeberg formula calculation."""
    wl = _calculate_rydeberg_spectra(n1, n2, R)

    assert np.isclose(
        wl, (R * (1.0 / n1 ** 2 - 1.0 / n2 ** 2)), atol=1e-15
    ), "Mismatch in Rydeberg formula calculation."


@pytest.mark.parametrize(
    "n1,n2_list",
    [
        (1.0, [2.0]),
        (1.0, [2.0, 3.0]),
        (1.0, [2.0, 3.0, 4.0]),
        (1.0, [2.0, 3.0, 4.0, 5.0]),
        (1.0, [2.0, 3.0, 4.0, 5.0, 6.0]),
        (2.0, [3.0, 4.0, 5.0, 6.0]),
        (3.0, [4.0, 5.0, 6.0]),
        (4.0, [5.0, 6.0]),
        (5.0, [6.0]),
    ],
)
def test_rydeberg(n1: float, n2_list: List[float]):
    """
    Test for the Rydeberg calculation, which involves several quantum
    numbers.
    """
    n2_arr = np.asarray(n2_list)
    n1_arr = np.ones_like(n2_list) * n1
    wl_test = RYDEBERG_CONSTANT * (1.0 / n1_arr ** 2 - 1.0 / n2_arr ** 2)

    wl = rydeberg(int(n1), len(n2_list))

    assert np.allclose(wl, wl_test, atol=1e-15), "Mismatch in Rydeberg."


@pytest.mark.parametrize(
    "n1,n2_max,parallel",
    [
        (1.0, 2.0, True),
        (1.0, 3.0, False),
        (1.0, 4.0, True),
        (1.0, 5.0, False),
        (1.0, 6.0, True),
        (2.0, 6.0, False),
        (3.0, 6.0, True),
        (4.0, 6.0, False),
        (5.0, 6.0, False),
    ],
)
def test_rydeberg_sound_creation(n1, n2_max, parallel):
    """Test for the Rydeberg sound creation.

    Creates temporary input values in a file, loads those, and verify the
    output with what is to be expected.
    """

    # Static parameters
    length = 5
    hz = 440
    amplitude = 0.01
    sampling_rate = 1000
    convertion_factor = 1.0
    wavelength_cutoff = 0  # No non-audible wavelengths cutoff when testing.
    envelope_length = 0.2

    tmp_folder = tempfile.TemporaryDirectory(suffix="_rydeberg_utest")
    output_folder = Path(tmp_folder.name)

    r = Rydeberg(int(n1), int(n2_max), parallel=parallel)
    # r.remove_beat()
    r.create_sound(
        length=length,
        hz=hz,
        amplitude=amplitude,
        sampling_rate=sampling_rate,
        convertion_factor=convertion_factor,
        wavelength_cutoff=wavelength_cutoff,
        envelope_length=envelope_length,
        output_folder=output_folder,
    )

    rate, rydeberg_sound = wavfile_read(
        output_folder / f"H_Rydeberg_n{int(n1)}_{int(length)}sec.wav"
    )

    # Sets up Rydeberg wavelength and spectra to test against
    def get_test_rydeberg(_n1, _n2):
        _n2_arr = np.arange(_n1 + 1, _n1 + _n2 + 1, dtype=int)
        _n1_arr = np.ones_like(_n2_arr) * _n1
        return RYDEBERG_CONSTANT * (1.0 / _n1_arr ** 2 - 1.0 / _n2_arr ** 2)

    # Generates test Rydeberg spectras
    spectra_arr = np.array(
        [
            (1 / get_test_rydeberg(i, n2_max)) * (10 ** 9)
            for i in range(1, int(n1) + 1)
        ]
    ).flatten()

    # Verifies that the spectra created match
    assert np.allclose(
        spectra_arr, r.spectra, atol=1e-15
    ), "Mismatch in spectra"

    # Creates sample sound
    N = int(length * sampling_rate)
    t = np.linspace(0, length, N)
    spectra_arr = (1 / spectra_arr) * convertion_factor
    spectrum = spectra_arr[:, np.newaxis] * t[np.newaxis, :]
    test_sound = np.sin(2 * np.pi * hz * spectrum).sum(axis=0)

    # Tests the envelope.
    N_envelope_length = int(sampling_rate * envelope_length)
    test_envelope = np.ones(N)
    envelope_t = np.linspace(0, np.pi, N_envelope_length)
    test_envelope_function = (1 - np.cos(envelope_t)) / 2
    test_envelope[:N_envelope_length] = test_envelope_function
    test_envelope[N - N_envelope_length:] = test_envelope_function[::-1]
    test_envelope *= amplitude

    env_func = r._envelope_function(N_envelope_length)
    assert np.allclose(
        env_func, test_envelope_function, atol=1e-15
    ), "Mismatch in envelope functions."
    env = r._envelope(N, sampling_rate, length, amplitude, envelope_length)
    assert np.allclose(
        env, test_envelope, atol=1e-15
    ), "Mismatch in envelope arrays."

    test_sound *= test_envelope

    # Final verification of the output sounds.
    assert rate == sampling_rate, (
        f"Output sampling_rate({rate}) not matching input sampling "
        f"rate({sampling_rate})"
    )
    assert np.allclose(
        rydeberg_sound, test_sound, atol=1e-15
    ), "Mismatch between output sound and expected sound"
    assert np.allclose(
        rydeberg_sound, r.tone, atol=1e-15
    ), "Mismatch between output sound and generated sound"
    assert np.allclose(
        r.tone, test_sound, atol=1e-15
    ), "Mismatch between generated sound and test sound"

    tmp_folder.cleanup()


@pytest.mark.parametrize("parallel", [(True), (False)])
def test_elemental_sound_creation(parallel):
    """Test for Elemental sound class creator.

    Creates temporary input values in a file, loads those, and verify the
    output with what is to be expected.
    """

    num_test_spectras = 10

    # Static parameters
    length = 5
    hz = 440
    amplitude = 0.01
    sampling_rate = 1000
    convertion_factor = 1.0
    wavelength_cutoff = 0  # No non-audible wavelengths cutoff when testing.
    envelope_length = 0.2

    test_name = "elemental_test"
    tmp_folder = tempfile.TemporaryDirectory(suffix="_elemental_utest")
    output_folder = Path(tmp_folder.name)
    test_spectra_file = output_folder / Path(f"{test_name}.dat")

    np.savetxt(
        test_spectra_file,
        np.random.uniform(low=80, high=120, size=num_test_spectras),
    )

    elem = Elemental(
        test_name, local_file=str(test_spectra_file), parallel=parallel
    )
    # r.remove_beat()
    elem.create_sound(
        length=length,
        hz=hz,
        amplitude=amplitude,
        sampling_rate=sampling_rate,
        convertion_factor=convertion_factor,
        wavelength_cutoff=wavelength_cutoff,
        envelope_length=envelope_length,
        output_folder=output_folder,
    )

    rate, output_sound = wavfile_read(
        output_folder / f"{test_name}_{int(length)}sec.wav"
    )

    spectra_arr = np.loadtxt(test_spectra_file, dtype=float)

    # Verifies that the spectra created match
    assert np.allclose(
        spectra_arr, elem.spectra, atol=1e-15
    ), "Mismatch in spectra"

    # Generates test spectra
    N = int(length * sampling_rate)
    t = np.linspace(0, length, N)
    spectra_arr = (1 / spectra_arr) * convertion_factor
    spectrum = spectra_arr[:, np.newaxis] * t[np.newaxis, :]
    test_sound = np.sin(2 * np.pi * hz * spectrum).sum(axis=0)

    env = elem._envelope(N, sampling_rate, length, amplitude, envelope_length)
    test_sound *= env

    # Final verification of the output sounds.
    assert rate == sampling_rate, (
        f"Output sampling_rate({rate}) not matching input sampling "
        f"rate({sampling_rate})"
    )
    assert np.allclose(
        output_sound, test_sound, atol=1e-15
    ), "Mismatch between output sound and expected sound"
    assert np.allclose(
        output_sound, elem.tone, atol=1e-15
    ), "Mismatch between output sound and generated sound"
    assert np.allclose(
        elem.tone, test_sound, atol=1e-15
    ), "Mismatch between generated sound and test sound"

    tmp_folder.cleanup()
