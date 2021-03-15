from pathlib import Path

import click
from pydub import AudioSegment
from tqdm import tqdm

from elemental.utils import PERIODIC_TABLE


def spectra_sort_key(key):
    """Sort key for elements.

    Sort key function for locating and returning the element index if
    and only if it exists in the PERIODIC_TABLE.

    Arguments:
        key {str} -- element file name to search for.

    Returns:
        int -- atomic number or index of element.

    Raises:
        ValueError -- if element is not found, error is raised.
    """
    element_name = key.name.split(".")[0].split("_")[0]
    for i, _element in enumerate(PERIODIC_TABLE):
        if element_name == _element[2]:
            return i
    else:
        raise ValueError(
            f"Element {element_name} not located for file input {key}."
        )


@click.command()
@click.argument("audio_folder", type=click.Path(exists=True))
@click.argument("output_folder", type=click.Path(exists=True))
def generate_elements_audio(audio_folder: str, output_folder: str):
    """Clips out audio segment for elements found in provided audio_folder.

    Using pydub, one will cut out 8 seconds from each element sound,
    and stitch them together to one, single file, of 8.2 seconds each,
    cross faded into each other, such that the total length will
    be 8 * N_elements.
    """

    audio_folder = Path(audio_folder)
    output_folder = Path(output_folder)

    # Milliseconds
    ms = 1000

    print(f"Loading audio clips from {audio_folder}.")

    sorted_elements = sorted(audio_folder.iterdir(), key=spectra_sort_key)

    assert (
        len(sorted_elements) > 3
    ), "Needs at least 3 elements for creating audio."

    output_audio = AudioSegment.from_wav(sorted_elements[0])

    assert len(output_audio) > 10, "Requires at least to seconds of audio."

    # Sets up the first element audio, fading sound in
    output_audio = output_audio[1 * ms : 9 * ms]
    output_audio = output_audio.fade_in(2 * ms)

    for element_path in tqdm(sorted_elements[1:-1]):

        audio = AudioSegment.from_wav(element_path)

        audio = audio[1 * ms : int(9.2 * ms)]
        output_audio = output_audio.append(audio, crossfade=int(ms * 0.2))

    # Sets up final audio element and fading out.
    end_audio = AudioSegment.from_wav(sorted_elements[-1])
    end_audio = end_audio[1 * ms : int(9.2 * ms)]
    end_audio = end_audio.fade_out(2 * ms)

    output_audio = output_audio.append(end_audio, crossfade=int(ms * 0.2))

    output_fname = output_folder / "sound_of_the_elements.wav"
    print(f"Generated {output_fname} with length of {len(output_audio)}")
    output_audio.export(output_fname, format="wav", bitrate="320k")


if __name__ == "__main__":
    generate_elements_audio()
