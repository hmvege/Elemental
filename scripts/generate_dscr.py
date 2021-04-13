import sys

from elemental.utils import PERIODIC_TABLE


def main(elem):
    """Short script for generating the description string based on
    input element."""
    github = "https://github.com/hmvege/Elemental"
    playlist = (
        "https://www.youtube.com/playlist?list"
        "=PLvrvRAqSLFNV5VrUYaBxeMSsUnlGIlIGM"
    )

    def generate_descr(element, el, code_url, playlist_url):
        """Generates description string."""
        description = (
            f"This is the atomic spectra of {element} [{el}] translated"
            " to an audible range.\n\nTo do this, we started by retrieving"
            " the frequency from the inverse of the wavelength. We ignored "
            "any phase velocity(in this case, the speed of light since we "
            "have EM radiation) since the frequency would be pitch shifted "
            "to an audible one anyway. The frequency is then used to generate"
            " a sine wave for each wavelength in the spectra, and then added "
            "together to form the sound.\n\nFull code and explanation can be"
            f" found can be found at {code_url}.\n\n"
            f"Playlist for all elements: {playlist_url}"
        )
        return description

    element = list(
        filter(
            lambda i: elem.capitalize() == i[0] or elem.capitalize() == i[2],
            PERIODIC_TABLE,
        )
    )
    assert len(element) == 1
    element = element[0]

    print(generate_descr(element[0], element[2], github, playlist))


if __name__ == "__main__":
    main(sys.argv[1])
