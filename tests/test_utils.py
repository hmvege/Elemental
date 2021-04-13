import tempfile
# from typing import List
from pathlib import Path

import numpy as np
import pytest

from elemental.utils import (
    PERIODIC_TABLE,
    element_downloader,
    check_folder,
    get_element,
)


def test_spectra_retriever():

    tmp_folder = tempfile.TemporaryDirectory(suffix="_spectra_retriever")
    output_folder = Path(tmp_folder.name)

    dummy_data = np.random.randn(len(PERIODIC_TABLE), 10)
    test_files = [
        output_folder / f"{elem}.dat" for (_, _, elem) in PERIODIC_TABLE
    ]

    # Generating dummy data
    for i, test_file in enumerate(test_files):

        with open(test_file, "w") as f:

            for d in dummy_data[i]:
                f.write(f"{d:.16e}\n")

    output_array = np.zeros_like(dummy_data)

    # Verifying that element_downloader locates the correct file.
    for i, (_, _, elem) in enumerate(PERIODIC_TABLE):

        output_array[i, :] = element_downloader(
            elem, local_file=str(test_files[i]),
        )

    assert np.array_equal(dummy_data, output_array), (
        "Mismatch in read input data, and test data."
    )

    tmp_folder.cleanup()


def test_check_folder():
    """Verifies that correct folder is created."""
    tmp_folder = tempfile.TemporaryDirectory(suffix="_check_folder")
    test_folder_path = Path(tmp_folder.name) / "test_folder"

    check_folder(test_folder_path, verbose=True)

    assert test_folder_path.exists(), (
        f"Folder {str(test_folder_path)} not found."
    )


@pytest.mark.parametrize(
    "element, exists",
    [
        ("H", True), ("Ag", True), ("U", True), ("Up", False), ("Du", False)
    ]
)
def test_get_element(element: str, exists: bool):
    """Verifies correct element is retrieved."""

    element_result = get_element(element) is False

    assert element_result is not exists, (
        f"Error in element '{element}' search. Should be {exists}"
    )
