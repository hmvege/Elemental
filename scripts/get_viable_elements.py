import warnings
from pathlib import Path

import click

from elemental.utils import PERIODIC_TABLE, element_downloader


def retrieve_viable_elements(spectra_folder: Path) -> dict:
    """Method for retrieving viable elements in a based of the spectra data.

    If a spectra don't contain any data, it will not be added to list of
    viable elements.

    Arguments:
        spectra_folder {Path} -- path to folder containing spectra.

    Returns:
        dict -- dictionary containing full element name, atomic number and
            element short hand name.
    """

    viable_elements = []

    for element_name, element_ids, element_short in PERIODIC_TABLE:

        element_path = spectra_folder / f"{element_short}.dat"

        if not element_path.exists():
            warnings.warn(f"{element_path} not found.")

        data = element_downloader(element_name, local_file=str(element_path))

        if data.shape[0] > 0:
            viable_elements.append(
                {
                    "name": element_name,
                    "ids": int(element_ids),
                    "short": element_short,
                }
            )

    assert len(viable_elements) > 0, "Fatal error: no elements found."

    return viable_elements


@click.group()
def cli():
    pass


@click.command()
@click.option(
    "--return-type",
    default="name",
    type=click.Choice(["name", "ids", "short"]),
    show_default=True,
    help="Prints list of all elements.",
)
@click.option(
    "--spectra-folder",
    default="spectras",
    type=click.Path(exists=True),
    show_default=True,
    help="Location of spectra data.",
)
def get_elements(return_type: str = "name", spectra_folder: str = "spectras"):
    """Wrapper for retrieving viable elements."""

    viable_elements = retrieve_viable_elements(Path(spectra_folder))

    for item in viable_elements:
        print(item[return_type])


@click.command()
@click.argument("elem", nargs=1, type=str)
def elem2ids(elem: str) -> int:
    """Retrieves element number for element name."""
    for element_name, element_ids, element_short in PERIODIC_TABLE:
        if elem == element_name or elem == element_short:
            print(element_ids)
            return element_ids
    else:
        raise ValueError(f"Element {elem} not found")


@click.command()
@click.argument("ids", nargs=1, type=str)
@click.option(
    "--full-name",
    default=False,
    is_flag=True,
    show_default=True,
    help="If flagged, will return full element name.",
)
def ids2elem(ids: int, full_name=bool) -> str:
    """Retrieves element number for element name."""
    for element_name, element_ids, element_short in PERIODIC_TABLE:
        if element_ids == ids:
            if full_name:
                print(element_name)
                return element_name
            else:
                print(element_short)
                return element_short
    else:
        raise ValueError(f"Index {ids} does not match any element.")


@click.command()
@click.argument("elem", nargs=1, type=str)
def elem2full(elem: str) -> str:
    """Retrieves full element name for short element name."""
    for element_name, element_ids, element_short in PERIODIC_TABLE:
        if elem == element_short:
            print(element_name)
            return element_name
    else:
        raise ValueError(f"Index {elem} does not match any element.")


cli.add_command(get_elements)
cli.add_command(elem2ids)
cli.add_command(ids2elem)
cli.add_command(elem2full)


if __name__ == "__main__":
    cli()
