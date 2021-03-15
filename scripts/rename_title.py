"""Short script for renaming output audio files to
a friendlier format."""
from pathlib import Path

from elemental.utils import PERIODIC_TABLE

video_folder = Path("output_mp4")

video_length = 600  # seconds

for elem_full, ids, elem in PERIODIC_TABLE:

    old_title = video_folder / Path(f"{elem}_{video_length}sec.mp4")
    if not old_title.is_file():
        continue

    new_name = Path(f"Sound_of_{elem_full.upper()}.mp4")
    new_title = video_folder / new_name

    old_title.rename(new_title)
    print(f"{old_title} -> {new_title}")
