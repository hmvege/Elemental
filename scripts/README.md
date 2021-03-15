# Scripts description and run guide


## File descriptions
All files a run in the `venv` set up by `setup.py`.

### `generate_dscr.py`
Run by `python scripts/generate_dscr.py <short element name>`, and generates video description.

### `create_elements_video.sh`
Runs the pipeline for generating a single audio file containing samples all elements.

### `create_videos.sh`
Runs the pipeline for generating a 10 minute long audio file for each of the elements.

### `rename_title.py`
Renames the generated audio files for each element.

### `generate_elements_audio.py`
Generates the audio file containing all elements.

### `generate_emission_spectra.py`
Generates the emission spectra images for all the elements.

### `get_viable_elements.py`
Small program which contain several help functions for element naming and verifications. These method include

- `python get_viable_elements.py get_elements [options]` which prints all available elements.
- `python get_viable_elements.py elem2ids [options]` returns the element index for its shortened element name.
- `python get_viable_elements.py ids2elem [options]` returns the element element name for its index. Has option to either return full or shorthand name.
- `python get_viable_elements.py elem2full [options]` returns the full element name for its shortened element name.

## Generating audio for all elements
To generate the audio for all elements, run
```
$ create_videos.sh
```

## Generating single file containing all elements
The generate single movie containing samples of all the elements, run
```bash
python generate_emission_spectra.py --element_watermark
bash create_elements_video.sh
```
The running of `generate_emission_spectra.py` is needed in order to generate output `.png`s with a watermark of the element name. Next, running `create_elements_video.sh` generates the video itself.