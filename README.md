# Elemental

A small program for converting atomic spectra to the audible spectrum.

There are two options for creating sounds from atomic spectra. One is using experimental data, which takes in the observed wavelengths of different spectroscopies. The other is using the Rydeberg formula, but which is limited to just the Hydrogen spectrum.

<iframe width="560" height="315" src="https://www.youtube.com/embed/8777qC9-W9Q" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Description
The wavelengths are generated by taking the wavelengths of the spectra `wl`, and converting them to the Hertz by
```python
spectra = 1 / wl * convertion_factor
```
where typically `convertion_factor=100`.

The spectra is then used to build the output sound, by summing over all sine waves for each of the spectra,

```python
sound = sum(sin(2 * pi * frequency * spectra * time))
```

the sampling rate is included in the time resolution,

```python
N_length = N_seconds * sampling_rate
```

## Installation
Create a Python virtual environment using
```bash
$ python3 -m venv venv
```
Then, activate the environment,
```bash
$ source venv/bin/activate
```
and install using
```bash
$ pip install -e .
```

### Install in `dev` mode
```bash
$ pip install -e ".[dev]"
```
The quotes are included in case you are using `zsh`.

## Downloading spectra
Download atomic spectra using
```bash
$ python spectra_retriever path/to/spectra/output/folder
```

## Usage
Use `elemental --help` or `rydeberg --help` to view run commands.

### Example using `elementa`
```bash
$ elemental U -lf spectra/U.dat -ln 20
```

### Example using `rydeberg`
```bash
$ rydeberg 4
```

`-lf spectra/H.dat` gets the Hydrogen spectrum from a local file. `-p` enables parallel processing with a default of 4 cores. `-ln 5` makes the recording 5 seconds long.


## Testing
Run `pytest` to unit test code base.
