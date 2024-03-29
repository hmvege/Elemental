# Elemental

A small program for converting atomic spectra to the audible spectrum.

There are two options for creating sounds from atomic spectra. One is using experimental data, which takes in the observed wavelengths of different spectroscopies. The other is using the Rydeberg formula, but which is limited to just the Hydrogen spectrum.

<div align="center">
    <a href="https://www.youtube.com/watch?v=8777qC9-W9Q">
     <img src="https://img.youtube.com/vi/8777qC9-W9Q/0.jpg" alt="Sound of the Elements" style="width:100%;">
    </a>
</div>
<!-- [![Sound of the Elements](https://img.youtube.com/vi/8777qC9-W9Q/0.jpg)](https://www.youtube.com/watch?v=8777qC9-W9Q) -->

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
```bash
$ pip install .
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
