# Elemental

A small program for converting atomic spectra to the audible spectrum.

There are two options for creating sounds from atomic spectra. One is using experimental data, which takes in the observed wavelengths of different spectroscopies. The other is using the Rydeberg formula, but which is limited to just the Hydrogen spectrum.

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

TODO:
- add option to create emission spectra images.
- check if one can retrieve the intensity from the energy in a transition.
- add dev branch?
- perform code refactoring on comments, style, modularity
- add unit tests