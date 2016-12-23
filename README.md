**ElementSound**

A small program for converting atomic spectra to the audible spectrum. 


Example usage:

`python ElementSound H -lf spectra/H.dat -p -ln 5`

`-lf spectra/H.dat` gets the Hydrogen spectrum from a local file. `-p` enables parallel processing with a default of 4 cores. `-ln 5` makes the recording 5 seconds long.