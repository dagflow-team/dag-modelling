
# Change Log

All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.14.2] - 2025-10-16

- chore: make `pygraphviz` optional.

## [0.14.1] - 2025-10-02

- feature: enable the file reader to read data from `TMatrixT`/`TGraph*` classes with uproot.
- feature: enable TSV file reader to read just a single file without object name.

## [0.14.0] - 2025-09-25

- feature: add `save_matrices` function similar to `save_records`, but for matrices. Saves dicitonary
    with matrices as hdf5, npz, root, tsv or archived tsv.
- [chore] minor changes and cleaning.

## [0.13.0] - 2025-09-08

- `make_fcn`: add positional parameters, clean.
- Add `disable_implicit_numpy_multithreading` module.
- Introduce `set_verbosity(int)` function, the simpler version to use instead of the logger's `set_level()`

## [0.12.0] - 2025-07-29

- First PYPI version.

## [0.11.0] - 2025-07-25

- Merge upstream changes for profiling.

## [0.10.0] - 2025-07-24
  
Major updates:
- Merge upstream changes
- Import `dgf_detector` nodes for:
    * energy resolution histogram smearing
    * rebinning
    * distortion of histogram via x axis
- Import `dgf_statistics` nodes for:
    * χ², CNP stat errors, log Poisson ratio
    * MonteCarlo random sampling
 
## [0.9.0] - 2025-04-07
  
The pre-release version
