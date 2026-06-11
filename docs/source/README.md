# coconut-tools

[![Documentation Status](https://readthedocs.org/projects/coconut-tools/badge/?version=latest)](https://coconut-tools.readthedocs.io/en/latest/?badge=latest)

Utilities to read and visualize 3D coronal simulation results, compare with observations, and prepare inputs for heliospheric models such as EUHFORIA.

> **Note**  
> The modules in this repository are primarily meant as **examples** to show how to work with COCONUT outputs, rather than a fixed, stable library. Feel free to modify the code to suit your own workflows.

---

## Documentation

Full documentation: **https://coconut-tools.readthedocs.io/**

- **Tutorial** (installation & first steps): see the “Tutorial” section in the docs  
- **How-To Guides** (short recipes & details): see the “How-To” section in the docs

---

## Features

### Reading and visualization
- `tools/how_to_read_output.py` – Example utilities to load COCONUT outputs
- `plot/` and `visualization_3d/` – Tools to visualize simulation outputs
- `postprocessing/group_vtu_files.py` – Prepares and groups VTU outputs for use in Paraview
- `plot/tomography.py` – Comparison between COCONUT results and observational data

### Input preparation
- `toheliosphere/create_dat.py`, `tools/rotation_angle.py` – Prepare boundary/input data for coupling with heliospheric models (e.g. EUHFORIA)

### CME flux rope models (`pyTDM/`)
- Adapted from the original package by Florian Regnault (for PLUTO), extended and modified by Luis Linan and Jinhan Guo to inject CME models (RBSL, TDm) into COCONUT initial conditions

### Inner boundary construction (`magnetogram/`)
- Scripts to download and preprocess solar magnetograms  
- Original routines by José Murteira, cleaned, modularized and refactored by Luis Linan

---

## License

MIT © 2025 Luis Linan
