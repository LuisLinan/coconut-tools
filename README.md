# coconut-tools

[![Documentation Status](https://readthedocs.org/projects/coconut-tools/badge/?version=latest)](https://coconut-tools.readthedocs.io/en/latest/?badge=latest)


Tools for **COCONUT**: utilities to read and visualize 3D coronal simulation results, compare with observations, and prepare inputs for heliospheric models such as EUHFORIA.

---

## Installation

Clone the repository and install in editable mode (recommended for development).

### On Windows (PowerShell)

```powershell
git clone https://github.com/LuisLinan/coconut-tools.git
cd coconut-tools
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
```

### On Linux / macOS (bash)

```bash
git clone https://github.com/LuisLinan/coconut-tools.git
cd coconut-tools
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
```

---

## Features

### Reading and visualization
- **`how_to_read_output.py`** – Example utilities to load COCONUT outputs.  
- **`plot.py`, `plot_convergence.py`, `pyvista_slice.py`** – Tools to visualize simulation outputs directly in Python.  
- **`group_vtu_files.py`** – Prepares and groups VTU outputs for use in Paraview.  
- **`tomography.py`** – Comparison between COCONUT results and observational data.  

### Input preparation
- **`create_dat.py`, `rotation_angle.py`** – Prepare boundary/input data for coupling with heliospheric models (e.g. EUHFORIA).  

### CME flux rope models (`pyTDM/`)
- Adapted from the original package by Florian Regnault (for PLUTO), extended and modified by Luis Linan to inject CME models (RBSL, TDm) into COCONUT initial conditions.  

### Inner boundary construction (`magnetogram/`)
- Scripts to download and preprocess solar magnetograms.  
- Original routines by José Murteira, **cleaned, modularized and refactored by Luis Linan**.  

---

## Usage

Each script can be:
- Imported as a module:
  ```python
  from coconut_tools import plot
  plot.main("output_directory")
  ```
- Or executed directly (includes an example in `if __name__ == "__main__":`):
  ```bash
  python src/coconut_tools/plot.py
  ```

---

## Documentation

The full documentation is hosted on [Read the Docs](https://coconut-tools.readthedocs.io/en/latest/).

---

## License

MIT © 2025 Luis Linan
