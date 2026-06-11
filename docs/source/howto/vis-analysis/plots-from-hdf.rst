Plotting utilities from precomputed HDF5 time series
====================================================

Goal
----
Once you have created HDF5 files with :doc:`create-hdf`, you can reuse them to
plot time series, generate surface maps, and compare multiple simulations.  
This page documents the functions in ``coconut_tools.plot.plot`` and the expected
content of the HDF5 files.

Prerequisites
-------------
- HDF5 files produced by :doc:`create-hdf` (one file **per satellite/position**).
- Typical datasets inside each file:  
  ``time``, ``Bx``, ``By``, ``Bz``, ``B`` (optional total),  
  ``Vx``, ``Vy``, ``Vz``, ``V`` (optional total),  
  ``rho`` (density), ``T`` (temperature), and optionally spherical components
  (``vr, vlon, vclt, br, blon, bclt``) if they were saved.

Module
------
``coconut_tools.plot.plot``

1) Plot boundary in-situ style profiles (one or several HDF5 files)
-------------------------------------------------------------------

.. code-block:: python

   def plot_boundary_profil(inputdir, outputfile, label_dict, color_map):
       """Plot velocity, magnetic field, density, temperature from HDF5."""

**Inputs**

- ``inputdir``: directory that contains the HDF5 files.
- ``label_dict``: ``{filename: "legend label"}`` (loop order = legend order).
- ``color_map``: ``{"legend label": "color"}`` to assign colors consistently.
- Each HDF5 must contain (at least):  
  ``vr, vlon, vclt, br, blon, bclt, rho (or density), T (temperature), time``.

**What it does**

- Computes total **B** and **V** from components, builds a common time axis (hours).
- Produces a 4-panel figure:  
  **V [km/s]**, **B [nT]**, **Density [m⁻³]**, **Temperature [K]**.
- Optionally adds vertical markers/annotations if you color one curve in **purple**
  (see the example in the source for “Sheath/ME” markers).

**Minimal example**

.. code-block:: python

   from coconut_tools.plot.plot import plot_boundary_profil

   inputdir = "./data_tests/"
   outputfile = "./data_tests/test_plot_boundary.png"
   label_dict = {"file1.h5": "CME1", "file2.h5": "CME2"}
   color_map  = {"CME1": "blue", "CME2": "purple"}

   plot_boundary_profil(inputdir, outputfile, label_dict, color_map)

2) One-time 2D surface maps from a ``.dat`` boundary
----------------------------------------------------
If you saved boundary maps (``.dat``) with :doc:`/howto/boundary/create-dat-rotate`,
you can render 2D fields for a **single snapshot**.

.. code-block:: python

   def Surface_2D_onetime(inputfile: str, outputfile: str,
                          mode: Literal['all','reduced'] = 'all') -> None:

- **mode='all'**: plots 8 variables (density, temperature, Br/Bclt/Blon, Vr/Vclt/Vlon)  
- **mode='reduced'**: density, temperature, **Br**, **Vr** only  
- Internally uses ``read_dat_files.read_data`` to parse the map.

**Example**

.. code-block:: python

   from coconut_tools.plot.plot import Surface_2D_onetime

   Surface_2D_onetime(
       inputfile="./data_tests/test.dat",
       outputfile="./data_tests/test_surface_plot.png",
       mode="all"
   )

3) Compare magnetic components for two cases (solar min/max)
------------------------------------------------------------

.. code-block:: python

   def create_plot_B(path_min: str, path_max: str,
                     save_path: str | None = None,
                     scale_factor: float = 0.402,
                     show: bool = False) -> None:

- Reads HDF5 files for two cases (e.g., **solar minimum** vs **solar maximum**),
  plots **B, Bx, By, Bz** vs time, with shaded “sheath/ME” intervals (indices set inside).
- ``scale_factor`` converts simulation time to **hours** (consistent with your pipeline).

**Example**

.. code-block:: python

   from coconut_tools.plot.plot import create_plot_B

   create_plot_B(
       path_min="/path/to/min/Data_16.hdf5",
       path_max="/path/to/max/Data_6.hdf5",
       save_path="B_components.pdf",
       show=False
   )

4) Compare several runs (ζ sweep) on V, B, density
--------------------------------------------------

.. code-block:: python

   def create_plot_comparison(
       data_indices: dict, input_dir: str,
       output_dir: str, output_filename: str = "comp_zeta_max.pdf",
       time_scale: float = 0.402) -> None:

- Expects HDF5 files named like ``Data_<index>.hdf5`` under ``input_dir``.  
- Plots **V [km/s]**, **B [nT]**, **rho** for **multiple runs** with a rainbow colormap.
- Handy to compare a **parameter sweep** (e.g., ζ) on the same axes.

**Example**

.. code-block:: python

   from coconut_tools.plot.plot import create_plot_comparison

   create_plot_comparison(
       data_indices={3: "ζ=3", 4: "ζ=4", 5: "ζ=5", 6: "ζ=6"},
       input_dir="/path/to/plot/",
       output_dir="/path/to/plots/",
       output_filename="comparison_maximum.pdf"
   )

5) Scaling of maxima vs background field B0 (min/max)
-----------------------------------------------------

.. code-block:: python

   def create_plot_max_quantities_vs_b0(
       dict_folder_min: dict, dict_folder_max: dict,
       input_dir: str, output_dir: str,
       time_limit_index: int = 50,
       output_filename: str = "max_quantities.pdf") -> None:

- For each ζ, loads the corresponding HDF5 (from ``min_result/plot`` and ``max_result/plot``),
  extracts **max V**, **max B**, **max density**, and fits trends **vs B0**:
  - linear fits for **V** and **B**,
  - quadratic fit for **density**.
- Useful to quantify how output extrema scale with input **B0** across solar min/max.

**Example**

.. code-block:: python

   from coconut_tools.plot.plot import create_plot_max_quantities_vs_b0
   import numpy as np

   dict_folder_min = {z: b0 for z, b0 in zip(range(2,21), np.linspace(1, 20, 19))}
   dict_folder_max = {z: b0 for z, b0 in zip(range(2,21), np.linspace(1.5, 25, 19))}

   create_plot_max_quantities_vs_b0(
       dict_folder_min=dict_folder_min,
       dict_folder_max=dict_folder_max,
       input_dir="/base/dir/",
       output_dir="/base/dir/plots/",
       time_limit_index=50
   )

Notes & best practices
----------------------
- **Start from HDF5**: :doc:`create-hdf` keeps your analysis fast and reproducible.
- **Units**: make sure your HDF5 contains data in **physical units** (or apply consistent conversions).
- **Consistent cadence**: the ``time`` array in HDF5 should already encode your
  chosen scaling (see :doc:`create-hdf` for the formula).
- **Legends/colors**: use ``label_dict`` and ``color_map`` to keep styles consistent across figures.
- **Publication ready**: increase DPI (``dpi=300/600``) and export **PDF** for papers.
- **Provenance**: commit tiny YAML/JSON files with the lists of HDF5 and labels you used for each plot.

See also
--------
- :doc:`create-hdf` — generate the HDF5 time series from CFmesh/VTU
- :doc:`read-output` — read CFmesh/VTU into Python dictionaries
- :doc:`pyvista-threeD` — quick 3D PyVista exploration of VTU snapshots
- :doc:`plot-convergence` — check solver convergence from COCONUT logs
