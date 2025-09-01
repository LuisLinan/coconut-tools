Precompute time series at fixed points → create HDF5 from CFmesh / VTU
======================================================================

Goal
----
COCONUT’s computational grid is **unstructured**. If you want the evolution of a quantity at
a *specific* point in space (e.g., spacecraft location), that exact point almost never
coincides with a mesh node. Interpolating **on the fly** for every plot is costly and brittle.
Instead, we **precompute** (and cache) the time series at the requested locations and store
them once in **HDF5**: fast to read, portable, and reusable across analyses.

This How-To shows how to:
- sample a **time sequence** of outputs (CFmesh or merged VTU),
- at **multiple positions** (e.g. multiple spacecraft),
- and **save one HDF5 file per position** containing the time evolution of magnetic and thermodynamic quantities (and spherical components when reading CFmesh).

Module
------
``coconut_tools.create_hdf``

Functions
-------------------
Two entry points (you can use either CFmesh or VTU as data source):

.. code-block:: python

   def create_hdf_from_vtu(
       satellite_positions: Dict[str, List[float]],
       folder_patterns: Dict[str, str],
       output_dir: Path,
       delta_t: float,
       time_step: float
   ) -> None:
       """Extract plasma quantities from VTU files at satellite positions and store them in HDF5."""

and

.. code-block:: python

   def create_hdf_from_cfmesh(
       satellite_positions: Dict[str, Tuple[float, float, float]],
       folder_patterns: Dict[str, str],
       output_dir: Path,
       delta_t: float,
       time_step: float,
       method: str = "nearest",
   ) -> None:
       """Extract plasma quantities from CFmesh files and save time series (incl. spherical components)."""

**Shared concepts**

- ``satellite_positions``: a mapping ``{name: (x, y, z)}`` with positions in **Cartesian R⊙**.
- ``folder_patterns``: a mapping ``{case: glob_pattern}`` (e.g., ``"COCONUT_A": "…/caseA/*.CFmesh"``).
- ``delta_t`` and ``time_step``: define the effective cadence of outputs (see *Time handling* below).
- ``output_dir``: an HDF5 per satellite will be written under this directory, with one group/dataset per case.

Why precompute?
---------------
- **Performance**: interpolation on an unstructured 3D grid per figure is expensive; HDF5 avoids recomputing.
- **Reproducibility**: a single file contains the exact sampled time series used in publications/plots.
- **Convenience**: downstream notebooks/scripts just read HDF5 and plot.

Interpolation strategies (CFmesh)
---------------------------------

1) Nearest-neighbor sampling (fastest)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Find the mesh node **closest** to the target position and read fields there.

.. code-block:: python

   def find_closest_index_cf(
       x: np.ndarray, y: np.ndarray, z: np.ndarray,
       target_coord: Tuple[float, float, float]
   ) -> int:
       """
       Return the index of the point closest to the target (Cartesian, in R⊙).
       """
       target = np.array(target_coord)
       points = np.stack((x, y, z), axis=1)
       distances = np.linalg.norm(points - target, axis=1)
       return np.argmin(distances)

Pros: simple, very fast, robust.
Cons: no sub-cell accuracy; can be noisy if the grid is coarse at the target.

2) RBF interpolation (smoother, more accurate)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Radial Basis Function interpolation from surrounding points to the requested location.

.. code-block:: python

   # grid: (N, 3) array of (x, y, z) node coordinates
   # Bx, By, Bz: arrays of per-node values
   # sat: (1, 3) satellite position (x, y, z) in R⊙

   bx = float(RBFInterpolator(grid, Bx, kernel="linear", neighbors=20)(sat)[0])
   by = float(RBFInterpolator(grid, By, kernel="linear", neighbors=20)(sat)[0])
   bz = float(RBFInterpolator(grid, Bz, kernel="linear", neighbors=20)(sat)[0])

Pros: sub-cell accuracy, smoother series.  
Cons: costlier; choose reasonable ``neighbors`` for speed/robustness.

What gets written (datasets)
----------------------------
- **From VTU**: cartesian or spherical fields depending on your reader; typically
  arrays of time, Bx, By, Bz, Vx, Vy, Vz, rho, P (and possibly derived quantities if you add them).
- **From CFmesh**: in addition to cartesian components, spherical components are saved:
  ``vr, vclt, vlon, br, bclt, blon`` (with the convention **θ is colatitude**).

The HDF5 file layout is typically:

- one **file per satellite** (e.g., ``STA.h5``),
- inside, **one group per case** (e.g., ``/CFMESH_A``, ``/CFMESH_B``),
- within each, **datasets**: ``time``, ``Bx``, ``By``, ``Bz``, ``Vx``, ``Vy``, ``Vz``, ``rho``, ``T``, ``P``,and for CFmesh also ``vr``, ``vclt``, ``vlon``, ``br``, ``bclt``, ``blon``.

Time handling
-------------
We map a file index ``i`` to a physical time via your scaling:

.. code-block:: text

   t[i] = i * delta_t * time_step * 0.402


- ``delta_t``: base timestep in **code units** (per your run configuration),
- ``time_step``: output stride (e.g., write every ``time_step`` base-steps),
- ``0.402``: your additional scaling factor used across the project.

Choose values consistent with how your files were written, so that plots are on a meaningful time axis.

Recommended source: CFmesh (when available)
-------------------------------------------
You can build HDF5 either from **VTU** or **CFmesh**. When you have both, **CFmesh is preferred**:
- full state, consistent across snapshots,
- easier access to both cartesian and spherical components,
- no need to first merge distributed VTU files (although you *can* and it’s supported).

Examples
--------

Example — VTU inputs (one HDF5 per satellite)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   from pathlib import Path
   from coconut_tools.create_hdf import create_hdf_from_vtu

   satellite_cartesian = {"STA": [-10.0, 0.0, 0.0]}   # positions in R_sun (x, y, z)
   delta_t  = 0.005
   time_step = 20
   output_dir = Path("test_hdf_output")

   vtu_folders = {
       "COCONUT_A": "test_data/vtu/caseA/*.vtu",
       "COCONUT_B": "test_data/vtu/caseB/*.vtu",
   }

   create_hdf_from_vtu(
       satellite_positions=satellite_cartesian,
       folder_patterns=vtu_folders,
       output_dir=output_dir / "vtu",
       delta_t=delta_t,
       time_step=time_step
   )

Example — CFmesh inputs (nearest or RBF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   from pathlib import Path
   from coconut_tools.create_hdf import create_hdf_from_cfmesh

   satellite_cartesian = {"STA": [-10.0, 0.0, 0.0]}   # (x, y, z) in R_sun
   delta_t  = 0.005
   time_step = 20
   output_dir = Path("test_hdf_output")

   cfmesh_cases = {
       "CFMESH_A": "test_data/cfmesh/caseA/*.CFmesh",
       "CFMESH_B": "test_data/cfmesh/caseB/*.CFmesh",
   }

   # Nearest neighbor (fast)
   create_hdf_from_cfmesh(
       satellite_positions=satellite_cartesian,
       folder_patterns=cfmesh_cases,
       output_dir=output_dir / "cfmesh_nearest",
       delta_t=delta_t,
       time_step=time_step,
       method="nearest"
   )

   # RBF interpolation (smoother)
   create_hdf_from_cfmesh(
       satellite_positions=satellite_cartesian,
       folder_patterns=cfmesh_cases,
       output_dir=output_dir / "cfmesh_rbf",
       delta_t=delta_t,
       time_step=time_step,
       method="interpolate"
   )

Notes & best practices
----------------------
- **CFmesh vs VTU**: Prefer CFmesh if available; use VTU (merged) otherwise. See :doc:`merge-vtu` for merging.
- **Multiple satellites**: pass several entries in ``satellite_positions``; you’ll get one HDF5 per key.
- **Multiple cases**: add as many glob patterns as needed; each becomes a group in the HDF5.
- **Units**: positions are in **R⊙**; ensure your spacecraft ephemerides are converted properly.
- **Method choice**: start with ``nearest`` (fast); if series look noisy or grid is coarse, switch to ``interpolate``.
- **Reproducibility**: commit the tiny config (satellite dict, delta_t, time_step) alongside the produced HDF5.
- **Downstream usage**: HDF5 reads are trivial in NumPy/Pandas/xarray; the files are ideal for notebooks and quick plotting.

Related
-------
- :doc:`read-output` — readers for CFmesh and VTU into Python dictionaries
- :doc:`merge-vtu` — if your VTU outputs are distributed per MPI rank
- :doc:`/api`
