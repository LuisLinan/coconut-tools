Nonlinear Diffusion Filtering
=============================

Goal
----
The nonlinear diffusion filter applies an edge-preserving Perona-Malik style
diffusion to the radial magnetic field. It reduces small-scale noise while
limiting smoothing across strong active-region gradients.

Module
------
``coconut_tools.magnetogram.NLD_implicit_method``

High-Level API
--------------
Use ``process_config`` for normal runs:

.. code-block:: python

   from coconut_tools.magnetogram.NLD_implicit_method import process_config

   results = process_config(config)

This high-level pipeline shares the same date handling, downloading,
GONG/ADAPT interpolation, net-flux correction, COCONUT ``.dat`` writing, and
diagnostic plotting logic as the SPH pipeline.

Single-Date Example
-------------------
To process one magnetogram, provide ``date`` and omit ``total_hours``:

.. code-block:: python

   from coconut_tools.magnetogram.NLD_implicit_method import process_config

   config = {
       "date": "2020-12-07T15:00:00",
       "map_type": "HMI_small",
       "lmax": 20,
       "tau": 5,
       "iterations": 7,
       "apply_gaussian": True,
       "gaussian_sigma": 1.0,
       "flux_correct": False,
       "write_map": True,
       "show_map": True,
       "output_dir": "./boundary/",
       "output_path_fig": "./figures/hmi_nld.png",
   }

   results = process_config(config)

Multi-Date GONG Example
-----------------------
The same time-series options used by SPH are available here. This example
processes three days at a 3-hour cadence and interpolates each target date from
four neighboring GONG maps.

.. code-block:: python

   from coconut_tools.magnetogram.NLD_implicit_method import process_config

   config = {
       "date": "2025-10-09T18:00:00",
       "map_type": "GONG",
       "cadence_hours": 3,
       "total_hours": 72,
       "interpolation": True,
       "interpolation_order": 2,
       "flux_correct": True,
       "lmax": 20,
       "tau": 5,
       "iterations": 7,
       "apply_gaussian": True,
       "gaussian_sigma": 1.0,
       "write_map": True,
       "show_map": True,
       "output_dir": "./boundary/",
       "download_dir": "./raw/",
       "output_path_fig": "./figures/gong_nld.png",
   }

   results = process_config(config)

For each target date, the output file is named like:

.. code-block:: text

   map_gong_lmax20_NLD_YYYYMMDDHHMMSS.dat

Filter Parameters
-----------------
- ``tau``: diffusion time step.
- ``iterations``: number of nonlinear diffusion iterations.
- ``apply_gaussian``: apply Gaussian pre-smoothing before nonlinear diffusion,
  default ``True``.
- ``gaussian_sigma``: Gaussian kernel width.
- ``dx_override`` and ``dy_override``: optional grid-spacing overrides used by
  the implicit solver, both default to ``1.0``.

Start with ``tau=5`` and around 6 to 7 iterations, then tune according to the
remaining small-scale structure and the stability of the COCONUT run.

Advanced API
------------
The low-level filter is still available for custom workflows:

.. code-block:: python

   from coconut_tools.magnetogram.NLD_implicit_method import filter_radial_field
   from coconut_tools.magnetogram.sph_filtering import write_bc_file, plot_maps

   Br_filtered, timestep = filter_radial_field(
       Br,
       Phi[0, :],
       Theta[:, 0],
       iterations=7,
       tau=5,
       apply_gaussian=True,
       gaussian_sigma=1.0,
   )

   write_bc_file("map_gong_lmax20_NLD_20251009180000.dat", Br_filtered, Theta[:, 0], Phi[0, :])
   plot_maps(
       Br,
       Br_filtered,
       Theta[:, 0],
       Phi[0, :],
       "GONG",
       output_path="./figures/gong_nld.png",
       date="2025-10-09T18:00:00",
   )
