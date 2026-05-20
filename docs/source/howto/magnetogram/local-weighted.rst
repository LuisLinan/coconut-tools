Local Weighted Filtering
========================

Goal
----
The local weighted filter, also called the Yaroslavsky filter, smooths each
pixel from its neighborhood. The weights depend on both spatial distance and
magnetic-field similarity, so sharp active-region boundaries are better
preserved than with a simple average.

Module
------
``coconut_tools.magnetogram.Yaroslavsky_filter``

High-Level API
--------------
Use ``process_config`` for normal runs:

.. code-block:: python

   from coconut_tools.magnetogram.Yaroslavsky_filter import process_config

   results = process_config(config)

This high-level pipeline shares the same date handling, downloading,
GONG/ADAPT interpolation, net-flux correction, COCONUT ``.dat`` writing, and
diagnostic plotting logic as the SPH pipeline.

Single-Date Example
-------------------
To process one magnetogram, provide ``date`` and omit ``total_hours``:

.. code-block:: python

   from coconut_tools.magnetogram.Yaroslavsky_filter import process_config

   config = {
       "date": "2020-12-07T15:00:00",
       "map_type": "HMI_small",
       "lmax": 20,
       "alpha": 1.2,
       "Rn": 5.0,
       "sig": 0.0,
       "flux_correct": False,
       "write_map": True,
       "show_map": True,
       "output_dir": "./boundary/",
       "output_path_fig": "./figures/hmi_yaroslavsky.png",
   }

   results = process_config(config)

Multi-Date GONG Example
-----------------------
The same time-series options used by SPH are available here. This example
processes three days at a 3-hour cadence and interpolates each target date from
four neighboring GONG maps.

.. code-block:: python

   from coconut_tools.magnetogram.Yaroslavsky_filter import process_config

   config = {
       "date": "2025-10-09T18:00:00",
       "map_type": "GONG",
       "cadence_hours": 3,
       "total_hours": 72,
       "interpolation": True,
       "interpolation_order": 2,
       "flux_correct": True,
       "lmax": 20,
       "alpha": 1.2,
       "Rn": 5.0,
       "sig": 0.0,
       "write_map": True,
       "show_map": True,
       "output_dir": "./boundary/",
       "download_dir": "./raw/",
       "output_path_fig": "./figures/gong_yaroslavsky.png",
   }

   results = process_config(config)

For each target date, the output file is named like:

.. code-block:: text

   map_gong_lmax20_Yaroslavsky_YYYYMMDDHHMMSS.dat

Filter Parameters
-----------------
- ``alpha``: contrast-weighting factor passed to
  ``filter_radial_field_weighted``. Larger values make the filter less willing
  to average pixels with different magnetic-field values.
- ``Rn``: neighborhood radius used by the local filter.
- ``sig``: optional Gaussian pre-smoothing width, default ``0.0``.
- ``write_gaussian_prepass``: optional debug output for the Gaussian prepass,
  default ``False``.

Advanced API
------------
The low-level filter is still available for custom workflows:

.. code-block:: python

   from coconut_tools.magnetogram.Yaroslavsky_filter import filter_radial_field_weighted
   from coconut_tools.magnetogram.sph_filtering import write_bc_file, plot_maps

   Br_filtered = filter_radial_field_weighted(
       Br,
       Phi[0, :],
       Theta[:, 0],
       alpha_factor=1.2,
       Rn=5.0,
       sig=0.0,
   )

   write_bc_file("map_gong_lmax20_Yaroslavsky_20251009180000.dat", Br_filtered, Theta[:, 0], Phi[0, :])
   plot_maps(
       Br,
       Br_filtered,
       Theta[:, 0],
       Phi[0, :],
       "GONG",
       output_path="./figures/gong_yaroslavsky.png",
       date="2025-10-09T18:00:00",
   )
