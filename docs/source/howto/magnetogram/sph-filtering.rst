Filtered Spherical Harmonics
============================

Goal
----
The SPH pipeline projects the radial magnetic field on spherical harmonics,
keeps modes up to ``lmax``, optionally damps high-degree modes with ``alpha``,
then reconstructs the map used as the COCONUT inner boundary.

Projection is performed on the native physical FITS latitude centers. The
quadrature uses exact solid angles reconstructed from the cell edges, whether
the product is uniform in sine latitude (GONG/HMI) or in latitude
(ADAPT/HMI-FDT). No uniform-latitude remapping is applied before or after the
spherical-harmonic decomposition.

Module
------
``coconut_tools.magnetogram.sph_filtering``

High-Level API
--------------
Use ``process_config`` for normal runs:

.. code-block:: python

   from coconut_tools.magnetogram.sph_filtering import process_config

   results = process_config(config)

The function accepts either a single-date configuration or a time-series
configuration. It downloads the magnetograms, reads or interpolates ``Br``,
applies the SPH reconstruction, writes the COCONUT ``.dat`` file, and writes a
diagnostic figure when ``show_map`` is true.

Single-Date Example
-------------------
To process one magnetogram, provide ``date`` and omit ``total_hours``:

.. code-block:: python

   from coconut_tools.magnetogram.sph_filtering import process_config

   config = {
       "date": "2020-12-07T15:00:00",
       "map_type": "HMI_small",
       "lmax": 30,
       "alpha": 1e-6,
       "amp": 1,
       "flux_correct": False,
       "write_map": True,
       "show_map": True,
       "output_dir": "./boundary/",
       "output_path_fig": "./figures/hmi_sph.png",
   }

   results = process_config(config)

HMI_SYNC Example
----------------
``HMI_SYNC`` downloads ``hmi.Mrdailysynframe_720s`` maps through JSOC DRMS.
Provide the JSOC email address in the configuration:

.. code-block:: python

   config = {
       "date": "2020-12-07T15:00:00",
       "map_type": "HMI_SYNC",
       "drms_email": "name@example.com",
       "lmax": 30,
       "write_map": True,
       "show_map": True,
       "output_dir": "./boundary/",
   }

   results = process_config(config)

Multi-Date GONG Example
-----------------------
To process three days with a 3-hour cadence, set ``cadence_hours=3`` and
``total_hours=72``. This processes 24 target dates.

.. code-block:: python

   from coconut_tools.magnetogram.sph_filtering import process_config

   config = {
       "date": "2025-10-09T18:00:00",
       "map_type": "GONG_mrzqs",
       "cadence_hours": 3,
       "total_hours": 72,
       "interpolation": True,
       "interpolation_order": 2,
       "flux_correct": True,
       "lmax": 20,
       "alpha": 0,
       "amp": 1,
       "write_map": True,
       "show_map": True,
       "output_dir": "./boundary/",
       "download_dir": "./raw/",
       "output_path_fig": "./figures/gong_sph.png",
   }

   results = process_config(config)

For each target date, the output file is named like:

.. code-block:: text

   map_gong_lmax20_sph_YYYYMMDDHHMMSS.dat

The diagnostic figures are named like:

.. code-block:: text

   gong_sph_YYYYMMDDHHMMSS.png

when a single ``output_path_fig`` is provided for a multi-date run.

SPH Parameters
--------------
- ``lmax``: maximum spherical harmonic degree kept in the reconstruction.
- ``amp``: global scaling factor applied during reconstruction, default ``1``.
- ``alpha``: high-degree damping factor. The coefficients are scaled by
  ``1 / (1 + alpha * l**2 * (l + 1)**2)``.

Start with ``lmax=20`` and ``alpha=0``. Raise ``lmax`` to keep more spatial
detail, and increase ``alpha`` to damp high-degree structure.

Advanced API
------------
The lower-level functions are still available when a custom workflow is needed:

.. code-block:: python

   from coconut_tools.magnetogram.io.downloads import generate_output_and_map_names
   from coconut_tools.magnetogram.io.readers import read_magnetogram
   from coconut_tools.magnetogram.io.writers import write_bc_file
   from coconut_tools.magnetogram.processing.spherical_harmonics import project_and_reconstruct
   from coconut_tools.magnetogram.visualization.plotting import plot_maps

   output_name, local_file = generate_output_and_map_names(
       "2020-12-07T15:00:00",
       "HMI_small",
       "./boundary/",
       lmax=30,
       method_used="sph",
   )

   Br, Theta, Phi = read_magnetogram(local_file, "HMI_small")
   Br_filt, coef = project_and_reconstruct(Br, Theta, Phi, lmax=30, amp=1, alpha=1e-6)
   write_bc_file(output_name, Br_filt, Theta[:, 0], Phi[0, :])
   plot_maps(
       Br,
       Br_filt,
       Theta[:, 0],
       Phi[0, :],
       "HMI_small",
       output_path="./figures/hmi_sph.png",
       date="2020-12-07T15:00:00",
   )

For GONG and ADAPT interpolation workflows, use
``generate_output_and_interpolation_map_names`` and
``read_interpolated_magnetogram`` instead of the single-map helpers.
