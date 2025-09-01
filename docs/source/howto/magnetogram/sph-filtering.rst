Filtered spherical harmonics (SPH)
==================================

Goal
----
Filter synoptic magnetograms via spherical harmonic decomposition to smooth
small-scale noise and control the intensity range, while keeping large-scale
features that drive the corona.

API
---
.. code-block:: python

   Br_filt, coef = project_and_reconstruct(
       Br, Theta, Phi,
       lmax, amp=1, alpha=0
   )

Parameters
~~~~~~~~~~
- **Br**: 2D array of radial magnetic field (θ, φ).
- **Theta, Phi**: grids (radians).
- **lmax**: maximum spherical harmonic degree kept.
- **amp**: global scaling factor (default 1).
- **alpha**: high-ℓ damping factor
  (scales coefficients by ``1 / (1 + alpha * l²(l+1)²)``).

Helpers
-------
- ``read_magnetogram(file, map_type)`` → load HMI, GONG, ADAPT maps.
- ``write_bc_file(name, Br, theta, phi, r_st=1.0)`` → save a COCONUT boundary.
- ``plot_maps(Br, Theta, Phi, visu_type="sinlat")`` → quick visualization.

Example workflow
----------------
.. code-block:: python

   from coconut_tools.magnetogram.sph_filtering import (
       generate_output_and_map_names,
       read_magnetogram, project_and_reconstruct,
       write_bc_file, plot_maps
   )

   date = "2020-12-07T15:00:00"
   map_type = "HMI_small"
   output_dir = "./"

   # Names
   output_name, local_file = generate_output_and_map_names(
       date, map_type, output_dir, lmax=30, method="SPH"
   )

   # Load map
   Br, Theta, Phi = read_magnetogram(local_file, map_type)

   # Filter
   Br_filt, _ = project_and_reconstruct(Br, Theta, Phi, lmax=30, alpha=1e-6)

   # Save and plot
   write_bc_file(output_name, Br_filt, Theta[:,0], Phi[0,:])
   plot_maps(Br_filt, Theta, Phi, save_path="filtered.png")

Notes
-----
- Start with ``lmax=20, alpha=0``.  
- Raise ``lmax`` gradually to add detail; increase ``alpha`` for stronger damping.
- ``lmax=50, alpha=3e-6`` worked well in practice, but is slower.
