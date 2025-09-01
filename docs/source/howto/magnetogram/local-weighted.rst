Local weighted filtering (Yaroslavsky)
======================================

Goal
----
Apply fast local weighted averaging to smooth magnetograms. The weight
depends on both spatial distance and field intensity similarity.

API
---
.. code-block:: python

   Br_filtered = filter_radial_field_weighted(
       Br, phi, theta,
       alpha_factor, Rn, sig=1.0
   )

Parameters
~~~~~~~~~~
- **Br**: 2D array of radial magnetic field.
- **phi, theta**: 1D coordinate arrays.
- **alpha_factor**: controls contrast weighting.
- **Rn**: neighborhood radius (physical units).
- **sig**: optional Gaussian pre-smoothing.

Example
-------
.. code-block:: python

   from coconut_tools.magnetogram.Yaroslavsky_filter import filter_radial_field_weighted
   from coconut_tools.magnetogram.sph_filtering import write_bc_file, plot_maps

   Br_yaro = filter_radial_field_weighted(
       Br, Phi[0,:], Theta[:,0],
       alpha_factor=1.2, Rn=5.0
   )

   write_bc_file("bcfile_yaro.dat", Br_yaro, Theta[:,0], Phi[0,:])
   plot_maps(Br_yaro, Theta, Phi, save_path="magnetogram_yaro.png")

Notes
-----
- Very fast (< 1 s for typical HMI maps).
- Edge-preserving by design.
- Parameters to tune: neighborhood radius (Rn) and contrast weight (alpha).
- Good for quick tests before committing to SPH or ND.
