Nonlinear diffusion filter (ND)
===============================

Goal
----
Apply edge-preserving nonlinear diffusion (Perona–Malik) to magnetograms,
reducing noise while maintaining active region boundaries.

API
---
.. code-block:: python

   Br_filtered, timestep = filter_radial_field(
       Br, phi, theta,
       iterations=7, tau=5,
       apply_gaussian=True, gaussian_sigma=1.0
   )

Parameters
~~~~~~~~~~
- **Br**: 2D array of radial magnetic field.
- **phi, theta**: 1D coordinate arrays.
- **iterations**: number of nonlinear diffusion iterations.
- **tau**: time step for diffusion.
- **apply_gaussian**: pre-smooth with a Gaussian (default True).
- **gaussian_sigma**: Gaussian kernel width.

Example
-------
.. code-block:: python

   from coconut_tools.magnetogram.NLD_implicit_method import filter_radial_field
   from coconut_tools.magnetogram.sph_filtering import write_bc_file, plot_maps

   Br_nd, _ = filter_radial_field(
       Br, Phi[0,:], Theta[:,0],
       iterations=7, tau=5
   )

   write_bc_file("bcfile.dat", Br_nd, Theta[:,0], Phi[0,:])
   plot_maps(Br_nd, Theta, Phi, save_path="magnetogram_nd.png")

Notes
-----
- Typical parameters: ``tau=5``, ~6–7 iterations.
- Good for HMI maps, preserves AR edges.
- Runtime: ~tens of seconds.
