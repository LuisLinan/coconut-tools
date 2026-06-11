Compare COCONUT with Tomographic Electron Density
=================================================

Goal
----
Tomographic reconstructions provide **electron density** maps at several heliocentric
distances. Comparing these with **COCONUT** helps validate the simulation and detect
systematic offsets (e.g., current sheet latitude/longitude).

This page shows how to:

1. Read a merged **COCONUT VTU** snapshot and a **tomography ``.dat``** map.
2. Interpolate onto a common grid and **identify the current sheet**.
3. Produce a **three-panel figure**:

- COCONUT density map at a given radius,
- Tomography density map at the same radius,
- **Latitude of maximum density vs longitude** (COCONUT vs tomography).

Script module
-------------
``coconut_tools.plot.tomography``

Docstring overview:

- Reads a ``.vtu`` simulation file and a tomography ``.dat`` file
- Interpolates fields and locates the current sheet
- Generates comparative plots

**Recommended citations** (when using images/data):

- Morgan 2015 — https://iopscience.iop.org/article/10.1088/0067-0049/219/2/23/meta
- Morgan 2019 — https://iopscience.iop.org/article/10.3847/1538-4365/ab125d/meta
- Morgan 2020 — https://iopscience.iop.org/article/10.3847/1538-4357/ab7e32/meta

API entry point
---------------
.. code-block:: python

   def compare_tomography_with_simulation(vtufile: str, datfile: str, output: str) -> None:
       """
       Compare electron density between a COCONUT MHD simulation and a tomographic reconstruction.

       Args:
           vtufile (str): Path to the COCONUT output (.vtu).
           datfile (str): Path to the tomography .dat file.
           output (str): Path to save the output figure.

       Returns:
           None
       """

What the 3-panel plot contains
------------------------------
Internally, the function produces:

- **Panel 1 (COCONUT)** — density map with contour overlays
- **Panel 2 (Tomography)** — density map (same colormap/extent for comparability)
- **Panel 3 (Line comparison)** — latitude (deg) of **maximum density** vs longitude (deg)
  for both datasets, to track the **current sheet** position

Illustrative snippet from the implementation:

.. code-block:: python

   im1 = ax1.imshow(dens_int, cmap='gist_stern', origin='upper', extent=extent)
   fig.colorbar(im1, ax=ax1).set_label('Density [norm.]', fontsize=12)
   ax1.contour(np.degrees(lon_tomo), np.degrees(clt_tomo - np.pi / 2.0), np.flip(dens_int, axis=0),
               levels=[0.2, 0.4, 0.6, 0.8], colors='k', linewidths=0.5, linestyles='dashed')
   ax1.set_title(f'COCONUT at R={r_tomo:.2f}$R_\\odot$')
   ax1.set_xlabel('Longitude [°]')
   ax1.set_ylabel('Latitude [°]')

   im2 = ax2.imshow(dens_tomo, cmap='gist_stern', origin='upper', extent=extent)
   fig.colorbar(im2, ax=ax2).set_label('Density [norm.]', fontsize=12)
   ax2.contour(np.degrees(lon_tomo), np.degrees(clt_tomo - np.pi / 2.0), np.flip(dens_tomo, axis=0),
               levels=[0.2, 0.4, 0.6, 0.8], colors='k', linewidths=0.5, linestyles='dashed')
   ax2.set_title(f'Tomography at R={r_tomo:.2f}$R_\\odot$')
   ax2.set_xlabel('Longitude [°]')
   ax2.set_ylabel('Latitude [°]')

   ax3.set_title(f'Maximum of density at R={r_tomo:.2f}$R_\\odot$')
   ax3.plot(np.degrees(lon_tomo), np.degrees(listcolatitude_smooth), label='Tomography')
   ax3.plot(np.degrees(lon_tomo), np.degrees(listcolatitude_coco_smooth), label='COCONUT')
   ax3.set_xlabel('Longitude [°]')
   ax3.set_ylabel('Latitude [°]')
   ax3.legend()
   ax3.set_xlim(extent[0], extent[1])

Usage example
-------------
Typical call pattern (with automatic download of the tomography file if missing):

.. code-block:: python

   import os
   from coconut_tools.plot.tomography import compare_tomography_with_simulation
   from coconut_tools.tools.download_tomo_file import download_tomography_file

   date = "20190704"
   altitude = "5-0"
   output_dir = "C:/Users/luisl/Desktop/Tinatin"
   datfile = os.path.join(output_dir, f"tomo_sta_cor2a_{date}_{altitude}.dat")

   if not os.path.exists(datfile):
       download_tomography_file(date, altitude, output_dir)

   vtufile = "C:/Users/luisl/Desktop/hmi/corona-mhd_0.vtu"
   output  = os.path.join(output_dir, "tomo_3.pdf")

   compare_tomography_with_simulation(vtufile, datfile, output)

Notes & guidance
----------------
- **Altitude consistency**: choose a tomography altitude (e.g., ``"5-0"``) that matches
  the radius you extract/compare in COCONUT.
- **Merged VTU**: if your COCONUT outputs are distributed per MPI rank, merge first
  (see :doc:`/howto/vis-analysis/merge-vtu`).
- **Color normalization**: use consistent colormaps/ranges across both panels for fair visual comparison.
- **Current sheet**: the third panel is very sensitive; apply reasonable smoothing to
  avoid over-interpreting small-scale numerical noise.
- **Attribution**: cite the Morgan papers listed above when using tomography products.
