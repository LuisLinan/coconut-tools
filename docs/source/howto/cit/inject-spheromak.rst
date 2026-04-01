Injecting a Spheromak into a COCONUT CFmesh
===========================================

Context
-------
The module ``coconut_tools.CIT.cfmesh_spheromak`` injects a local linear
force-free spheromak directly into an existing ``CFmesh`` while preserving the
original file structure outside the modified state block.

This is useful when you already have a steady-state **COCONUT** solution and
want to prepare a perturbed initial condition for a time-dependent restart.

Module
------
``coconut_tools.CIT.cfmesh_spheromak``

High-level workflow
-------------------
1. Start from an existing steady-state ``corona.CFmesh``.
2. Create an example ``.ini`` configuration.
3. Adjust the spheromak placement and plasma parameters.
4. Run the injection.
5. Restart your time-dependent workflow from the generated ``CFmesh``.

Create an example configuration
-------------------------------
Use ``create_example_config`` to generate a ready-to-edit ``.ini`` file:

.. code-block:: python

   from pathlib import Path

   from coconut_tools.CIT import create_example_config

   config_path = Path("tests/_outputs/cit/spheromak_example.ini")
   create_example_config(config_path)

This writes a configuration template with the sections:

- ``[Paths]``: input mesh, output folder, case name
- ``[Spheromak]``: geometry, flux rope orientation, plasma targets
- ``[Placement]``: insertion radius
- ``[Plasma]``: scaling factors used when density/temperature are ``auto``
- ``[Visualization]``: optional VTU/VTS exports for diagnostics

Typical configuration
---------------------
The generated file can then be edited as needed:

.. code-block:: ini

   [Paths]
   input_cfmesh = tests/_bigdata_cache/corona.CFmesh
   output_dir = tests/_outputs/cit
   case_name = spheromak_test

   [Spheromak]
   lat_deg = 0.0
   lon_deg = 0.0
   radius_rsun = 5.0
   speed_km_s = 800.0
   mass_density_kg_m3 = auto
   temperature_k = auto
   helicity_sign = 1
   tilt_deg = 0.0
   toroidal_flux_wb = 1.0e14

   [Placement]
   center_radius_rsun = 10.0

   [Plasma]
   density_factor = 1.5
   temperature_factor = 1.5

   [Visualization]
   write_vtu_before = true
   write_vtu_after = true
   write_vts_before = true
   write_vts_after = true
   vts_nb_r = 200
   vts_nb_theta = 200
   vts_nb_phi = 200
   vts_eps = 0.01

Run the injection
-----------------
Once the configuration exists, call ``apply_spheromak_to_cfmesh``:

.. code-block:: python

   from coconut_tools.CIT import apply_spheromak_to_cfmesh

   result = apply_spheromak_to_cfmesh("tests/_outputs/cit/spheromak_example.ini")

   print(result.output_cfmesh)
   print(result.modified_cell_count)

The returned ``SpheromakInjectionResult`` contains:

- ``output_cfmesh``: path to the modified mesh
- ``before_vtu`` and ``after_vtu``: optional volumetric diagnostics
- ``before_vts`` and ``after_vts``: optional structured-grid diagnostics
- ``modified_cell_count``: number of mesh cells whose state was changed

Parameter notes
---------------
- ``lat_deg`` and ``lon_deg`` define the radial propagation direction.
- ``center_radius_rsun`` sets the distance of the spheromak center from the Sun.
- ``radius_rsun`` controls which cell centers are considered inside the inserted
  structure.
- ``mass_density_kg_m3 = auto`` and ``temperature_k = auto`` estimate ambient
  values around the insertion region and rescale them using
  ``density_factor`` and ``temperature_factor``.
- ``helicity_sign`` is typically ``1`` or ``-1``.
- ``toroidal_flux_wb`` sets the magnetic flux normalization.

Outputs
-------
The modified mesh is written to:

.. code-block:: text

   <output_dir>/<case_name>/<case_name>_spheromak.CFmesh

If visualization is enabled, the same folder also receives:

- ``<case_name>_before_volume.vtu``
- ``<case_name>_after_volume.vtu``
- ``<case_name>_before_structured.vts``
- ``<case_name>_after_structured.vts``

Practical notes
---------------
- The input ``CFmesh`` is not overwritten.
- File structure is preserved outside the modified state rows.
- If no cell center intersects the requested spheromak volume, the code raises
  a ``ValueError``.
- Large production meshes can generate heavy ``VTU`` and ``VTS`` outputs; set
  the visualization flags to ``false`` if you only need the modified ``CFmesh``.
