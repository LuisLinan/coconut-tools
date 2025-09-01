Read CFmesh & VTU in Python
===========================

Goal
----
Load **COCONUT** outputs either from the 3D **CFmesh** format or from the (merged) **VTU** files,
and expose the main physical quantities in convenient Python dictionaries.

Module
------
``coconut_tools.how_to_read_output``

Prerequisites
-------------
- For VTU workflows, you usually want the **merged** files first (one per snapshot). See :doc:`merge-vtu`.
- `how_to_read_output.py` provides two reader utilities:
  - ``read_cfmesh_file`` → parse CFmesh and return cartesian fields and metadata.
  - ``read_vtu_to_spherical_vectors`` → read VTU and convert to **spherical** components.



CFmesh reader
-------------

.. code-block:: python

   def read_cfmesh_file(input_cfmesh: str) -> dict:
       """Read and parse a CFmesh output file from COCONUT simulations.

       Args:
           input_cfmesh (str): Path to the CFmesh input file.

       Returns:
           dict: Dictionary containing parsed mesh and physical quantities:
               - 'connectivity': element connectivity array
               - 'coordinates': node coordinates array
               - 'cell_centers': computed center positions for each element
               - 'B': magnetic field components (Bx, By, Bz)
               - 'V': velocity components (Vx, Vy, Vz)
               - 'rho': density (cm^-3)
               - 'P': pressure (Pa)
               - 'T': temperature (K)
               - 'r': radial distances of cell centers
       """

VTU → spherical reader
----------------------

.. code-block:: python

   def read_vtu_to_spherical_vectors(input_vtu: str) -> dict:
       """Convert a VTU file from COCONUT into spherical coordinates and physical quantities.

       Args:
           input_vtu (str): Path to the VTU file to read.

       Returns:
           dict: Dictionary of spherical components and coordinates including:
               - 'r': radial distances
               - 'theta': polar angle (rad)
               - 'phi': azimuthal angle (rad)
               - 'phi2pi': azimuth in [0, 2pi]
               - 'vr': radial velocity
               - 'vclt': colatitudinal velocity
               - 'vlon': longitudinal velocity
               - 'br': radial magnetic field
               - 'bclt': colatitudinal magnetic field
               - 'blon': longitudinal magnetic field
               - 'rho': density
               - 'p': pressure
       """

**Unit notes (if normalization is required)**

.. code-block:: python

   # velocity (code → km/s)
   vrkms = vr * 480248.0
   vpkms = vlon * 480248.0
   vtkms = vclt * 480248.0

   # magnetic field (code → Tesla)
   brt   = br   * 2.2e-4
   blont = blon * 2.2e-4
   bclt  = bclt * 2.2e-4

   # temperature and density
   tk  = p / rho * 1.7756e7   # K
   nsi = rho * 1.0e14         # kg/m^3

Examples
--------

Minimal VTU example
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from coconut_tools import how_to_read_output as reader

   # Example VTU file path (merged snapshot)
   input_path = "./example_coconut_output.vtu"   # replace with a real file path
   result = reader.read_vtu_to_spherical_vectors(input_path)

   print(result.keys())        # r, theta, phi, vr, vclt, vlon, br, bclt, blon, rho, p, ...

Minimal CFmesh example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from coconut_tools import how_to_read_output as reader

   input_path = "./example_coconut_output.CFmesh"   # replace with a real file path
   result = reader.read_cfmesh_file(input_path)

   print(result.keys())        # connectivity, coordinates, cell_centers, B, V, rho, P, T, r

Usage tips
----------

- **Merged VTU first**: when your simulation produced distributed parts per MPI rank, use the merge utility first
  (:doc:`merge-vtu`) so you can read a single file per snapshot.
- **Pick your frame**: CFmesh reader exposes **cartesian** field components, while the VTU reader returns **spherical**
  components (``vr, vclt, vlon, br, bclt, blon``). Choose what best suits your analysis.
- **Normalizations**: if your run uses code units, apply the conversion factors above to obtain SI / physical units.
- **Downstream plots**: once loaded, you can pass the arrays to your own matplotlib/pyvista routines, or use
  helpers like ``pyvista_slice`` for quicklooks.
