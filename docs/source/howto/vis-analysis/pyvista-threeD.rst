Interactive 3D visualization with PyVista (alternative to ParaView)
===================================================================

Goal
----
Provide a **Python-only** path to explore COCONUT outputs in 3D: slices, clipping,
streamlines, and isosurfaces — without opening ParaView. This page showcases the
module ``coconut_tools.visualization_3d.pyvista_slice`` and highlights PyVista features you can
adapt to your own analysis. The demo figure is **illustrative**, not publication-ready:
read the code to see **how** each element is built and tailor it to your needs. 

When to use this
----------------
- You want quick **3D sanity checks** directly from a notebook or script.
- You prefer scripting your visuals (reproducibility) instead of clicking through a GUI.
- You’d like to **prototype** streamlines, 2D slices, spherical clipping, and isosurfaces
  to later refine them in a dedicated workflow. 

Module overview
---------------
``coconut_tools.visualization_3d.pyvista_slice`` exposes a minimal pipeline:

1) **Read** a merged VTU snapshot:

.. code-block:: python

    mesh = pv.read(filename)  # read_mesh()
    # returns a pyvista.DataSet

(The helper logs progress and returns a PyVista mesh.) 

2) **Convert units** & add derived fields (density/pressure/temperature, V and B in physical units):

.. code-block:: python

    mesh['rho_dim'] = mesh['rho'] * 1.67e-16
    mesh['prs_dim'] = mesh['p'] * 0.3851
    mesh['temp']    = 7.7e-9 * mesh['prs_dim'] / mesh['rho_dim']
    v = mesh['v'] * 480.24838  # km/s
    mesh['vx_dim'], mesh['vy_dim'], mesh['vz_dim'] = v[:,0], v[:,1], v[:,2]
    mesh['bx_dim'] = mesh['Bx'] * 2.2
    mesh['by_dim'] = mesh['By'] * 2.2
    mesh['bz_dim'] = mesh['Bz'] * 2.2

*(These are the unit conversions implemented in the example pipeline — adjust as needed.)* 

3) **Convert to spherical** & compute radial components:

.. code-block:: python

    r   = np.sqrt(x**2 + y**2 + z**2)
    rxy = np.sqrt(x**2 + y**2) + 1e-20

    mesh['vr']     = (x*vx + y*vy + z*vz) / r
    mesh['vtheta'] = (x*z*vx + y*z*vy - rxy**2*vz) / (rxy*r)
    mesh['vphi']   = (-y*vx + x*vy) / rxy

    mesh['br']     = (x*bx + y*by + z*bz) / r
    mesh['btheta'] = (x*z*bx + y*z*by - rxy**2*bz) / (rxy*r)
    mesh['bphi']   = (-y*bx + x*by) / rxy

*(Gives you vr/vθ/vφ and Br/Bθ/Bφ for plotting.)*

4) **Visualize** in PyVista with several layers:

   - a 2D **slice** colored by ``vr``,
   - a **clipped sphere** colored by ``br``,
   - **magnetic field streamlines**,
   - an **isosurface** of density (optional). 

Feature gallery (how it’s done)
-------------------------------
**2D slice**

.. code-block:: python

   slice_plane = mesh.slice(normal='y')
   p.add_mesh(slice_plane, scalars='vr', cmap='coolwarm', scalar_bar_args={...})

**Clipped spherical shell**

.. code-block:: python

   sphere  = pv.Sphere(center=(0,0,0), radius=1.01)
   clipped = mesh.clip_surface(sphere)
   p.add_mesh(clipped, scalars='br', cmap='seismic', clim=[-1,1], scalar_bar_args={...})

**Streamlines (with robust typing)**

.. code-block:: python

   mesh['B'] = np.column_stack((mesh['bx_dim'], mesh['by_dim'], mesh['bz_dim']))
   # helper ensures ints where VTK requires them
   stream = make_streamlines(mesh, vectors='B', max_steps=1000, n_points=1000, source_radius=2.0, source_center=(0,0,0))
   p.add_mesh(stream.tube(radius=0.01), cmap='binary', scalar_bar_args={...})


.. note::
   The helper casts ``max_steps`` and ``n_points`` to ``int`` to avoid VTK type errors — a common pitfall.

**Isosurface**

.. code-block:: python

   iso = mesh.contour([1e-16], scalars='rho_dim')
   p.add_mesh(iso, scalars='rho_dim', cmap='plasma', opacity=0.4, scalar_bar_args={...})

End-to-end demo
---------------
A minimal script wired in the module shows the pipeline:

.. code-block:: python

    mesh = read_mesh("corona-mhd.vtu")
    mesh = convert_units(mesh)
    mesh = convert_to_spherical(mesh)
    visualize(mesh, slice_normal='y', save_path='pyvista_slice.png', show=False)

The ``visualize`` function configures a camera, adds axes, and can either **display**
an interactive window or **save** a high-resolution screenshot (PNG/JPG/PDF/SVG). 

Why this matters (vs ParaView)
------------------------------
- It’s **scriptable**: perfect for reproducible notebooks and batch runs.
- You can keep **analysis and visualization** in the **same Python** pipeline,
  pass arrays around, and tune colorbars/thresholds programmatically.
- It’s a **learning scaffold**: the module collects PyVista calls you can re-use.
  *Don’t hesitate to open the code and borrow functions (streamlines, 2D slices, clipping, isosurfaces).* 

Tips & troubleshooting
----------------------
- **Use merged VTU** inputs (:doc:`merge-vtu`) so you visualize a **single** file per snapshot.
- **VTK offscreen** (clusters/servers): you may need environment vars (e.g. ``PYVISTA_OFF_SCREEN=1``)
  to generate screenshots without a display.
- **Streamlines** often fail if kwargs are wrong type (floats vs ints). The helper ``make_streamlines``
  normalizes types and forwards only non-None options — copy this idiom if you customize. 
- **Colormaps & bars**: the demo shows how to position/tune scalar bars (``scalar_bar_args``).
- **Units**: conversions in ``convert_units`` are illustrative; adjust constants to your run’s definitions. 

See also
--------
- :doc:`merge-vtu` — prepare merged VTU snapshots before loading in PyVista
- :doc:`read-output` — dictionary readers for CFmesh/VTU (spherical & cartesian)
- :doc:`/howto/boundary/create-dat-rotate` — align boundary maps when comparing with synoptic magnetograms
- :doc:`/api`
