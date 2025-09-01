"""
Coconut Visualization with PyVista

This module provides tools to read, convert, and visualize MHD simulation data from
COCONUT using PyVista. The main objective is to demonstrate how to load a VTU file,
convert the data to physical units and spherical coordinates, and create multiple
visualizations (slice, clipping, streamlines, isosurfaces).

Typical visualizations include:
    - Slice of radial velocity (vr)
    - Clipped spherical region colored by radial magnetic field (br)
    - Streamlines of the magnetic field
    - Isosurface of density (optional)

Author: Luis
"""

import numpy as np
import pyvista as pv
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def _to_int(x, name: str):
    try:
        return int(x)
    except Exception as e:
        raise TypeError(f"{name} must be an integer-like value, got {x!r}") from e

def make_streamlines(mesh,
                     vectors="B",
                     max_steps=1000,
                     n_points=1000,
                     source_radius=2.0,
                     source_center=(0, 0, 0),
                     initial_step_length=None,
                     step_unit_val=None,
                     max_error=None,
                     max_step_length=None):
    """
    Wrapper that normalizes types for VTK/PyVista streamline settings.
    - Casts step-like params to int where VTK expects integers.
    - Uses `max_steps` (replacement for deprecated `max_time`).
    """
    # Ensure ints where VTK requires ints
    max_steps = _to_int(max_steps, "max_steps")
    n_points = _to_int(n_points, "n_points")

    # Build streamlines via PyVista
    stream = mesh.streamlines(
        vectors=vectors,
        max_steps=max_steps,      # must be int
        n_points=n_points,        # must be int
        source_radius=source_radius,
        source_center=source_center,
        # Note: the following kwargs are only passed if not None
        **({} if initial_step_length is None else {"initial_step_length": initial_step_length}),
        **({} if step_unit_val is None else {"integration_step_unit": step_unit_val}),
        **({} if max_error is None else {"maximum_error": max_error}),
        **({} if max_step_length is None else {"maximum_step_length": max_step_length}),
    )
    return stream

def read_mesh(filename):
    """Load a VTU mesh file.

    Args:
        filename (str): Path to the VTU file.

    Returns:
        pv.DataSet: PyVista mesh object.
    """
    logging.info('Reading file...')
    mesh = pv.read(filename)
    logging.info('Done!')
    return mesh

def convert_units(mesh):
    """Convert mesh quantities into physical units and add them to the mesh.

    Args:
        mesh (pv.DataSet): PyVista mesh.

    Returns:
        pv.DataSet: Updated mesh with new fields.
    """
    logging.info('Converting to physical units...')
    pts = mesh.points * 6.955e8  # Convert to meters
    mesh['x'], mesh['y'], mesh['z'] = pts[:,0], pts[:,1], pts[:,2]
    mesh['rho_dim'] = mesh['rho'] * 1.67e-16
    mesh['prs_dim'] = mesh['p'] * 0.3851
    mesh['temp'] = 7.7e-9 * mesh['prs_dim'] / mesh['rho_dim']

    v = mesh['v'] * 480.24838  # km/s
    mesh['vx_dim'], mesh['vy_dim'], mesh['vz_dim'] = v[:,0], v[:,1], v[:,2]
    mesh['bx_dim'] = mesh['Bx'] * 2.2
    mesh['by_dim'] = mesh['By'] * 2.2
    mesh['bz_dim'] = mesh['Bz'] * 2.2
    return mesh

def convert_to_spherical(mesh):
    """Convert Cartesian coordinates to spherical and compute radial components.

    Args:
        mesh (pv.DataSet): Mesh with Cartesian and dimensional quantities.

    Returns:
        pv.DataSet: Updated mesh with spherical quantities.
    """
    logging.info('Converting to spherical coordinates...')
    x, y, z = mesh['x'], mesh['y'], mesh['z']
    vx, vy, vz = mesh['vx_dim'], mesh['vy_dim'], mesh['vz_dim']
    bx, by, bz = mesh['bx_dim'], mesh['by_dim'], mesh['bz_dim']

    r = np.sqrt(x**2 + y**2 + z**2)
    rxy = np.sqrt(x**2 + y**2) + 1e-20

    mesh['r'] = r
    mesh['rxy'] = rxy
    mesh['vr'] = (x*vx + y*vy + z*vz) / r
    mesh['vtheta'] = (x*z*vx + y*z*vy - rxy**2*vz) / (rxy*r)
    mesh['vphi'] = (-y*vx + x*vy) / rxy
    mesh['br'] = (x*bx + y*by + z*bz) / r
    mesh['btheta'] = (x*z*bx + y*z*by - rxy**2*bz) / (rxy*r)
    mesh['bphi'] = (-y*bx + x*by) / rxy
    return mesh

def visualize(mesh, slice_normal='y', save_path=None, show=True):
    """Create a PyVista visualization of the MHD quantities.

    Args:
        mesh (pv.DataSet): The mesh with physical fields.
        slice_normal (str): Normal direction of the slice ('x', 'y', 'z').
        save_path (str | None): If given, path where the figure will be saved
            (supports .png, .jpg, .pdf, .svg).
        show (bool): If True, display the interactive scene.
    """
    logging.info('Creating plotter...')
    off = not show
    # Optionnel mais pratique en CI:
    if off:
        os.environ.setdefault("PYVISTA_OFF_SCREEN", "1")
        pv.OFF_SCREEN = True

    p = pv.Plotter(off_screen=off)

    rr = 18.0  # reculer la caméra
    phi_rad = np.radians(60)
    cpos = [(rr*np.cos(phi_rad), rr*np.sin(phi_rad), 4.0),  # vue légèrement décalée en z
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0)]

    logging.info('Adding slice plot (vr)...')
    slice_plane = mesh.slice(normal=slice_normal)
    p.add_mesh(slice_plane, scalars='vr', cmap='coolwarm', scalar_bar_args={
        'height': 0.25, 'vertical': True, 'title_font_size': 30, 'width': 0.05,
        'title': 'Radial Velocity (vr) [km/s]', 'position_x': 0.85, 'position_y': 0.05
    })

    logging.info('Adding clipped sphere (br)...')
    sphere = pv.Sphere(center=(0,0,0), radius=1.01)
    clipped = mesh.clip_surface(sphere)
    p.add_mesh(clipped, scalars='br', cmap='seismic', clim=[-1,1], scalar_bar_args={
        'height': 0.25, 'vertical': True, 'title_font_size': 30, 'width': 0.05,
        'title': 'Radial Magnetic Field (br) [G]', 'position_x': 0.88, 'position_y': 0.35
    })

    logging.info('Adding magnetic field streamlines...')
    mesh['B'] = np.column_stack((mesh['bx_dim'], mesh['by_dim'], mesh['bz_dim']))
    stream = make_streamlines(
        mesh,
        vectors="B",
        max_steps=1000,  # ensure it's an int (not 1000.0)
        n_points=1000,
        source_radius=2.0,
        source_center=(0, 0, 0),
    )
    p.add_mesh(stream.tube(radius=0.01), cmap='binary', scalar_bar_args={
        'height': 0.25, 'vertical': True, 'title_font_size': 30, 'width': 0.05,
        'title': 'B-field Streamlines', 'position_x': 0.88, 'position_y': 0.65
    })

    logging.info('Adding density isosurface...')
    iso = mesh.contour([1e-16], scalars='rho_dim')
    p.add_mesh(iso, scalars='rho_dim', cmap='plasma', opacity=0.4, scalar_bar_args={
        'height': 0.25, 'vertical': True, 'title_font_size': 30, 'width': 0.05,
        'title': 'Isosurface of Density [g/cm^3]', 'position_x': 0.10, 'position_y': 0.65
    })
    p.show(interactive=False, auto_close=False, window_size=[1800, 900])


    if save_path:
        logging.info(f'Saving figure to {save_path}...')
        p.screenshot(save_path)
    if show:
        logging.info('Displaying scene...')
        p.show()
    p.close()


if __name__ == '__main__':
    input_path = 'C:/Users/luisl/Documents/Travail/processing_scripts/corona-mhd.vtu'
    mesh = read_mesh(input_path)
    mesh = convert_units(mesh)
    mesh = convert_to_spherical(mesh)
    visualize(mesh, slice_normal='y', save_path='pyvista_slice.png', show=False)
