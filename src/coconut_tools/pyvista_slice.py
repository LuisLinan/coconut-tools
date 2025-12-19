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

Author: Luis, Quentin
"""

import numpy as np
import pyvista as pv
import logging
import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
    mesh['x'], mesh['y'], mesh['z'] = pts[:,0], pts[:,1], pts[:,2] # m
    mesh['rho_dim'] = mesh['rho'] * 1.67e-16 # g/cm3
    mesh['prs_dim'] = mesh['p'] * 0.3851 # Ba = dyn/cm2
    mesh['temp'] = 7.7e-9 * mesh['prs_dim'] / mesh['rho_dim'] # K
    # T = mu*mp/kb * P / rho -> mu=0.64 hardcoded here !
    
    v = mesh['v'] * 480.24838  # km/s
    mesh['vx_dim'], mesh['vy_dim'], mesh['vz_dim'] = v[:,0], v[:,1], v[:,2]
    mesh['bx_dim'] = mesh['Bx'] * 2.2 # G
    mesh['by_dim'] = mesh['By'] * 2.2 # G
    mesh['bz_dim'] = mesh['Bz'] * 2.2 # G
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

def visualizeQ(
    mesh,
    slice_normal: str = "y",
    slice_plane_scalar: str = "vr",
    save_path: str | None = None,
    show: bool = True,
    vr_clim: tuple[float, float] | None = None,
    br_clim: tuple[float, float] | None = None,
    stream_clim: tuple[float, float] | None = None,
    rho_iso: float = 1e-13,
    AlfvSurf: bool = False,
    volumic_vr: bool = False,
    # ---- Camera parameters ----
    camera_radius: float = 18.0,
    camera_phi_deg: float = 60.0,
    camera_z: float = 4.0,
):
    """Create a PyVista visualization of the MHD quantities, with tunable colorbars,
    density isosurface level, and camera position.
    contact: Q. Noraz
    
    Parameters
    ----------
    mesh : pv.DataSet
        The mesh with physical fields.
    slice_normal : {'x','y','z'}
        Normal of the velocity slice plane.
    save_path : str or None
        Where to save the rendered figure.
    show : bool
        Show interactive viewer (True) or off-screen only (False).
    vr_clim, br_clim, stream_clim : tuple(float,float) or None
        Colorbar limits for vr, br, and streamline magnitude.
    rho_iso : float
        Density isosurface value (g/cm^3).
    camera_radius : float
        Radial distance of the camera.
    camera_phi_deg : float
        Azimuthal angle of the camera in degrees.
    camera_z : float
        Camera height above equatorial plane.
    """

    logging.info("Creating plotter...")
    off = not show
    if off:
        os.environ.setdefault("PYVISTA_OFF_SCREEN", "1")
        pv.OFF_SCREEN = True

    p = pv.Plotter(off_screen=off, window_size=[1920, 1080])

    # --------------------------------------------------------------------------
    # Camera configuration
    phi = np.radians(camera_phi_deg)
    camera_position = [
        (camera_radius * np.cos(phi), camera_radius * np.sin(phi), camera_z),
        (0.0, 0.0, 0.0),        # look at origin
        (0.0, 0.0, 1.0),        # "up" direction
    ]
    p.camera_position = camera_position

    # --------------------------------------------------------------------------
    # Slice plot: radial velocity vr
    logging.info("Adding slice plot ("+slice_plane_scalar+")…")
    slice_plane = mesh.slice(normal=slice_normal, origin=(0, 0, 0))
    if slice_plane_scalar=="vr":
        slice_plane_tit="Radial Velocity (vr) [km/s]"
        plane_log_scale=False
        slice_plane_cmap='viridis'
        opac=1.
    elif slice_plane_scalar=="rho_dim":
        slice_plane_tit=r"Density ($\rho$) [g/cm$^3$]"
        plane_log_scale=True
        slice_plane_cmap='plasma'
        # discretizes
        levels = 16
        base = plt.get_cmap(slice_plane_cmap)
        slice_plane_cmap = ListedColormap(base(np.linspace(0, 1, levels)))
        #
        opac=0.7
    else:
        slice_plane_tit=""
        plane_log_scale=False
        slice_plane_cmap='viridis'
        opac=1.
    p.add_mesh(
        slice_plane,
        scalars=slice_plane_scalar,
        cmap=slice_plane_cmap,
        clim=vr_clim,
        log_scale=plane_log_scale,
        opacity=opac,
        scalar_bar_args={
            "height": 0.25,
            "vertical": True,
            "title_font_size": 30,
            "width": 0.05,
            "title": slice_plane_tit,
            "position_x": 0.10,
            "position_y": 0.65,
        },
    )
    p.enable_depth_peeling(number_of_peels=200, occlusion_ratio=0.0) # > Log scaling compresses values > large flat regions of nearly identical color. VTK’s default alpha blending then depends strongly on triangle draw order > patchiness.

    # --------------------------------------------------------------------------
    # Alfven surface
    if AlfvSurf:
        logging.info("Adding Alfven surface…")
        va = np.sqrt(mesh['br']**2+mesh['btheta']**2+mesh['bphi']**2)/np.sqrt(4.*np.pi*mesh['rho_dim'])
        Ma = np.sqrt(mesh['vr']**2+mesh['vtheta']**2+mesh['vphi']**2)*1.e5/va
        #print(mesh.n_points, mesh.n_cells, Ma.shape, Ma.size)
        mesh["Ma"]  = Ma # .point_array ? Ma.swapaxes(-2,-1).ravel("C") ?
        AlfSurf=mesh.contour(scalars="Ma",isosurfaces=[1.])
        AlfvenSurf=p.add_mesh(AlfSurf,opacity=0.4,color='lightblue')

    if volumic_vr:
        scalar='vr'
        vlim=500.
        VrSurf=mesh.contour(scalars=scalar,isosurfaces=[vlim])
        VradialSurf=p.add_mesh(VrSurf,opacity=0.4,color='red')
        #opacity='geom'
        #logging.info("Adding volumic rendering ("+scalar+")…")
        #p.add_volume(mesh,scalars=scalar,cmap=['r','r','r'],opacity=opacity,clim=vlim)

    # --------------------------------------------------------------------------
    # Clipped inner sphere: br
    logging.info("Adding clipped sphere (br)…")
    sphere = pv.Sphere(center=(0, 0, 0), radius=1.01)
    clipped = mesh.clip_surface(sphere)

    #if br_clim is None:
    #    br_clim = (-1.0, 1.0)

    p.add_mesh(
        clipped,
        scalars="br",
        cmap="seismic",
        clim=br_clim,
        scalar_bar_args={
            "height": 0.25,
            "vertical": True,
            "title_font_size": 30,
            "width": 0.05,
            "title": "Radial Magnetic Field (br) [G]",
            "position_x": 0.85,
            "position_y": 0.65,
        },
    )

    # --------------------------------------------------------------------------
    # Magnetic-field streamlines
    logging.info("Adding magnetic field streamlines…")
    mesh["B"] = np.column_stack([mesh["bx_dim"], mesh["by_dim"], mesh["bz_dim"]])

    stream = make_streamlines(
        mesh,
        vectors="B",
        max_steps=1000,
        n_points=500,
        source_radius=2.0,
        source_center=(0, 0, 0),
    )

    stream_tube = stream.tube(radius=0.01)
    p.add_mesh(
        stream_tube,
        cmap="binary",
        clim=stream_clim,
        scalar_bar_args={
            "height": 0.25,
            "vertical": True,
            "title_font_size": 30,
            "width": 0.05,
            "title": "B-field Streamlines",
            "position_x": 0.85,
            "position_y": 0.05,
        },
    )

    # --------------------------------------------------------------------------
    # Density isosurface
    if rho_iso != 0. :
        logging.info("Adding density isosurface...")
        iso = mesh.contour([rho_iso], scalars="rho_dim")
        p.add_mesh(
            iso,
            scalars="rho_dim",
            cmap="plasma",
            opacity=0.4,
            show_scalar_bar=False,
        )
        logging.info(f"Density isosurface at rho = {rho_iso:.2e} g/cm^3")

    # --------------------------------------------------------------------------
    # Render + save
    p.show(interactive=False, auto_close=False, window_size=[1800, 900])

    if save_path:
        logging.info(f"Saving figure to {save_path}…")
        p.screenshot(save_path)

    if show:
        p.show()

    p.close()

if __name__ == '__main__':
    input_path = 'C:/Users/luisl/Documents/Travail/processing_scripts/corona-mhd.vtu'
    mesh = read_mesh(input_path)
    mesh = convert_units(mesh)
    mesh = convert_to_spherical(mesh)
    visualize(mesh, slice_normal='y', save_path='pyvista_slice.png', show=False)
