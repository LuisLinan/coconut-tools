"""Utilities to read and visualize COCONUT outputs.

This module provides helpers to extract boundary data from CFmesh files,
generate 2D/3D plots, and run basic diagnostics on boundary conditions.
"""

from __future__ import annotations

# contact: Q. Noraz

from pathlib import Path
from datetime import datetime
import os
from typing import Any, Mapping, Sequence

import numpy as np
import pyvista as pv
import cmocean as cm
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from coconut_tools.create_dat import (readstruct,create_boundary_fromcfmesh)
from coconut_tools.read_dat_files import read_data
from coconut_tools.plot import Surface_2D_onetime
from coconut_tools.logger_config import setup_logger
from coconut_tools.pyvista_slice import (
    read_mesh,
    convert_units,
    convert_to_spherical,
    visualize,
)

logger = setup_logger(__name__)

# Physical constants
KB_SI = 1.380649e-23    # Boltzmann constant [J/K]
MP_SI = 1.6726219e-27  # Proton mass [kg]
MU_MEAN = 0.6           # Mean molecular weight (≈ fully ionized coronal plasma)
MU0_SI = 4.*np.pi*1.e-7 # Perméabilité du vide
Rsun = 6.9599e8

def remove_invalid_cells(mesh, field, eps=1.e-30):
    """
    Return a new mesh where cells with invalid values in `field`
    (e.g. equal to zero for MPI halo/guard cells) are removed.
    This is intended for visualization cleanup when certain
    cell-centered quantities are not defined everywhere.
    But one should be careful to filtered invalid cells as well
    if wanting to compute quantitative analysis from cell-centered qtys
    multipart in vtu.
    """
    return mesh.extract_cells(mesh.cell_data[field] > eps)

def run_coconut_reader(
    base_path: str | Path = ".",
    cfmesh_name: str = "corona.CFmesh",
    vtu_relpath: str = "vtu/corona-mhd_0000.vtu",
    radii: Sequence[float] = (1.0, 21.0),
    when: datetime | str | None = None,
    ntheta: int = 180,
    nphi: int = 360,
    dr: float = 0.01,
    surface_kwargs: Mapping[str, Any] | None = None,
    slice_kwargs: Mapping[str, Any] | None = None,
    off_screen: bool = False,
    inner_bc_check: bool = False,
    outer_bc_check: bool = False,
    AlfvSurf: bool = False,
) -> None:
    """Run a complete COCONUT post-processing pipeline.

    Args:
        base_path: Path to the run directory containing the CFmesh and `vtu/`.
        cfmesh_name: CFmesh filename inside `base_path`.
        vtu_relpath: Relative path from `base_path` to the VTU file.
        radii: Radii (code units) at which to extract spherical boundaries.
        when: Timestamp for metadata. If None, use current system time.
        ntheta: Number of colatitude sampling points.
        nphi: Number of longitude sampling points.
        dr: Radial shell thickness for the boundary extraction.
        surface_kwargs: Extra kwargs forwarded to `Surface_2D_onetime`.
        slice_kwargs: Extra kwargs forwarded to `visualize`.
        off_screen: If True, enable PyVista off-screen rendering.
        inner_bc_check: If True, plot inner boundary diagnostics.
        outer_bc_check: If True, plot outer boundary diagnostics.
        AlfvSurf: If True, render a rho and Alfven surface visualization.

    Raises:
        FileNotFoundError: If the CFmesh or VTU file is missing.

    Returns:
        None. Outputs are written to `dat/` and `plots/`.
    """
    
    base_path = Path(base_path).resolve()

    # Directories
    dat_dir = base_path / "dat"
    plots_dir = base_path / "plots"
    dat_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    cfmesh_path = base_path / cfmesh_name
    vtu_path = base_path / vtu_relpath

    if not cfmesh_path.is_file():
        raise FileNotFoundError(f"CFmesh file not found: {cfmesh_path}")
    if not vtu_path.is_file():
        raise FileNotFoundError(f"VTU file not found: {vtu_path}")

    # "when" formatting for create_boundary_fromcfmesh
    if when is None:
        when_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    elif isinstance(when, datetime):
        when_str = when.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        when_str = str(when)

    # Defaults for Surface_2D_onetime
    if surface_kwargs is None:
        surface_kwargs = dict(mode="all", extended=True, showP=True)

    # ---------------------------------------------------------
    # Boundary-condition diagnostics (inner & outer)
    # ---------------------------------------------------------
    if inner_bc_check:
        _inner_bc_check_from_cfmesh(
            cfmesh_path=cfmesh_path,
            dat_dir=dat_dir,
            plots_dir=plots_dir,
            when_str=when_str,
            ntheta=ntheta,
            nphi=nphi,
            dr=dr,
        )

    if outer_bc_check:
        _outer_bc_check_from_cfmesh(
            cfmesh_path=cfmesh_path,
            dat_dir=dat_dir,
            plots_dir=plots_dir,
            when_str=when_str,
            ntheta=ntheta,
            nphi=nphi,
            dr=dr,
        )
    
    # -----------------------
    # 1) CFmesh -> .dat + PNG for each radius
    # -----------------------
    
    
    for R in radii:
        dat_file = dat_dir / f"{R:g}Rsun.dat"
        png_file = plots_dir / f"{R:g}Rsun.png"
        
        # -----------------------------------------------------------
        # Do NOT recompute .dat if it already exists
        # -----------------------------------------------------------
        if dat_file.exists():
            logger.info("%s already exists - skipping .dat computation.", dat_file)
        else:
            print(cfmesh_path)
            logger.info("Extracting boundary at R = %s into %s", R, dat_file)
            _ = create_boundary_fromcfmesh(
                inputfile=cfmesh_path,
                time=when_str,
                rad_out=R,
                nb_th=ntheta,
                nb_phi=nphi,
                eps=dr,
                output_dat=str(dat_file),
                full_output=True,
            )
        
        # -----------------------------------------------------------
        # Always regenerate the PNG (or also skip if preferred)
        # -----------------------------------------------------------
        logger.info("Making 2D surface plot -> %s", png_file)
        Surface_2D_onetime(inputfile=str(dat_file),
            outputfile=str(png_file),
            **surface_kwargs,
        )



    

    # -----------------------
    # 2) PyVista slice plots from VTU
    # -----------------------
    if off_screen:
        os.environ.setdefault("PYVISTA_OFF_SCREEN", "1")
        pv.OFF_SCREEN = True
    
    #vr vertical slice
    if slice_kwargs is None:
        slice_kwargs = dict(
            slice_normal="y",
            vr_clim=None,
            br_clim=None,
            stream_clim=None,
            rho_iso=1e-16,
            save_path=str(plots_dir / "pyvista_slice.png"),
            show=not off_screen,
        )
    else:
        # ensure we at least provide a default save_path if not given
        slice_kwargs = dict(slice_kwargs)
        slice_kwargs.setdefault(
            "save_path", str(plots_dir / "pyvista_slice.png")
        )
        slice_kwargs.setdefault("show", not off_screen)

    logger.info("Reading VTU mesh from %s", vtu_path)
    mesh = read_mesh(str(vtu_path))
    mesh = convert_units(mesh)
    mesh = convert_to_spherical(mesh)
    #print("Mesh bounds:", mesh.bounds)
    logger.info("Making PyVista slice plot -> %s", slice_kwargs["save_path"])
    visualize(mesh, **slice_kwargs)
    
    #rho horiz. slice + alfven surface
    if AlfvSurf:
        logger.info("Making Rho/AlfvSurf PyVista slice plot -> %s", slice_kwargs["save_path"])
        visualize(mesh, slice_normal="z",
            slice_plane_scalar="rho_dim",
            AlfvSurf=True,
            save_path=str(plots_dir / "pyvista_slice_Alv.png"),
            rho_iso = 0.,
            show=not off_screen)

    # -----------------------
    # 3) Extra full y-plane disk plot
    disk_png = plots_dir
    psfile= "yplane_vr_disk.png"
    
    visualize_yplane_disk(
        mesh,
        save_path=str(disk_png),
        field="vr",                   # or "br", "rho_dim", etc.
        clim=slice_kwargs.get("vr_clim", None),
        cmap="viridis",
        psfile=psfile,
        fig_size=(1920, 1920),
    )

def Quick_Vr_Viewer(
    base_path: str | Path = ".",
    field: str = "vr",
    vr_clim: tuple[float, float] | None = None,
    do_fieldlines: bool = True,
    V_name: str = "B",
    mycmap: str = "viridis",
    vtu_relpath: str = "vtu/corona-mhd_0000.vtu",
    figpath: str | Path = "./",
    psfile: str | None = None,
    plane: str = "y",         
    phi: float = 0.0,         
    phi_degrees: bool = True, 
    cam_dist: float = 100.0,
    clean_invalid: bool = False,
) -> None:
    """Quick viewer for a y-plane disk plot.

    Args:
        base_path: Path to the run directory containing the VTU file.
        field: Scalar field to plot.
        vr_clim: Color limits for the field.
        do_fieldlines: If True, draw field lines.
        V_name: Name of the vector array on the mesh ("B" or "V").
        mycmap: Matplotlib colormap name.
        vtu_relpath: Relative path to the VTU file.
        figpath: Output directory for saved figures.
        psfile: If set, save the plot to this filename.
        plane: "x", "y", "z", or "lon"
               NB: plane="y", equivalent to plane="lon"+phi=0.
        phi: longitude value (deg or rad), used if plane=="lon"
        phi_degrees: interpret phi as degrees if True
        cam_dist: camera distance (your current 100)

    Raises:
        FileNotFoundError: If the VTU file is missing.

    Returns:
        None.
    """
    
    base_path = Path(base_path).resolve()

    vtu_path = base_path / vtu_relpath
    
    if not vtu_path.is_file():
        raise FileNotFoundError(f"VTU file not found: {vtu_path}")
    
    # -----------------------
    # PyVista slice plots from VTU
    
    if psfile is not None: #then save instead of plotting
        os.environ.setdefault("PYVISTA_OFF_SCREEN", "1")
        pv.OFF_SCREEN = True
    
    logger.info("Reading VTU mesh from %s", vtu_path)
    mesh = read_mesh(str(vtu_path))
    mesh = convert_units(mesh)
    mesh = convert_to_spherical(mesh)
    
    visualize_yplane_disk(
        mesh,
        save_path=figpath,
        field=field,                   # or "br", "rho_dim", etc.
        clim=vr_clim,
        cmap=mycmap,
        do_fieldlines=do_fieldlines,
        V_name = V_name,
        psfile=psfile,
        fig_size=(1920, 1920),
        plane=plane,
        phi=phi,
        phi_degrees=phi_degrees,
        cam_dist=cam_dist,
        clean_invalid=clean_invalid
    )



def Quick_Ra_viewer(
    base_path: str | Path = ".",
    vtu_relpath: str = "vtu/corona-mhd_0000.vtu",
    volumic_vr: np.ndarray | None = None,
    off_screen: bool = False,
) -> None:
    """Create PyVista 3D visualizations from a VTU file.

    Args:
        base_path: Path to the run directory containing the `vtu/` folder.
        vtu_relpath: Relative path from `base_path` to the VTU file.
        volumic_vr: Optional 3D array for volumetric vr rendering.
        off_screen: If True, enable PyVista off-screen rendering.

    Raises:
        FileNotFoundError: If the VTU file is missing.

    Returns:
        None.
    """
    
    base_path = Path(base_path).resolve()

    # Directories
    plots_dir = base_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    vtu_path = base_path / vtu_relpath
    
    if not vtu_path.is_file():
        raise FileNotFoundError(f"VTU file not found: {vtu_path}")
    
    # -----------------------
    # PyVista slice plots from VTU
    #if off_screen:
    #    os.environ.setdefault("PYVISTA_OFF_SCREEN", "1")
    #    pv.OFF_SCREEN = True
    
    logger.info("Reading VTU mesh from %s", vtu_path)
    mesh = read_mesh(str(vtu_path))
    mesh = convert_units(mesh)
    mesh = convert_to_spherical(mesh)
    
    #rho horiz. slice + alfven surface
    save_path=str(plots_dir / "pyvista_slice_Alv.png")
    logger.info("Making Rho/AlfvSurf PyVista slice plot -> %s", save_path)
    visualize(mesh, slice_normal="z",
               slice_plane_scalar="rho_dim",
               AlfvSurf=True,
               volumic_vr=volumic_vr,
               save_path=save_path,
               rho_iso = 0.,
               show=not off_screen)

def _add_carrington_grid(
    plotter: pv.Plotter,
    r: float,
    lon_step: float,
    lat_step: float,
    color: str,
    width: float,
):
    '''
    Developed for visualize_spherical_surface_from_vtu
    '''
    # Longitudes (meridians)
    lons = np.arange(0.0, 360.0, lon_step)
    lat = np.linspace(-90.0, 90.0, 361)

    for lon in lons:
        phi = np.deg2rad(lon)
        lam = np.deg2rad(lat)

        x = r * np.cos(lam) * np.cos(phi)
        y = r * np.cos(lam) * np.sin(phi)
        z = r * np.sin(lam)

        pts = np.column_stack((x, y, z))
        line = pv.lines_from_points(pts)
        plotter.add_mesh(line, color=color, line_width=width)

    # Latitudes (parallels)
    lats = np.arange(-90.0 + lat_step, 90.0, lat_step)
    lon = np.linspace(0.0, 360.0, 721)

    for lat in lats:
        lam = np.deg2rad(lat)
        phi = np.deg2rad(lon)

        x = r * np.cos(lam) * np.cos(phi)
        y = r * np.cos(lam) * np.sin(phi)
        z = r * np.sin(lam) * np.ones_like(phi)

        pts = np.column_stack((x, y, z))
        line = pv.lines_from_points(pts)
        plotter.add_mesh(line, color=color, line_width=width)

    
def visualize_spherical_surface_from_vtu(
    vtu_path: str | Path,
    r_surf: float,
    save_path: str | Path = "plots/",
    field: str = "vr",
    clim: tuple[float, float] | None = None,
    cmap: str = "viridis",
    psfile: str | None = None,
    fig_size: tuple[int, int] = (1920, 1920),
    discrete: bool = True,
    theta_res: int = 360,
    phi_res: int = 180,
    view: str = "iso",        # "x", "y", "z", or "iso"
    show: bool = True,
    show_grid: bool = True,
    lon_step: float = 30.0,   # degrees
    lat_step: float = 15.0,   # degrees
    grid_color: str = "white",
    grid_width: float = 1.0,
    cam_dist=None,
    clean_invalid: bool=False,

) -> None:
    """
    Plot a scalar quantity on a spherical surface at radius r_surf
    by sampling a VTU volume mesh.

    Args:
        vtu_path: Path to the VTU file.
        r_surf: Radius of the spherical surface (Rsun units).
        save_path: Output directory or full file path.
        field: Scalar field to visualize.
        clim: Color limits.
        cmap: Matplotlib colormap name.
        psfile: If set, save a screenshot with this filename.
        fig_size: Window size in pixels.
        discrete: If True, use a discretized colormap.
        theta_res: Longitude resolution of the sphere.
        phi_res: Colatitude resolution of the sphere.
        view: Camera view ("x", "y", "z", or "iso").
        show: Whether to display the plot interactively.
        show_grid: show carrington grid if True + args

    Returns:
        None.
    """

    if clean_invalid:
        tmp = remove_invalid_cells(mesh, field)
        mesh = tmp

    vtu_path = Path(vtu_path)
    if not vtu_path.is_file():
        raise FileNotFoundError(f"VTU file not found: {vtu_path}")

    logger.info("Reading VTU mesh from %s", vtu_path)

    # ------------------------------------------------------------------
    # 1) Read and prepare mesh (same pipeline as elsewhere in reader.py)
    # ------------------------------------------------------------------
    mesh = read_mesh(str(vtu_path))
    mesh = convert_units(mesh)
    mesh = convert_to_spherical(mesh)

    # Ensure the field is available as point data for sampling
    if field not in mesh.point_data and field in mesh.cell_data:
        mesh = mesh.cell_data_to_point_data()

    if field not in mesh.point_data:
        raise ValueError(
            f"Field '{field}' not found in VTU mesh. "
            f"Available point_data: {list(mesh.point_data.keys())}"
        )

    # ------------------------------------------------------------------
    # 2) Build spherical surface and sample the volume mesh onto it
    # ------------------------------------------------------------------
    sphere = pv.Sphere(
        radius=float(r_surf),
        center=(0.0, 0.0, 0.0),
        theta_resolution=int(theta_res),
        phi_resolution=int(phi_res),
    )

    surf = sphere.sample(mesh)

    # ------------------------------------------------------------------
    # 3) Plot
    # ------------------------------------------------------------------
    off_screen = psfile is not None
    p = pv.Plotter(off_screen=off_screen, window_size=fig_size)

    if discrete:
        levels = 16
        base = plt.get_cmap(cmap)
        mycmap = ListedColormap(base(np.linspace(0, 1, levels)))
    else:
        mycmap = cmap

    p.add_mesh(
        surf,
        scalars=field,
        cmap=mycmap,
        clim=clim,
        show_edges=False,
    )

    if show_grid:
        _add_carrington_grid(
            plotter=p,
            r=float(r_surf),
            lon_step=lon_step,
            lat_step=lat_step,
            color=grid_color,
            width=grid_width,
        )

    # Camera presets
    if cam_dist==None: cam_dist = 3.0 * float(r_surf)
    v = view.lower()
    if v == "x":
        eye, up = (cam_dist, 0.0, 0.0), (0.0, 0.0, 1.0)
    elif v == "y":
        eye, up = (0.0, cam_dist, 0.0), (0.0, 0.0, 1.0)
    elif v == "z":
        eye, up = (0.0, 0.0, cam_dist), (0.0, 1.0, 0.0)
    else:  # "iso"
        eye, up = (cam_dist, cam_dist, cam_dist), (0.0, 0.0, 1.0)

    p.camera_position = [eye, (0.0, 0.0, 0.0), up]

    out = str(Path(save_path) / psfile) if psfile is not None else None
    if out is not None:
        logger.info("Saving spherical surface plot -> %s", out)
        p.screenshot(out)

    if show:
        p.show()

    p.close()


def visualize_yplane_disk(
    mesh: pv.DataSet,
    save_path: str | Path = "plots/",
    field: str = "vr",
    clim: tuple[float, float] | None = None,
    cmap: str = "viridis",
    psfile: str | None = None,
    fig_size: tuple[int, int] = (1920, 1920),
    discrete: bool = True,
    do_fieldlines: bool = True,
    V_name: str = "B",
    r_seed: float = 1.0,
    n_seeds: int = 200,
    max_steps: int = 4000,
    step_size: float = 0.02,
    max_time: float = 200.0,
    line_radius: float = 0.01,
    plane: str = "y",         
    phi: float = 0.0,         
    phi_degrees: bool = True, 
    cam_dist: float = 100.0,
    clean_invalid: bool = False,
) -> None:
    """Produce a centered, full-disk 2D visualization of the y=0 plane.

    Args:
        mesh: Converted and spherical mesh.
        save_path: Output PNG directory or file path.
        field: Scalar to visualize on the disk.
        clim: Color limits for the scalar.
        cmap: Colormap name.
        psfile: If set, save a screenshot to this filename.
        fig_size: Window size in pixels (width, height).
        discrete: If True, discretize the colormap.
        do_fieldlines: If True, compute and plot field lines.
        V_name: Name of the vector array on the mesh ("B" or "V").
        r_seed: Seed radius in Rsun units.
        n_seeds: Number of seed points around the circle.
        max_steps: Max steps for field line computation.
        step_size: Step size for integration (Rsun units).
        max_time: Max integration time (controls arc length).
        line_radius: Field line tube radius.
        plane: "x", "y", "z", or "lon"
               NB: plane="y", equivalent to plane="lon"+phi=0.
        phi: longitude value (deg or rad), used if plane=="lon"
        phi_degrees: interpret phi as degrees if True
        cam_dist: camera distance (your current 100)

    Returns:
        None.
    """

    if clean_invalid:
        tmp = remove_invalid_cells(mesh, field)
        mesh = tmp

    logger.info("Creating centered y-plane disk plot -> %s", save_path)
    
    # 1) choose slice plane normal
    plane_l = plane.lower()
    
    if plane_l in ("x", "y", "z"):
        normal = plane_l  # PyVista accepts "x"/"y"/"z" directly
        slice_plane = mesh.slice(normal=normal, origin=(0.0, 0.0, 0.0))
        
        # camera: look from +axis direction, keep +z as "up" when possible
        if plane_l == "x":
            eye = (cam_dist, 0.0, 0.0)
            up  = (0.0, 0.0, 1.0)
            n = np.array([1.0, 0.0, 0.0])
        elif plane_l == "y":
            eye = (0.0, cam_dist, 0.0)
            up  = (0.0, 0.0, 1.0)
            n = np.array([0.0, 1.0, 0.0])
        else:  # "z"
            eye = (0.0, 0.0, cam_dist)
            up  = (0.0, 1.0, 0.0)   # avoid degeneracy (up parallel to view dir)
            n = np.array([0.0, 0.0, 1.0])
            
    elif plane_l in ("lon", "phi", "longitude"):
        # longitude slice: plane contains z-axis, rotated around z by phi
        phi_rad = np.deg2rad(phi) if phi_degrees else float(phi)
        
        # plane normal for longitude phi (phi=0 -> y=0 plane; phi=90deg -> x=0 plane)
        n = np.array([np.sin(phi_rad), -np.cos(phi_rad), 0.0])
        slice_plane = mesh.slice(normal=n, origin=(0.0, 0.0, 0.0))
        
        # camera: look perpendicular to that longitude plane (rotate like the slice)
        eye = tuple((-cam_dist * n).tolist()) # face-on to the slice, ie. the other way to the plane normal
        up  = (0.0, 0.0, 1.0)
        
    else:
        raise ValueError('plane must be one of {"x","y","z","lon"} (or "phi"/"longitude").')

    
    #2) Make a top-down plotter
    if psfile != None:
        off_screen=True
    else:
        off_screen=False
    p = pv.Plotter(off_screen=off_screen, window_size=fig_size)

    levels = 16
    base = plt.get_cmap(cmap)
    cmap_discrete = ListedColormap(base(np.linspace(0, 1, levels)))
    
    p.add_mesh(
        slice_plane,
        scalars=field,
        cmap=cmap_discrete,
        clim=clim,
        show_edges=False,
    )

    if do_fieldlines:
        # 1) Build an orthonormal basis (e1, e2) spanning the plane perpendicular to n, to get the seeds in there
        n_hat = n / np.linalg.norm(n)
        
        # pick a helper vector not parallel to n_hat
        a = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(a, n_hat)) > 0.99:
            a = np.array([0.0, 1.0, 0.0])
            
        e1 = np.cross(n_hat, a)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n_hat, e1)  # already normalized if n_hat and e1 are
        
        # seed circle of radius r_seed in the slice plane, centered at origin
        t = np.linspace(0.0, 2*np.pi, n_seeds, endpoint=False)
        seeds = r_seed * (np.cos(t)[:, None] * e1[None, :] + np.sin(t)[:, None] * e2[None, :])
        seed_src = pv.PolyData(seeds)
        
        
        # 2) compute streamlines (field lines) in the *full* 3D mesh, seeded from that circle
        #    (use mesh.streamlines_from_source; it uses the active vectors or vectors=...)
        # --------------------------------------------------------------------------
        # Magnetic-field streamlines
        logger.info("Adding magnetic field streamlines.")
        if V_name == "B":
            mesh["B"] = np.column_stack([mesh["bx_dim"], mesh["by_dim"], mesh["bz_dim"]])
        elif V_name == "V":
            mesh["V"] = np.column_stack([mesh["vx_dim"], mesh["vy_dim"], mesh["vz_dim"]])
        lines = mesh.streamlines_from_source(
            source=seed_src,
            vectors=V_name,
            integrator_type=45,      # Runge-Kutta 4/5
            initial_step_length=step_size,
            max_time=max_time,
            max_steps=max_steps,
            compute_vorticity=False,
        )
        
        ## 3) Shows only the field lines near the slice
        #slab_thickness = 0.05  # Rsun units; tune
        #lines_clipped = lines.clip(normal=tuple(n_hat), origin=(0.0, 0.0, 0.0), invert=False)
        #lines_clipped = lines_clipped.clip(normal=tuple(-n_hat), origin=tuple((slab_thickness*n_hat)), invert=False)
        #lines = lines_clipped
        
        # optional: make them prettier
        tubes = lines.tube(radius=line_radius)  # radius in Rsun units; tune this
        
        # add to plot
        p.add_mesh(tubes, color="white", opacity=1.0)
        # show the seed circle
        #p.add_mesh(seed_src, color="white", point_size=6, render_points_as_spheres=True)
        
    # Camera looking straight at the disk from +y direction
    #p.camera_position = [
    #    (0, 100, 0),   # eye
    #    (0, 0, 0),    # look center
    #    (0, 0, 1),    # up vector
    #]

    # 3) set camera
    p.camera_position = [eye, (0.0, 0.0, 0.0), up]
    p.camera.zoom(1.0)

    if psfile != None:
        out = str(Path(save_path) / psfile)
        p.screenshot(out)
        
    else:
        p.show()

    p.close()


def _get_mesh_radii_from_cfmesh(cfmesh_path: str | Path) -> tuple[float, float]:
    """Read the CFmesh and return (r_min, r_max) in code units.

    Args:
        cfmesh_path: Path to a CFmesh file.

    Returns:
        Tuple of (r_min, r_max).
    """
    
    if not os.path.isfile(cfmesh_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), cfmesh_path)

    with open(cfmesh_path, "r") as f:
        lines = f.readlines()
    
    idx0, idx1, idx2, idx3, nbelements, nend, comment = readstruct(lines)
    # Read connectivity and coordinates exactly like in create_dat.py
    connectivity = np.loadtxt(lines[idx0:idx0 + nbelements], dtype=int)
    coordinates = np.loadtxt(lines[idx1:idx2 - 1], dtype=float)
    # First 6 node indices for each element
    nodes = connectivity[:, :6]
    # Cell centers = mean of the 6 node positions
    centers = coordinates[nodes].mean(axis=1)
    # Radial distance of each center
    r = np.linalg.norm(centers, axis=1)

    # return min & max
    return float(r.min()), float(r.max())

def _inner_bc_check_from_cfmesh(
    cfmesh_path: str | Path,
    dat_dir: Path,
    plots_dir: Path,
    when_str: str,
    ntheta: int,
    nphi: int,
    dr: float,
    show: bool = True,
) -> None:
    """Plot inner boundary conditions at the minimum mesh radius.

    Args:
        cfmesh_path: Path to the CFmesh file.
        dat_dir: Directory for .dat outputs.
        plots_dir: Directory for plots.
        when_str: Timestamp string for metadata.
        ntheta: Number of colatitude samples.
        nphi: Number of longitude samples.
        dr: Radial shell thickness.
        show: If True, show the plot interactively.

    Returns:
        None.
    """
    rmin, _ = _get_mesh_radii_from_cfmesh(cfmesh_path)
    R = rmin

    dat_path = dat_dir / f"{R:g}Rsun.dat"

    # Ensure .dat exists at the inner radius
    if not dat_path.exists():
        logger.info("Inner BC: creating %s at R = %s", dat_path, f"{R:g}")
        create_boundary_fromcfmesh(
            inputfile=str(cfmesh_path),
            time=when_str,
            rad_out=R,
            nb_th=ntheta,
            nb_phi=nphi,
            eps=dr,
            output_dat=str(dat_path),
            full_output=True,
        )
    else:
        logger.info("Inner BC: using existing %s", dat_path)

    clat_ticks = [0, 45, 90, 135, 180]
    lon_ticks = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
    
    # Read with coconut-tools helper
    date, r_arr, clt, lon, vr, vlon, vclt, density, br, bclt, blon, temp = read_data(str(dat_path), reduced=False, extended=True)

    vars_to_plot = [
        (density.T, cm.cm.thermal, 'Density [$m^{-3}$]'),
        (temp.T, cm.cm.haline, 'Temperature [K]'),
        (br.T * 1e4, cm.cm.balance, 'Br [G]'),
        (density.T * KB_SI * temp.T, 'viridis', 'P [Pa]')
    ]

    indices = list(range(4))
    
    # Plot 2×2: rho, p, T, Br
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    for idx, plot_idx in enumerate(indices):
        row = idx % 2
        col = idx // 2
        data, cmap, label = vars_to_plot[plot_idx]
        data_shifted = np.roll(data, data.shape[1] // 2, axis=1)

        im = axs[row][col].imshow(
            data_shifted, aspect='auto', origin='lower', cmap=cmap,
            extent=[lon_ticks[0], lon_ticks[-1], clat_ticks[-1], clat_ticks[0]]
        )

        axs[row][col].set_xticks(lon_ticks)
        axs[row][col].set_yticks(clat_ticks)
        axs[row][col].invert_yaxis()
        axs[row][col].tick_params(axis='both', which='major', labelsize=12)
        if row == 3:
            axs[row][col].set_xlabel('Longitude (degrees)', fontsize=12)
        axs[row][col].set_ylabel('Colatitude (degrees)', fontsize=12)

        cbar = plt.colorbar(im, ax=axs[row][col])
        cbar.set_label(label, fontsize=12)
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.yaxis.offsetText.set_fontsize(12)

    fig.suptitle(f"Inner BC at R = {R:g} R$_\\odot$", fontsize=12)
    fig.tight_layout()
    
    out_png = plots_dir / f"{R:g}Rsun_inner_BC.png"
    logger.info("Saving inner BC map -> %s", out_png)
    fig.savefig(out_png, dpi=200)
    if show: plt.show()
    plt.close(fig)

def _outer_bc_check_from_cfmesh(
    cfmesh_path: str | Path,
    dat_dir: Path,
    plots_dir: Path,
    when_str: str,
    ntheta: int,
    nphi: int,
    dr: float,
    show: bool = True,
    gamma: float = 5.0 / 3.0,
) -> None:
    """Plot outer boundary quantities at the maximum mesh radius.

    Args:
        cfmesh_path: Path to the CFmesh file.
        dat_dir: Directory for .dat outputs.
        plots_dir: Directory for plots.
        when_str: Timestamp string for metadata.
        ntheta: Number of colatitude samples.
        nphi: Number of longitude samples.
        dr: Radial shell thickness.
        show: If True, show the plot interactively.
        gamma: Adiabatic index.

    Returns:
        None.
    """
    _, rmax = _get_mesh_radii_from_cfmesh(cfmesh_path)
    R = rmax

    dat_path = dat_dir / f"{R:g}Rsun.dat"

    # Ensure .dat exists at the outer radius
    if not dat_path.exists():
        logger.info("Outer BC: creating %s at R = %s", dat_path, f"{R:g}")
        create_boundary_fromcfmesh(
            inputfile=str(cfmesh_path),
            time=when_str,
            rad_out=R,
            nb_th=ntheta,
            nb_phi=nphi,
            eps=dr,
            output_dat=str(dat_path),
            full_output=True,
        )
    else:
        logger.info("Outer BC: using existing %s", dat_path)


    clat_ticks = [0, 45, 90, 135, 180]
    lon_ticks = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
        
    # Read with coconut-tools helper
    date, r_arr, clt, lon, vr, vlon, vclt, density, br, bclt, blon, temp = read_data(str(dat_path), reduced=False, extended=True)
    
    n = density.T # m-3 -> mu*n here
    T   = temp.T # K
    Br   = br.T # T
    Bclt = bclt.T # T
    Blon = blon.T # T
    Bmag = np.sqrt(Br**2 + Bclt**2 + Blon**2) # T
    Vr = vr.T # km/s
    Vclt = vclt.T # km/s
    Vlon = vlon.T # km/s
    Vmag = np.sqrt(Vr**2 + Vclt**2 + Vlon**2) # T
    
    # Gas pressure from ideal gas law
    p = n * KB_SI * T # Pa -> mu*n here, p may be underestimated
    
    # Sound speed and Alfvén speed in cgs
    cs = np.sqrt(gamma * KB_SI * T / MU_MEAN /MP_SI ) #m/s #careful here as T here consider
    vA = Bmag / np.sqrt(MU0_SI * MP_SI * n) #m/s -> MU_MEAN * MP_SI * n = rho [kg/m^3] #again n=mu*n with here (see create_dat.py) so no need for MU_MEAN
    
    eps = 1e-30
    M  = Vmag / np.maximum(cs/1000., eps)                      # sonic Mach
    MA = Vmag / np.maximum(vA/1000., eps)                      # Alfvénic Mach

    vars_to_plot = [
        (cs/1000., 'plasma', r"Sound speed $c_s$ [km/s]"),
        (vA/1000., 'plasma', r"Alfvén speed $v_A$ [km/s]"),
        (M,  'viridis', r"Mach number $Ma$"),
        (MA, 'viridis',r"Alfvénic Mach $M_A$")
    ]

    indices = list(range(4))
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    for idx, plot_idx in enumerate(indices):
        row = idx % 2
        col = idx // 2
        data, cmap, label = vars_to_plot[plot_idx]
        data_shifted = np.roll(data, data.shape[1] // 2, axis=1)

        im = axs[row][col].imshow(
            data_shifted, aspect='auto', origin='lower', cmap=cmap,
            extent=[lon_ticks[0], lon_ticks[-1], clat_ticks[-1], clat_ticks[0]]
        )

        axs[row][col].set_xticks(lon_ticks)
        axs[row][col].set_yticks(clat_ticks)
        axs[row][col].invert_yaxis()
        axs[row][col].tick_params(axis='both', which='major', labelsize=12)
        if row == 3:
            axs[row][col].set_xlabel('Longitude (degrees)', fontsize=12)
        axs[row][col].set_ylabel('Colatitude (degrees)', fontsize=12)

        cbar = plt.colorbar(im, ax=axs[row][col])
        cbar.set_label(label, fontsize=12)
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.yaxis.offsetText.set_fontsize(12)

    fig.suptitle(f"Outer BC at R = {R:g} R$_\\odot$", fontsize=12)
    fig.tight_layout()

    out_png = plots_dir / f"{R:g}Rsun_outer_BC.png"
    logger.info("Saving outer BC map -> %s", out_png)
    fig.savefig(out_png, dpi=200)
    if show: plt.show()
    plt.close(fig)

if __name__ == "__main__":
    vtu_path = Path("C:/Users/luisl/Documents/Travail/Article_COCORIA/corona-mhd_0.vtu")
    cfmesh_path = Path("C:/Users/luisl/Documents/Travail/Article_COCORIA/corona.CFmesh")
    output_dir = Path("E:/euhforia/image")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dat").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYVISTA_OFF_SCREEN", "1")
    pv.OFF_SCREEN = True

    when_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    logger.info("Testing run_coconut_reader.")
    run_coconut_reader(
        base_path=output_dir,
        cfmesh_name=str(cfmesh_path),
        vtu_relpath=str(vtu_path),
        radii=(1.0,),
        when=when_str,
        ntheta=90,
        nphi=180,
        dr=0.01,
        surface_kwargs=dict(mode="all", extended=True, showP=True),
        slice_kwargs=dict(
            slice_normal="y",
            vr_clim=None,
            br_clim=None,
            stream_clim=None,
            rho_iso=1e-16,
            save_path=str(output_dir / "pyvista_slice.png"),
            show=False,
        ),
        off_screen=True,
        inner_bc_check=False,
        outer_bc_check=False,
        AlfvSurf=False,
    )

    logger.info("Testing Quick_Vr_Viewer.")
    Quick_Vr_Viewer(
        base_path=vtu_path.parent,
        vtu_relpath=vtu_path.name,
        figpath=output_dir,
        psfile="quick_vr.png",
        do_fieldlines=False,
    )

    logger.info("Testing Quick_Ra_viewer.")
    Quick_Ra_viewer(
        base_path=vtu_path.parent,
        vtu_relpath=vtu_path.name,
        volumic_vr=None,
        off_screen=True,
    )

    logger.info("Testing visualize_yplane_disk.")
    mesh = read_mesh(str(vtu_path))
    mesh = convert_units(mesh)
    mesh = convert_to_spherical(mesh)
    visualize_yplane_disk(
        mesh,
        save_path=output_dir,
        field="vr",
        clim=None,
        cmap="viridis",
        psfile="yplane_disk.png",
        fig_size=(1920, 1920),
        do_fieldlines=False,
    )

    logger.info("Testing _get_mesh_radii_from_cfmesh.")
    rmin, rmax = _get_mesh_radii_from_cfmesh(cfmesh_path)
    logger.info("Mesh radii: rmin=%s rmax=%s", rmin, rmax)

    logger.info("Testing _inner_bc_check_from_cfmesh.")
    _inner_bc_check_from_cfmesh(
        cfmesh_path=cfmesh_path,
        dat_dir=output_dir / "dat",
        plots_dir=output_dir / "plots",
        when_str=when_str,
        ntheta=90,
        nphi=180,
        dr=0.01,
        show=False,
    )

    logger.info("Testing _outer_bc_check_from_cfmesh.")
    _outer_bc_check_from_cfmesh(
        cfmesh_path=cfmesh_path,
        dat_dir=output_dir / "dat",
        plots_dir=output_dir / "plots",
        when_str=when_str,
        ntheta=90,
        nphi=180,
        dr=0.01,
        show=False,
    )
