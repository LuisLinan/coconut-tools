# contact: Q. Noraz

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import os
import logging

import numpy as np
import pyvista as pv
import cmocean as cm
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from coconut_tools.create_dat import (readstruct,create_boundary_fromcfmesh)
from coconut_tools.read_dat_files import read_data
from coconut_tools.plot import Surface_2D_onetime
from coconut_tools.pyvista_slice import (
    read_mesh,
    convert_units,
    convert_to_spherical,
    visualizeQ,
)

# Physical constants
KB_SI = 1.380649e-23    # Boltzmann constant [J/K]
MP_SI = 1.6726219e-27  # Proton mass [kg]
MU_MEAN = 0.6           # Mean molecular weight (≈ fully ionized coronal plasma)
MU0_SI = 4.*np.pi*1.e-7 # Perméabilité du vide
Rsun = 6.9599e8

def run_coconut_reader(
    base_path=".",
    cfmesh_name="corona.CFmesh",
    vtu_relpath="vtu/corona-mhd_0000.vtu",
    radii=(1.0, 21.0),
    when=None,
    ntheta=180,
    nphi=360,
    dr=0.01,
    surface_kwargs=None,
    slice_kwargs=None,
    off_screen=False,
    inner_bc_check: bool = False,
    outer_bc_check: bool = False,
    AlfvSurf: bool = False
):
    """
    Run a complete COCONUT post-processing pipeline:
    - extract spherical boundary layers from a CFmesh file into .dat files,
    - generate 2D surface plots for these boundaries,
    - create PyVista 3D visualizations from a VTU file.

    This function is designed to be executed **inside the results directory of a
    given COCONUT run**, i.e., in a folder containing:
        - the main CFmesh solution file (e.g., corona.CFmesh)
        - a subdirectory `vtu/` containing VTU outputs
        - optional existing folders `dat/` and `plots/`

    A typical directory structure is:
        run/
          results-<name>/
             corona.CFmesh
             vtu/
                corona-mhd_0000.vtu
             dat/            (created automatically if missing)
             plots/          (created automatically if missing)

    The function can then be called with `base_path="."` to operate relative
    to the current working directory. This allows reusing the tool for any
    COCONUT run without modifying paths.
    
    --------------------------------------------------------------------------
    Parameters
    --------------------------------------------------------------------------
    
    base_path : str or Path, optional
        Path to the run directory containing the CFmesh file and the `vtu/`
        folder. All outputs (`dat/`, `plots/`) will be created inside this
        directory. Default is `"."` (current working directory).

    cfmesh_name : str, optional
        Filename of the CFmesh file to read (inside `base_path`). This file is
        used to extract physical quantities on spherical shells. Default:
        `"corona.CFmesh"`.
    
    vtu_relpath : str, optional
        Relative path from `base_path` to the VTU file used for 3D PyVista
        visualization. Default: `"vtu/corona-mhd_0000.vtu"`.
    
    radii : tuple of float, optional
        Radii (in code units) at which to extract spherical boundaries.
        For each radius R, a file `dat/<R>Rsun.dat` and a plot
        `plots/<R>Rsun.png` will be created unless the .dat file already
        exists. Default: `(1.0, 21.0)`.
    
    when : datetime, str, or None, optional
        Timestamp associated with the CFmesh extraction.
        - If None: use current system time.
        - If datetime: formatted internally as "%Y-%m-%dT%H:%M:%S".
        - If str: used as-is (must match the expected format).
        This affects only metadata inside the .dat file.
    
    ntheta : int, optional
        Number of latitudinal sampling points for boundary extraction.
        Default: 180.
    
    nphi : int, optional
        Number of longitudinal sampling points. Default: 360.
    
    dr : float, optional
        Radial shell thickness for the boundary extraction. Default: 0.01.
    
    surface_kwargs : dict or None, optional
        Additional keyword arguments forwarded to
        `coconut_tools.plot.Surface_2D_onetime()`.
        Typical examples include:
            dict(mode="all", extended=True, showP=True)
        Default: None (equivalent to dict(mode="all", extended=True, showP=True)).
    
    slice_kwargs : dict or None, optional
        Additional keyword arguments forwarded to `visualize2()` from
        `coconut_tools.pyvista_slice`. These control the PyVista 3D visualization.
        Examples:
            dict(
                slice_normal="y",
                slice_plane_scalar: str = "vr"
                AlfSurf: bool = False
                vr_clim=(-200, 300),
                br_clim=(-1.0, 1.0),
                stream_clim=(0.0, 1.0),
                rho_iso=1e-16,
                save_path="plots/slice.png",
                show=True,
            )
        If None, sensible defaults are provided, including automatically
        saving the PyVista visualization to `plots/pyvista_slice.png`.
    
    off_screen : bool, optional
        If True, PyVista runs in OFF_SCREEN mode (no GUI window). This is
        strongly recommended for batch or remote environments (HPC clusters).
        Default: True.

    inner_bc_check : bool
        If True, plot inner boundary conditions (ρ, p, T, |B|) at the smallest radius.
    outer_bc_check : bool
        If True, plot outer boundary conditions (c_s, v_A, Mach, M_A) at the largest radius.
    
    AlfvSurf : bool
        If True, plot a 3D render with rho horiz. slice and alfven surface
    
    --------------------------------------------------------------------------
    Behavior
    --------------------------------------------------------------------------

    - The function creates `dat/` and `plots/` inside `base_path` if missing.

    - For each radius in `radii`:
        > If `<R>Rsun.dat` already exists → it will NOT be recomputed  
        > The PNG surface plot for that radius WILL be regenerated
          (unless explicitly modified in the code).

    - The PyVista visualization is always saved to the path specified in
      `slice_kwargs["save_path"]` and may or may not appear interactively
      depending on `slice_kwargs["show"]` and `off_screen`.

    - All paths are constructed relative to `base_path`, allowing seamless
      reuse across different runs with identical folder structure.

    --------------------------------------------------------------------------
    Returns
    --------------------------------------------------------------------------

    None. The function writes output files to `dat/` and `plots/` and logs
    progress messages. No data structure is returned.

    --------------------------------------------------------------------------
    Example
    --------------------------------------------------------------------------

    # Inside the run directory (.../results-mycase/)
    run_coconut_reader(
        cfmesh_name="corona.CFmesh",
        vtu_relpath="vtu/corona-mhd_0000.vtu",
        radii=(1.0, 21.0, 30.0),
        when="2024-04-09T05:04:00",
        slice_kwargs=dict(
            slice_normal="y",
            vr_clim=(-200, 300),
            show=False,
        ),
    )
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
    	    print(f"[INFO] {dat_file} already exists — skipping .dat computation.")
    	else:
    	    print(f"[INFO] Extracting boundary at R = {R} into {dat_file}")
    	    _ = create_boundary_fromcfmesh(
    	        inputfile=str(cfmesh_path),
    	        time=when_str,
    	        rad_out=R*Rsun,
    	        nb_th=ntheta,
    	        nb_phi=nphi,
    	        eps=dr,
    	        output_dat=str(dat_file),
    	        full_output=True,
    	    )
    	
    	# -----------------------------------------------------------
    	# Always regenerate the PNG (or also skip if preferred)
    	# -----------------------------------------------------------
    	print(f"[INFO] Making 2D surface plot -> {png_file}")
    	Surface_2D_onetime(
    	    inputfile=str(dat_file),
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

    print(f"[INFO] Reading VTU mesh from {vtu_path}")
    mesh = read_mesh(str(vtu_path))
    mesh = convert_units(mesh)
    mesh = convert_to_spherical(mesh)
    #print("Mesh bounds:", mesh.bounds)
    print(f"[INFO] Making PyVista slice plot -> {slice_kwargs['save_path']}")
    visualizeQ(mesh, **slice_kwargs)
    
    #rho horiz. slice + alfven surface
    if AlfvSurf:
        print(f"[INFO] Making Rho/AlfvSurf PyVista slice plot -> {slice_kwargs['save_path']}")
        visualizeQ(mesh, slice_normal="z",
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
    base_path=".",
    field = "vr",
    vr_clim = None,
    do_fieldlines=True,
    V_name = "B",
    mycmap = "viridis",
    vtu_relpath="vtu/corona-mhd_0000.vtu",
    figpath="./",
    psfile=None,
):
    """
    base_path : str or Path, optional
        Path to the run directory containing the CFmesh file and the `vtu/`
        folder. All outputs (`dat/`, `plots/`) will be created inside this
        directory. Default is `"."` (current working directory).

    field : str
        field to plot
    
    psfile : str or None
        save the plot as "str" in figpath
    """
    
    base_path = Path(base_path).resolve()

    vtu_path = base_path / vtu_relpath
    
    if not vtu_path.is_file():
        raise FileNotFoundError(f"VTU file not found: {vtu_path}")
    
    # -----------------------
    # PyVista slice plots from VTU
    
    if psfile!=None: #then save instead of plotting
        os.environ.setdefault("PYVISTA_OFF_SCREEN", "1")
        pv.OFF_SCREEN = True
    
    print(f"[INFO] Reading VTU mesh from {vtu_path}")
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
    )

def Quick_Ra_viewer(
    base_path=".",
    vtu_relpath="vtu/corona-mhd_0000.vtu",
    volumic_vr=False,
    off_screen=False,
):
    """
    Create PyVista 3D visualizations from a VTU file:
    - mag. surface map
    - field line tracing
    - density equatorial plane
    - alfven's surface
    
    This function is designed to be executed **inside the results directory of a
    given COCONUT run**, i.e., in a folder containing:
        - a subdirectory `vtu/` containing VTU outputs
        - `plots/`
    
    A typical directory structure is:
        run/
          results-<name>/
             vtu/
                corona-mhd_0000.vtu
             plots/          (created automatically if missing)
    
    The function can then be called with `base_path="."` to operate relative
    to the current working directory. This allows reusing the tool for any
    COCONUT run without modifying paths.
    
    --------------------------------------------------------------------------
    Parameters
    --------------------------------------------------------------------------
    
    base_path : str or Path, optional
        Path to the run directory containing the CFmesh file and the `vtu/`
        folder. All outputs (`dat/`, `plots/`) will be created inside this
        directory. Default is `"."` (current working directory)
    
    vtu_relpath : str, optional
        Relative path from `base_path` to the VTU file used for 3D PyVista
        visualization. Default: `"vtu/corona-mhd_0000.vtu"`.
    
    off_screen : bool, optional
        If True, PyVista runs in OFF_SCREEN mode (no GUI window). This is
        strongly recommended for batch or remote environments (HPC clusters).
        Default: True.
    
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
    
    print(f"[INFO] Reading VTU mesh from {vtu_path}")
    mesh = read_mesh(str(vtu_path))
    mesh = convert_units(mesh)
    mesh = convert_to_spherical(mesh)
    
    #rho horiz. slice + alfven surface
    save_path=str(plots_dir / "pyvista_slice_Alv.png")
    print(f"[INFO] Making Rho/AlfvSurf PyVista slice plot -> {save_path}")
    visualizeQ(mesh, slice_normal="z",
               slice_plane_scalar="rho_dim",
               AlfvSurf=True,
               volumic_vr=volumic_vr,
               save_path=save_path,
               rho_iso = 0.,
               show=not off_screen)

def visualize_yplane_disk(
    mesh,
    save_path="plots/",
    field="vr",               # which scalar field to plot
    clim=None,                # color limits
    cmap="viridis",
    psfile=None,
    fig_size=(1920, 1920),
    discrete=True,
    do_fieldlines=True,
    V_name = "B",
    r_seed = 1.0,
    n_seeds = 200,
    max_steps = 4000,
    step_size = 0.02,
    max_time = 200.0,
    line_radius=0.01,
):
    """
    Produce a centered, full-disk 2D visualization of the y=0 plane.

    Parameters
    ----------
    mesh : pv.DataSet
        Converted and spherical mesh.
    save_path : str
        Output PNG file path.
    field : str
        Scalar to visualize on the disk (e.g. 'vr', 'rho_dim', 'br', ...)
    clim : tuple or None
        Color limits for the scalar.
    cmap : str
        Colormap name.
    psfile : str
        Whether to save an interactive PyVista window or show it only if None.
    fig_size : tuple
        Window size in pixels (width, height).
    discrete: bool
        discretises mpl cmap if True
    V_name: str
        name of the vector array on the mesh,
        currently coded: "B" and "V"
    r_seed: flt
        1 Rsun (or in your mesh units)
    n_seeds: int
        number of seed points around the circle
    max_steps: int
        number of step max for field line computation
    step_size: flt
        in Rsun units (adjust to your grid)
    max_time: flt
        effectively controls max arc length (together with step_size)
    line_radius: flt
        ajust field line radius
    """

    logging.info(f"Creating centered y-plane disk plot -> {save_path}")

    # Extract the slice at y=0
    slice_plane = mesh.slice(normal="y", origin=(0, 0, 0))

    # Make a top-down plotter
    show=True
    p = pv.Plotter(off_screen=not show, window_size=fig_size)

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
        # 1) build seed points on the circle: x=r cos(t), y=0, z=r sin(t)
        t = np.linspace(0.0, 2*np.pi, n_seeds, endpoint=False)
        seeds = np.c_[r_seed*np.cos(t), np.zeros_like(t), r_seed*np.sin(t)]
        seed_src = pv.PolyData(seeds)
        
        # 2) compute streamlines (field lines) in the *full* 3D mesh, seeded from that circle
        #    (use mesh.streamlines_from_source; it uses the active vectors or vectors=...)
        # --------------------------------------------------------------------------
        # Magnetic-field streamlines
        logging.info("Adding magnetic field streamlines…")
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
        
        # optional: make them prettier
        tubes = lines.tube(radius=line_radius)  # radius in Rsun units; tune this
        
        # add to plot
        p.add_mesh(tubes, color="white", opacity=1.0)
        # show the seed circle
        #p.add_mesh(seed_src, color="white", point_size=6, render_points_as_spheres=True)
        
    # Camera looking straight at the disk from +y direction
    p.camera_position = [
        (0, 100, 0),   # eye
        (0, 0, 0),    # look center
        (0, 0, 1),    # up vector
    ]

    # Make sure full disk is visible
    #p.camera.zoom(1.5)
    p.camera.zoom(1.0)

    if psfile!=None:
        p.show() #remove this if off_screen=True
        p.screenshot(save_path+"/"+psfile)
    else:
        p.show()

    p.close()

def _get_mesh_radii_from_cfmesh(cfmesh_path: str | Path):
    """
    Read the CFmesh and return (r_min, r_max) in code units.

    Parameters
    ----------
    cfmesh_path : str | Path
        Path to a CFmesh file (CFmesh or other VTK-readable format).

    Returns
    -------
    (r_min, r_max) : tuple of floats
        Minimum and maximum radial coordinates in the mesh.
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
    cfmesh_path,
    dat_dir,
    plots_dir,
    when_str,
    ntheta,
    nphi,
    dr,
    show=True
):
    """
    Determine the minimum radius from the CFmesh, create/read the .dat at that
    radius, then plot inner boundary conditions:
    density, pressure (from ρ,T), temperature, |B|.
    """
    rmin, _ = _get_mesh_radii_from_cfmesh(cfmesh_path)
    R = rmin

    dat_path = dat_dir / f"{R:g}Rsun.dat"

    # Ensure .dat exists at the inner radius
    if not dat_path.exists():
        print(f"[INFO] Inner BC: creating {dat_path} at R = {R:g}")
        create_boundary_fromcfmesh(
            inputfile=str(cfmesh_path),
            time=when_str,
            rad_out=R*Rsun,
            nb_th=ntheta,
            nb_phi=nphi,
            eps=dr,
            output_dat=str(dat_path),
            full_output=True,
        )
    else:
        print(f"[INFO] Inner BC: using existing {dat_path}")

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
    print(f"[INFO] Saving inner BC map -> {out_png}")
    fig.savefig(out_png, dpi=200)
    if show: plt.show()
    plt.close(fig)

def _outer_bc_check_from_cfmesh(
    cfmesh_path,
    dat_dir,
    plots_dir,
    when_str,
    ntheta,
    nphi,
    dr,
    show=True,
    gamma: float = 5./3.,
):
    """
    Determine the maximum radius from the CFmesh, create/read the .dat at that
    radius, then plot outer boundary quantities:
    sound speed, Alfvén speed, Mach number, Alfvénic Mach number.
    """
    _, rmax = _get_mesh_radii_from_cfmesh(cfmesh_path)
    R = rmax

    dat_path = dat_dir / f"{R:g}Rsun.dat"

    # Ensure .dat exists at the outer radius
    if not dat_path.exists():
        print(f"[INFO] Outer BC: creating {dat_path} at R = {R:g}")
        create_boundary_fromcfmesh(
            inputfile=str(cfmesh_path),
            time=when_str,
            rad_out=R*Rsun,
            nb_th=ntheta,
            nb_phi=nphi,
            eps=dr,
            output_dat=str(dat_path),
            full_output=True,
        )
    else:
        print(f"[INFO] Outer BC: using existing {dat_path}")


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
    print(f"[INFO] Saving outer BC map -> {out_png}")
    fig.savefig(out_png, dpi=200)
    if show: plt.show()
    plt.close(fig)

def cfmesh_to_binned_spherical_grid(
    inputfile: str,
    nr: int = 50,
    ntheta: int = 90,
    nphi: int = 180,
    r_min=None,
    r_max=None,
    auto_resolution: bool = False,
    auto_kwargs: dict | None = None,
):
    """
    Convert an UNSTRUCTURED COOLFluiD CFmesh into a binned spherical grid.

    Parameters
    ----------
    inputfile : str
        Path to the CFmesh file.
    nr : int
        Number of radial bins.
    ntheta : int
        Number of colatitude bins.
    nphi : int
        Number of longitude bins.
    r_min, r_max : float or None
        Radial limits for the binning. If None, they are inferred from the mesh.
    auto_resolution : bool = False,
        enabling the binning grid resolution to be determined automatically from CFmesh density
    auto_kwargs: dict | None = None,
        automatic binning args
    
    Returns
    -------
    r_1d, theta_1d, phi_1d : 1-D arrays
        Coordinates of the centers of the spherical grid bins.
    vr_3d, vlon_3d, vclt_3d, rho0_3d, temp_3d, br_3d, blon_3d, bclt_3d : 3-D arrays
        Physical fields averaged inside each (r, θ, φ) bin.

    Notes
    -----
    - Bins with no contributing CFmesh cells get np.nan.
    - Binning is robust for unstructured tetra/hexa meshes used in COOLFluiD.
    - Auto-binning suggestions:
        out = cfmesh_to_binned_spherical_grid( "corona.CFmesh", auto_resolution=True) > Automatic resolution
        out = cfmesh_to_binned_spherical_grid( "corona.CFmesh", auto_resolution=True, auto_kwargs=dict(
            sample_max=100_000,
            max_nr=200,
            max_ntheta=360,
            max_nphi=720,
            q=0.20) )            > Automatic, but with caps (recommended on big runs)
    
        out = cfmesh_to_binned_spherical_grid( "corona.CFmesh", nr=80, ntheta=180, nphi=360, auto_resolution=False)  > Manual override still possible



    """

    # ------------------------------
    # 1. READ CFmesh structure
    if not os.path.isfile(inputfile):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), inputfile)
    
    with open(inputfile, "r") as f:
        lines = f.readlines()
    
    idx0, idx1, idx2, idx3, nbelements, nend, comment = readstruct(lines)

    connectivity = np.loadtxt(lines[idx0:idx0 + nbelements], dtype=int)
    coordinates = np.loadtxt(lines[idx1:idx2 - 1], dtype=float)

    # first 6 vertices of each element → cell center
    nodes = connectivity[:, :6]
    centers = coordinates[nodes].mean(axis=1)
    x, y, z = centers.T
    
    # ------------------------------
    # 2. SPHERICAL COORDS
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(z / r)                     # [0, π]
    phi = np.arctan2(y, x)
    phi[phi < 0] += 2*np.pi                      # force into [0, 2π]
    
    if r_min is None:
        r_min = r.min()
    if r_max is None:
        r_max = r.max()

    if auto_resolution:
        auto_kwargs = {} if auto_kwargs is None else dict(auto_kwargs)
        nr, ntheta, nphi = _auto_spherical_binning_resolution(
            r, theta, phi,
            r_min=r_min, r_max=r_max,
            **auto_kwargs
        )
        print(f"[INFO] Auto binning resolution: nr={nr}, ntheta={ntheta}, nphi={nphi}")

    else:
        if nr is None or ntheta is None or nphi is None:
            raise ValueError("nr, ntheta, nphi must be set unless auto_resolution=True")
        
    # ------------------------------
    # 3. READ INITIAL DATA (same as your function)
    bd = comment[-nend - 1][0] + 1
    bf = comment[-nend][0]
    Init = np.loadtxt(lines[bd:bf], dtype=np.float64)
    
    rho0 = Init[:, 0] * 1.67e-13 / 1.67e-27 # [m^-3]
    Vx0  = Init[:, 1] * 480248.0 # [m/s]
    Vy0  = Init[:, 2] * 480248.0 # [m/s]
    Vz0  = Init[:, 3] * 480248.0 # [m/s]
    Bx   = Init[:, 4] * 2.2e-4 # [T]
    By   = Init[:, 5] * 2.2e-4 # [T]
    Bz   = Init[:, 6] * 2.2e-4 # [T]
    Pressure = Init[:, 7] * 0.03851 
    temp = Pressure / rho0 / 2.0 / 1.38e-23
    
    # spherical projections (unchanged)
    r_bis = np.hypot(x, y)
    eps = 1e-12

    vr   = (x*Vx0 + y*Vy0 + z*Vz0) / (r + eps)
    vlon = (-y*Vx0 + x*Vy0) / (r_bis + eps)
    vclt = (x*z*Vx0 + y*z*Vy0 - (r_bis+eps)*Vz0) / ((r + eps)*(r_bis + eps))

    br   = (x*Bx + y*By + z*Bz) / (r + eps)
    blon = (-y*Bx + x*By) / (r_bis + eps)
    bclt = (x*z*Bx + y*z*By - (r_bis+eps)*Bz) / ((r + eps)*(r_bis + eps))

    # ------------------------------
    # 4. PREPARE SPHERICAL BINS
    # ------------------------------
    r_edges     = np.linspace(r_min, r_max, nr+1)
    theta_edges = np.linspace(0, np.pi, ntheta+1)
    phi_edges   = np.linspace(0, 2*np.pi, nphi+1)

    r_centers     = 0.5 * (r_edges[:-1] + r_edges[1:])
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi_centers   = 0.5 * (phi_edges[:-1] + phi_edges[1:])

    # indices of each cell into the bins
    i_r   = np.digitize(r,     r_edges)     - 1
    i_th  = np.digitize(theta, theta_edges) - 1
    i_ph  = np.digitize(phi,   phi_edges)   - 1

    # mask out-of-range indices
    valid = (
        (i_r  >= 0) & (i_r  < nr) &
        (i_th >= 0) & (i_th < ntheta) &
        (i_ph >= 0) & (i_ph < nphi)
    )

    # ------------------------------
    # 5. ACCUMULATE INTO BINS
    # ------------------------------
    shape = (nr, ntheta, nphi)
    accum = lambda: np.zeros(shape)
    count = np.zeros(shape)

    vr_sum   = accum(); vlon_sum = accum(); vclt_sum = accum()
    rho_sum  = accum(); temp_sum = accum()
    br_sum   = accum(); blon_sum = accum(); bclt_sum = accum()

    # accumulate contributions
    for idx in np.where(valid)[0]:
        ii = i_r[idx]; jj = i_th[idx]; kk = i_ph[idx]
        count[ii,jj,kk] += 1

        vr_sum[ii,jj,kk]   += vr[idx]
        vlon_sum[ii,jj,kk] += vlon[idx]
        vclt_sum[ii,jj,kk] += vclt[idx]
        rho_sum[ii,jj,kk]  += rho0[idx]
        temp_sum[ii,jj,kk] += temp[idx]
        br_sum[ii,jj,kk]   += br[idx]
        blon_sum[ii,jj,kk] += blon[idx]
        bclt_sum[ii,jj,kk] += bclt[idx]

    # ------------------------------
    # 6. AVERAGE (bins with no hits → nan)
    # ------------------------------
    with np.errstate(invalid="ignore", divide="ignore"):

        vr_3d   = vr_sum   / count
        vlon_3d = vlon_sum / count
        vclt_3d = vclt_sum / count
        rho_3d  = rho_sum  / count
        temp_3d = temp_sum / count
        br_3d   = br_sum   / count
        blon_3d = blon_sum / count
        bclt_3d = bclt_sum / count
        print(br_3d)

    filled = np.isfinite(rho_3d).sum()
    total = rho_3d.size
    print(f"Filled bins: {filled}/{total} = {filled/total:.3%}")
    #As a rule of thumb for unstructured binning:
    #< 1% filled → resolution is much too fine
    #5–30% filled → usually OK
    #> 50% filled → bins probably too coarse (or mesh is close to structured)

    return (
        r_centers,
        theta_centers,
        phi_centers,
        vr_3d, vlon_3d, vclt_3d,
        rho_3d, temp_3d,
        br_3d, blon_3d, bclt_3d,
    )

def _auto_spherical_binning_resolution(
    r, theta, phi,
    r_min=None, r_max=None,
    sample_max=100_000,
    q=0.35,
    min_nr=8, max_nr=120,
    min_ntheta=24, max_ntheta=180,
    min_nphi=48, max_nphi=360,
):
    """
    Estimate (nr, ntheta, nphi) from point density in (r,theta,phi).

    Strategy:
      - subsample points if huge
      - estimate typical spacing as a low-quantile of sorted unique diffs
        (robust to refinement regions and duplicates)
      - convert spacings into bin counts over the full domain
      - clamp to min/max to avoid pathological grids
    """
    if r_min is None: r_min = float(np.min(r))
    if r_max is None: r_max = float(np.max(r))

    N = r.size
    if N > sample_max:
        idx = np.random.default_rng(0).choice(N, size=sample_max, replace=False)
        r_s, th_s, ph_s = r[idx], theta[idx], phi[idx]
    else:
        r_s, th_s, ph_s = r, theta, phi

    # Helper: robust spacing estimate from unique sorted values
    def typical_spacing(x):
        xu = np.unique(x)
        if xu.size < 3:
            return None
        dx = np.diff(np.sort(xu))
        dx = dx[dx > 0]
        if dx.size == 0:
            return None
        return float(np.quantile(dx, q))

    dr = typical_spacing(r_s)
    dth = typical_spacing(th_s)
    dph = typical_spacing(ph_s)

    # Fallbacks if something is degenerate
    if dr is None:  dr = (r_max - r_min) / 50.0
    if dth is None: dth = np.pi / 180.0
    if dph is None: dph = 2*np.pi / 360.0

    nr = int(np.ceil((r_max - r_min) / dr))
    ntheta = int(np.ceil(np.pi / dth))
    nphi = int(np.ceil((2*np.pi) / dph))

    # Clamp
    nr = int(np.clip(nr, min_nr, max_nr))
    ntheta = int(np.clip(ntheta, min_ntheta, max_ntheta))
    nphi = int(np.clip(nphi, min_nphi, max_nphi))

    return nr, ntheta, nphi

# To use instead of the former if not successful enough
# estimate spacing from nearest neighbors (not unique diffs)
# The “unique-diff quantile” approach is fragile on unstructured meshes.
# A more robust method is: for a random sample of points, compute distance to the nearest neighbor in (r, theta, phi)  with a periodic phi, then set bin widths from the median.
## !! NB: not fully sanity checked yet
from scipy.spatial import cKDTree
def _auto_resolution_nn(r, theta, phi, r_min, r_max,
                        sample_max=80_000,
                        safety=1.8,
                        max_nr=160, max_ntheta=240, max_nphi=480):
    N = r.size
    rng = np.random.default_rng(0)
    idx = rng.choice(N, size=min(N, sample_max), replace=False)

    rs = r[idx]
    ts = theta[idx]
    ps = phi[idx]

    # handle periodic phi by duplicating points shifted by ±2π
    X = np.column_stack([rs, ts, ps])
    Xp = np.column_stack([rs, ts, ps + 2*np.pi])
    Xm = np.column_stack([rs, ts, ps - 2*np.pi])
    Xall = np.vstack([X, Xp, Xm])

    tree = cKDTree(Xall)
    d, _ = tree.query(X, k=2)  # k=2: first is itself, second is nearest neighbor
    d_nn = d[:, 1]

    # Use median nearest-neighbor distance as typical spacing
    d0 = float(np.median(d_nn)) * safety

    # Convert "distance" into separate bin widths (simple heuristic)
    dr = 0.6 * d0
    dth = 0.2 * d0
    dph = 0.2 * d0

    nr = int(np.clip(np.ceil((r_max - r_min) / dr), 8, max_nr))
    ntheta = int(np.clip(np.ceil(np.pi / dth), 24, max_ntheta))
    nphi = int(np.clip(np.ceil((2*np.pi) / dph), 48, max_nphi))
    return nr, ntheta, nphi
