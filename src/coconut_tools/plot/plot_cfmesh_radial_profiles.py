"""
Plot radial profiles from a CFmesh file at fixed longitude/colatitude angles.

This module reads a COCONUT CFmesh output, converts quantities to physical
units, and interpolates them along 1D radial lines for selected (phi, clt).
It creates a 3-panel figure, one panel per (phi, clt) pair.
"""

from __future__ import annotations

import errno
import os
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator

from coconut_tools.toheliosphere.create_dat import readstruct
from coconut_tools.tools.logger_config import setup_logger

logger = setup_logger(__name__)

QuantityKey = str


def _read_cfmesh_quantities(input_cfmesh: str) -> Dict[str, np.ndarray]:
    """Read a CFmesh file and compute cartesian + spherical quantities.

    Args:
        input_cfmesh (str): Path to the input CFmesh file.

    Returns:
        Dict[str, np.ndarray]: Dict containing arrays for coordinates and quantities.
    """
    if not os.path.isfile(input_cfmesh):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), input_cfmesh)

    with open(input_cfmesh, "r") as f:
        lines = f.readlines()

    idx0, idx1, idx2, _, nbelements, nend, comment = readstruct(lines)

    connectivity = np.loadtxt(lines[idx0:idx0 + nbelements], dtype=int)
    coordinates = np.loadtxt(lines[idx1:idx2 - 1], dtype=float)

    nodes = connectivity[:, :6]
    centers = coordinates[nodes].mean(axis=1)

    bd = comment[-nend - 1][0] + 1
    bf = comment[-nend][0]
    initial = np.loadtxt(lines[bd:bf], dtype=np.float64)

    # Physical quantities (same conversion as create_boundary_fromcfmesh)
    rho = initial[:, 0] * 1.67e-13 / 1.67e-27  # m^-3
    vx = initial[:, 1] * 480248.0              # m/s
    vy = initial[:, 2] * 480248.0              # m/s
    vz = initial[:, 3] * 480248.0              # m/s
    bx = initial[:, 4] * 2.2e-4                # T
    by = initial[:, 5] * 2.2e-4                # T
    bz = initial[:, 6] * 2.2e-4                # T
    pressure = initial[:, 7] * 0.03851         # Pa
    temperature = pressure / rho / 2 / 1.38e-23  # K (mu=0.5 hardcoded)

    x, y, z = centers.T
    r = np.sqrt(x**2 + y**2 + z**2)
    rho_xy = np.sqrt(x**2 + y**2)

    r_safe = np.maximum(r, 1e-8)
    rho_xy_safe = np.maximum(rho_xy, 1e-8)

    # Spherical components (clt: colatitude)
    vr = (x * vx + y * vy + z * vz) / r_safe
    vlon = (-y * vx + x * vy) / rho_xy_safe
    vclt = (z * (x * vx + y * vy) - (rho_xy**2) * vz) / (r_safe * rho_xy_safe)

    br = (x * bx + y * by + z * bz) / r_safe
    blon = (-y * bx + x * by) / rho_xy_safe
    bclt = (z * (x * bx + y * by) - (rho_xy**2) * bz) / (r_safe * rho_xy_safe)

    phi = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    clt = np.arccos(np.clip(z / r_safe, -1.0, 1.0))

    return {
        "x": x,
        "y": y,
        "z": z,
        "r": r,
        "phi": phi,
        "clt": clt,
        "density": rho,
        "temperature": temperature,
        "pressure": pressure,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "bx": bx,
        "by": by,
        "bz": bz,
        "vr": vr,
        "vlon": vlon,
        "vclt": vclt,
        "br": br,
        "blon": blon,
        "bclt": bclt,
    }


def _quantity_catalog() -> Dict[QuantityKey, Tuple[QuantityKey, str]]:
    """Define supported quantities and their labels.

    Returns:
        Dict[str, Tuple[str, str]]: Maps alias -> (canonical_key, label)
    """
    return {
        "density": ("density", r"Density [$m^{-3}$]"),
        "temperature": ("temperature", "Temperature [K]"),
        "pressure": ("pressure", "Pressure [Pa]"),
        "bx": ("bx", "Bx [T]"),
        "by": ("by", "By [T]"),
        "bz": ("bz", "Bz [T]"),
        "vx": ("vx", "Vx [m/s]"),
        "vy": ("vy", "Vy [m/s]"),
        "vz": ("vz", "Vz [m/s]"),
        "br": ("br", "Br [T]"),
        "bclt": ("bclt", "Bclt [T]"),
        "blon": ("blon", "Blon [T]"),
        "bphi": ("blon", "Blon [T]"),
        "vr": ("vr", "Vr [m/s]"),
        "vclt": ("vclt", "Vclt [m/s]"),
        "vlon": ("vlon", "Vlon [m/s]"),
        "vphi": ("vlon", "Vlon [m/s]"),
    }


def plot_cfmesh_radial_profiles(
    input_cfmesh: str,
    output_png: str,
    quantity: str,
    phi_values: Iterable[float],
    clt_values: Iterable[float],
    r_min_rsun: float = 1.0,
    r_max_rsun: float | None = None,
    n_samples: int = 200,
    phi_degrees: bool = True,
    clt_degrees: bool = True,
    neighbors: int = 50,
) -> None:
    """Create a 3-panel radial profile plot for a chosen quantity.

    The three panels correspond to three fixed (phi, clt) pairs. Each panel
    shows the selected quantity along radius from ``r_min_rsun`` to
    ``r_max_rsun``.

    Args:
        input_cfmesh (str): Path to the CFmesh file.
        output_png (str): Path to the output figure (PNG).
        quantity (str): Quantity to plot. Supported:
            density, temperature, pressure, bx, by, bz, vx, vy, vz,
            br, bclt, blon, vr, vclt, vlon (aliases: bphi, vphi).
        phi_values (Iterable[float]): Three longitudes for the panels.
        clt_values (Iterable[float]): Three colatitudes for the panels.
        r_min_rsun (float, optional): Minimum radius in Rsun. Defaults to 1.0.
        r_max_rsun (float | None, optional): Maximum radius in Rsun. Defaults
            to max radius from the mesh.
        n_samples (int, optional): Number of radial samples per panel.
        phi_degrees (bool, optional): If True, ``phi_values`` are in degrees.
        clt_degrees (bool, optional): If True, ``clt_values`` are in degrees.
        neighbors (int, optional): Neighbor count for RBFInterpolator.

    Returns:
        None
    """
    phi_values = list(phi_values)
    clt_values = list(clt_values)
    if len(phi_values) != 3 or len(clt_values) != 3:
        raise ValueError("phi_values and clt_values must each contain exactly 3 values.")

    catalog = _quantity_catalog()
    key = quantity.strip().lower()
    if key not in catalog:
        supported = ", ".join(sorted(catalog.keys()))
        raise ValueError(f"Unsupported quantity '{quantity}'. Supported: {supported}")

    data = _read_cfmesh_quantities(input_cfmesh)
    coords = np.column_stack((data["x"], data["y"], data["z"]))
    coords_unique, idx_unique = np.unique(coords, axis=0, return_index=True)

    canonical_key, ylabel = catalog[key]
    values = data[canonical_key][idx_unique]

    if r_max_rsun is None:
        r_max_rsun = float(np.max(data["r"]))
    if r_max_rsun <= r_min_rsun:
        raise ValueError("r_max_rsun must be larger than r_min_rsun.")

    r_line = np.linspace(r_min_rsun, r_max_rsun, n_samples)

    n_neighbors = min(neighbors, len(coords_unique))
    interpolator = RBFInterpolator(coords_unique, values, kernel="linear", neighbors=n_neighbors)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for i, ax in enumerate(axes):
        phi = np.deg2rad(phi_values[i]) if phi_degrees else float(phi_values[i])
        clt = np.deg2rad(clt_values[i]) if clt_degrees else float(clt_values[i])

        x_line = r_line * np.sin(clt) * np.cos(phi)
        y_line = r_line * np.sin(clt) * np.sin(phi)
        z_line = r_line * np.cos(clt)
        line_points = np.column_stack((x_line, y_line, z_line))

        q_line = interpolator(line_points)

        ax.plot(r_line, q_line, linewidth=1.6)
        ax.set_xlabel(r"Radius [$R_\odot$]")
        ax.grid(True, alpha=0.3)

        phi_label = f"{phi_values[i]:.2f}°" if phi_degrees else f"{phi_values[i]:.3f} rad"
        clt_label = f"{clt_values[i]:.2f}°" if clt_degrees else f"{clt_values[i]:.3f} rad"
        ax.set_title(f"phi={phi_label}, clt={clt_label}")

    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"Radial profile: {canonical_key}")
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.savefig(output_png, dpi=300)
    plt.close(fig)

    logger.info("Saved figure to %s", output_png)


if __name__ == "__main__":
    # --- User inputs (edit these) ---
    input_cfmesh = "E:/euhforia/2017/result_fullmhd/corona.CFmesh"
    output_png = "E:/euhforia/2017/result_fullmhd/radial_profiles.png"

    # Quantity to plot
    quantity = "vr"

    # Three panels: phi and clt for each panel
    phi_values = [0.0, -90.0, 90.0]   # degrees by default
    clt_values = [90.0, 90.0, 90.0]   # degrees by default

    # Radial range and sampling
    r_min_rsun = 1.0
    r_max_rsun = None  # use mesh max
    n_samples = 200

    plot_cfmesh_radial_profiles(
        input_cfmesh=input_cfmesh,
        output_png=output_png,
        quantity=quantity,
        phi_values=phi_values,
        clt_values=clt_values,
        r_min_rsun=r_min_rsun,
        r_max_rsun=r_max_rsun,
        n_samples=n_samples,
        phi_degrees=True,
        clt_degrees=True,
    )
