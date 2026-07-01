"""Utilities to read and and post-process COCONUT outputs.

This module provides helpers to extract data from CFmesh and vtu files,
for MHD quantities.

Example:
    from coconut_tools.coconut_to_numpy import (
        compute_cfmesh_unstructured_cartesian_gradients,
        compute_cfmesh_unstructured_spherical_gradients,
        compute_binned_spherical_grid_cartesian_gradients,
        compute_binned_spherical_grid_spherical_gradients,
    )
    from coconut_tools.create_dat import readstruct

    centers, cart_grads = compute_cfmesh_unstructured_cartesian_gradients(
        "corona.CFmesh",
        readstruct_fn=readstruct,
    )
    # cart_grads[name] has shape (n_cells, 3) for (d/dx, d/dy, d/dz)

    centers, sph_grads = compute_cfmesh_unstructured_spherical_gradients(
        "corona.CFmesh",
        readstruct_fn=readstruct,
    )
    # sph_grads[name] has shape (n_cells, 3) for (d/dr, d/dtheta, d/dphi)

    grad_cart = compute_binned_spherical_grid_cartesian_gradients(
        r_centers, theta_centers, phi_centers, rho_3d
    )
    # grad_cart has shape (nr, ntheta, nphi, 3)

    grad_sph = compute_binned_spherical_grid_spherical_gradients(
        r_centers, theta_centers, phi_centers, rho_3d
    )
    # grad_sph has shape (nr, ntheta, nphi, 3)
"""

# contact: Q. Noraz

from __future__ import annotations

import errno
import os
from collections import defaultdict

import numpy as np
import pyvista as pv
from scipy.stats import binned_statistic_dd
import matplotlib.pyplot as plt
from coconut_tools.toheliosphere.create_dat import readstruct
from coconut_tools.tools.logger_config import setup_logger

logger = setup_logger(__name__)

###########################################
# From .vtu !!! Should filter invalid cells if needed -> see reader.py
###########################################

def _cart_to_spherical(x, y, z):
    r = np.sqrt(x*x + y*y + z*z)
    # colatitude theta in [0, pi]
    theta = np.arccos(np.clip(z / np.where(r == 0, 1.0, r), -1.0, 1.0))
    # longitude phi in [0, 2pi)
    phi = np.mod(np.arctan2(y, x), 2*np.pi)
    return r, theta, phi


def _auto_spherical_binning_resolution(r, theta, phi, q=0.05):
    """
    Robust-ish version: use a low quantile of non-zero spacings instead of min,
    to avoid one tiny refined region forcing ultra-fine bins everywhere.
    """
    def _step(a):
        a = np.asarray(a)
        a = np.sort(a[np.isfinite(a)])
        if a.size < 2:
            return np.nan
        d = np.diff(a)
        d = d[d > 0]
        if d.size == 0:
            return np.nan
        return float(np.quantile(d, q))
    return _step(r), _step(theta), _step(phi)


def _weighted_mean_binned(sample, values, bins, weights):
    """
    Return weighted mean per bin: sum(values*weights)/sum(weights)
    """
    num = binned_statistic_dd(sample, values * weights, statistic="sum", bins=bins).statistic
    den = binned_statistic_dd(sample, weights,          statistic="sum", bins=bins).statistic
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out


def vtu_to_binned_spherical_grid(
    vtu_path: str,
    *,
    scalars: dict[str, str] = None,
    vectors: dict[str, tuple[str, str, str]] = None,
    r_range: tuple[float, float] | None = None,
    theta_range: tuple[float, float] = (0.0, np.pi),
    phi_range: tuple[float, float] = (0.0, 2*np.pi),
    dr: float | None = None,
    dtheta: float | None = None,
    dphi: float | None = None,
    prefer_cell_data: bool = True,
    volume_weighted: bool = True,
):
    """
    Bin VTU data onto a regular (r, theta, phi) grid.

    scalars: mapping output_name -> array_name_in_vtu
             e.g. {"rho": "rho", "p": "p", "T": "T"}
    vectors: mapping output_name -> (ax, ay, az) array names in vtu
             e.g. {"B": ("Bx","By","Bz"), "v": ("vx","vy","vz")}
             (If your VTU stores vector arrays already, you can pass just one name
              via scalars and treat it separately.)
    """
    if scalars is None:
        scalars = {}
    if vectors is None:
        vectors = {}

    mesh = pv.read(vtu_path)

    # Work on cell data (closest analogue to CFmesh element-centered values)
    if prefer_cell_data:
        # If some requested arrays only exist on points, convert once.
        need_convert = False
        for name in list(scalars.values()) + [c for t in vectors.values() for c in t]:
            if (name not in mesh.cell_data) and (name in mesh.point_data):
                need_convert = True
                break
        if need_convert:
            mesh = mesh.point_data_to_cell_data(pass_point_data=True)

    centers = mesh.cell_centers()  # PolyData with points at cell centers
    xyz = np.asarray(centers.points)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r, theta, phi = _cart_to_spherical(x, y, z)

    # Apply ranges
    if r_range is None:
        rmin, rmax = float(np.nanmin(r)), float(np.nanmax(r))
    else:
        rmin, rmax = r_range

    tmin, tmax = theta_range
    pmin, pmax = phi_range

    m = (
        np.isfinite(r) & np.isfinite(theta) & np.isfinite(phi)
        & (r >= rmin) & (r <= rmax)
        & (theta >= tmin) & (theta <= tmax)
        & (phi >= pmin) & (phi <= pmax)
    )
    r, theta, phi = r[m], theta[m], phi[m]
    sample = np.column_stack([r, theta, phi])

    # Choose bin widths
    if dr is None or dtheta is None or dphi is None:
        adr, adt, adp = _auto_spherical_binning_resolution(r, theta, phi)
        dr = adr if dr is None else dr
        dtheta = adt if dtheta is None else dtheta
        dphi = adp if dphi is None else dphi

    if not np.isfinite(dr) or not np.isfinite(dtheta) or not np.isfinite(dphi):
        raise ValueError("Could not infer bin widths (try setting dr/dtheta/dphi manually).")

    r_edges = np.arange(rmin, rmax + dr, dr)
    t_edges = np.arange(tmin, tmax + dtheta, dtheta)
    p_edges = np.arange(pmin, pmax + dphi, dphi)
    bins = (r_edges, t_edges, p_edges)

    # Weights (cell volumes) if requested
    weights = None
    if volume_weighted:
        sized = mesh.compute_cell_sizes(volume=True, area=False, length=False)
        vol = np.asarray(sized.cell_data["Volume"])
        weights = vol[m].astype(float)

    # Fetch arrays (cell_data preferred; fallback to point_data if needed)
    out = {}
    coords = {"r_edges": r_edges, "theta_edges": t_edges, "phi_edges": p_edges}
    out.update(coords)

    def _get_array(arr_name: str):
        if arr_name in mesh.cell_data:
            return np.asarray(mesh.cell_data[arr_name])[m]
        if arr_name in mesh.point_data:
            # If still point-based here, we’re binning cell centers with point data: not ideal, but workable.
            return np.asarray(mesh.point_data[arr_name])[m]
        raise KeyError(f"Array '{arr_name}' not found in cell_data or point_data.")

    for out_name, arr_name in scalars.items():
        vals = _get_array(arr_name).astype(float)
        if weights is None:
            stat = binned_statistic_dd(sample, vals, statistic="mean", bins=bins).statistic
        else:
            stat = _weighted_mean_binned(sample, vals, bins=bins, weights=weights)
        out[out_name] = stat

    for out_name, (ax, ay, az) in vectors.items():
        vx = _get_array(ax).astype(float)
        vy = _get_array(ay).astype(float)
        vz = _get_array(az).astype(float)

        if weights is None:
            bx = binned_statistic_dd(sample, vx, statistic="mean", bins=bins).statistic
            by = binned_statistic_dd(sample, vy, statistic="mean", bins=bins).statistic
            bz = binned_statistic_dd(sample, vz, statistic="mean", bins=bins).statistic
        else:
            bx = _weighted_mean_binned(sample, vx, bins=bins, weights=weights)
            by = _weighted_mean_binned(sample, vy, bins=bins, weights=weights)
            bz = _weighted_mean_binned(sample, vz, bins=bins, weights=weights)

        out[out_name] = np.stack([bx, by, bz], axis=-1)

    return out

def spherical_average_on_grid(q_rtp, theta_centers):
    """
    q_rtp: array (Nr, Nθ, Nφ)
    theta_centers: array (Nθ,) in radians (colatitude)
    """
    w = np.sin(theta_centers)                      # dΩ weight factor
    w2 = w[:, None]                                # broadcast over φ
    num = np.nansum(q_rtp * w2, axis=(1, 2))
    den = np.nansum(np.isfinite(q_rtp) * w2, axis=(1, 2))
    return num / den

# Example usage:
# theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
# r_centers     = 0.5 * (r_edges[:-1] + r_edges[1:])
# qbar_r = spherical_average_on_grid(grid["rho"], theta_centers)
# plt.plot(r_centers, qbar_r)
# plt.xlabel("r"); plt.ylabel("<rho>(r)"); plt.yscale("log"); plt.show()

def radial_shell_average_from_vtu(vtu_path, array_name, r_edges):
    mesh = pv.read(vtu_path)

    # cell-centered positions
    centers = mesh.cell_centers().points
    x, y, z = centers[:,0], centers[:,1], centers[:,2]
    r = np.sqrt(x*x + y*y + z*z)

    # get cell-centered quantity (convert if needed)
    if array_name not in mesh.cell_data and array_name in mesh.point_data:
        mesh = mesh.point_data_to_cell_data(pass_point_data=True)
    q = np.asarray(mesh.cell_data[array_name], dtype=float)

    # volumes for weighting
    sized = mesh.compute_cell_sizes(volume=True, area=False, length=False)
    vol = np.asarray(sized.cell_data["Volume"], dtype=float)

    # bin by radius
    idx = np.digitize(r, r_edges) - 1
    Nr = len(r_edges) - 1
    qbar = np.full(Nr, np.nan)

    for k in range(Nr):
        m = (idx == k) & np.isfinite(q) & np.isfinite(vol) & (vol > 0)
        if np.any(m):
            qbar[k] = np.sum(q[m] * vol[m]) / np.sum(vol[m])

    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    return r_centers, qbar

# Example plot:
# r, qbar = radial_shell_average_from_vtu("out.vtu", "rho", r_edges=np.linspace(rmin, rmax, 200))
# plt.plot(r, qbar); plt.xlabel("r"); plt.ylabel("<rho>_shell"); plt.show()

###########################################
# From .CFmesh to binned spherical structured grid
###########################################

def cfmesh_to_binned_spherical_grid(
    inputfile: str,
    nr: int = 50,
    ntheta: int = 90,
    nphi: int = 180,
    r_min: float | None = None,
    r_max: float | None = None,
    extra_field_names: list[str] | None = None,
    auto_resolution: bool = False,
    auto_kwargs: Mapping[str, Any] | None = None,
):
    """Convert an unstructured CFmesh into a binned spherical grid.

    Args:
        inputfile: Path to the CFmesh file.
        nr: Number of radial bins.
        ntheta: Number of colatitude bins.
        nphi: Number of longitude bins.
        r_min: Minimum radius for binning. If None, inferred from the mesh.
        r_max: Maximum radius for binning. If None, inferred from the mesh.
        auto_resolution: If True, compute bin counts automatically.
        auto_kwargs: Optional arguments for auto-resolution.

    Returns:
        Tuple of arrays: (r_centers, theta_centers, phi_centers, vr_3d, vlon_3d,
        vclt_3d, rho_3d, temp_3d, br_3d, blon_3d, bclt_3d).
    + potential extra variables

    e.g. (r, th, ph, vr, vlon, vclt, rho, temp, br, blon, bclt) = cfmesh_to_binned_spherical_grid("corona.CFmesh", nr=80, ntheta=90, nphi=180)
    + potential extra variables
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
        logger.info("Auto binning resolution: nr=%s, ntheta=%s, nphi=%s", nr, ntheta, nphi)

    else:
        if nr is None or ntheta is None or nphi is None:
            raise ValueError("nr, ntheta, nphi must be set unless auto_resolution=True")
        
    # ------------------------------
    # 3. READ INITIAL DATA (same as your function)
    bd = comment[-nend - 1][0] + 1
    bf = comment[-nend][0]
    Init = np.loadtxt(lines[bd:bf], dtype=np.float64)
    
    rho0 = Init[:, 0] * 1.67e-13 # rho # / 1.67e-27 # n'=rho/mp=mu*n
    Vx0  = Init[:, 1] * 480248.0 # [m/s]
    Vy0  = Init[:, 2] * 480248.0 # [m/s]
    Vz0  = Init[:, 3] * 480248.0 # [m/s]
    Bx   = Init[:, 4] * 2.2e-4 # [T]
    By   = Init[:, 5] * 2.2e-4 # [T]
    Bz   = Init[:, 6] * 2.2e-4 # [T]
    Pressure = Init[:, 7] * 0.03851 # Pa
    temp = Pressure / rho0 / 2.0 / 1.38e-23 * 1.67e-27 # temp with mu=0.5 hardcoded # if * 1.67e-27 removed, P*mu/n'/kb = T [K] with mu=0.5 already hardcoded here !!!
    phi_div = Init[:, 8] * 480248.0 * 2.2e-4 # divergence cleaning variable phi ; vRef*bRef

    # If extra fields in CFmesh
    ncols = Init.shape[1]
    extra_fields = {}
    if ncols > 9:
        extra_field_names = [] if extra_field_names is None else list(extra_field_names)
        
        for j in range(9, ncols):
            if (j - 9) < len(extra_field_names):
                name = extra_field_names[j - 9]
            else:
                name = f"extra_{j}"
            extra_fields[name] = Init[:, j]
    
    # spherical projections (unchanged)
    r_bis = np.hypot(x, y)
    eps = 1e-12

    vr   = (x*Vx0 + y*Vy0 + z*Vz0) / (r + eps)
    vlon = (-y*Vx0 + x*Vy0) / (r_bis + eps)
    vclt = (x*z*Vx0 + y*z*Vy0 - (r_bis**2)*Vz0) / ((r + eps)*(r_bis + eps))

    br   = (x*Bx + y*By + z*Bz) / (r + eps)
    blon = (-y*Bx + x*By) / (r_bis + eps)
    bclt = (x*z*Bx + y*z*By - (r_bis**2)*Bz) / ((r + eps)*(r_bis + eps))

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
    rho_sum  = accum(); temp_sum = accum(); phid_sum = accum()
    br_sum   = accum(); blon_sum = accum(); bclt_sum = accum()
    # Dynamics accumulator if more variables
    extra_sums = {name: accum() for name in extra_fields}

    # accumulate contributions
    for idx in np.where(valid)[0]:
        ii = i_r[idx]; jj = i_th[idx]; kk = i_ph[idx]
        count[ii,jj,kk] += 1

        vr_sum[ii,jj,kk]   += vr[idx]
        vlon_sum[ii,jj,kk] += vlon[idx]
        vclt_sum[ii,jj,kk] += vclt[idx]
        rho_sum[ii,jj,kk]  += rho0[idx]
        temp_sum[ii,jj,kk] += temp[idx]
        phid_sum[ii,jj,kk] += phi_div[idx]
        br_sum[ii,jj,kk]   += br[idx]
        blon_sum[ii,jj,kk] += blon[idx]
        bclt_sum[ii,jj,kk] += bclt[idx]
        # if more variables
        for name, values in extra_fields.items():
            extra_sums[name][ii, jj, kk] += values[idx]

    # ------------------------------
    # 6. AVERAGE (bins with no hits → nan)
    # ------------------------------
    with np.errstate(invalid="ignore", divide="ignore"):

        vr_3d   = vr_sum   / count
        vlon_3d = vlon_sum / count
        vclt_3d = vclt_sum / count
        rho_3d  = rho_sum  / count
        temp_3d = temp_sum / count
        phid_3d = phid_sum / count
        br_3d   = br_sum   / count
        blon_3d = blon_sum / count
        bclt_3d = bclt_sum / count
        logger.debug("br_3d: %s", br_3d)
        # if more variables
        extra_3d = {name: arr / count for name, arr in extra_sums.items()}

    filled = np.isfinite(rho_3d).sum()
    total = rho_3d.size
    logger.info("Filled bins: %s/%s = %.3f%%", filled, total, 100.0 * filled / total)
    #As a rule of thumb for unstructured binning:
    #< 1% filled → resolution is much too fine
    #5–30% filled → usually OK
    #> 50% filled → bins probably too coarse (or mesh is close to structured)

    return (
        r_centers,
        theta_centers,
        phi_centers,
        vr_3d, vlon_3d, vclt_3d,
        rho_3d, temp_3d, phid_3d,
        br_3d, blon_3d, bclt_3d,
        extra_3d
    )

def _auto_spherical_binning_resolution(
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    r_min: float | None = None,
    r_max: float | None = None,
    sample_max: int = 100_000,
    q: float = 0.35,
    min_nr: int = 8,
    max_nr: int = 120,
    min_ntheta: int = 24,
    max_ntheta: int = 180,
    min_nphi: int = 48,
    max_nphi: int = 360,
) -> tuple[int, int, int]:
    """Estimate (nr, ntheta, nphi) from point density.

    Args:
        r: Radial coordinates.
        theta: Colatitude coordinates.
        phi: Longitude coordinates.
        r_min: Minimum radius. If None, inferred from `r`.
        r_max: Maximum radius. If None, inferred from `r`.
        sample_max: Max samples for spacing estimate.
        q: Quantile for spacing estimate.
        min_nr: Minimum radial bins.
        max_nr: Maximum radial bins.
        min_ntheta: Minimum theta bins.
        max_ntheta: Maximum theta bins.
        min_nphi: Minimum phi bins.
        max_nphi: Maximum phi bins.

    Returns:
        Tuple of (nr, ntheta, nphi).
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

def _centers_to_edges(c: np.ndarray) -> np.ndarray:
    """Infer bin edges from monotonically increasing bin centers."""
    c = np.asarray(c, dtype=float)
    if c.ndim != 1 or c.size < 2:
        raise ValueError("centers must be 1D with length >= 2")
    dc = np.diff(c)
    if np.any(dc <= 0):
        raise ValueError("centers must be strictly increasing")
    edges = np.empty(c.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0] = c[0] - 0.5 * dc[0]
    edges[-1] = c[-1] + 0.5 * dc[-1]
    return edges


def _weighted_moments(x: np.ndarray, w: np.ndarray):
    """
    Return weighted mean, std, skewness, kurtosis(excess).
    Uses population definitions with weights (not unbiased sample estimators).
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)

    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan, np.nan, np.nan, np.nan

    x = x[m]
    w = w[m]
    wsum = np.sum(w)
    if wsum <= 0:
        return np.nan, np.nan, np.nan, np.nan

    mu = np.sum(w * x) / wsum
    xc = x - mu
    m2 = np.sum(w * xc**2) / wsum
    std = np.sqrt(m2)

    if std == 0 or not np.isfinite(std):
        return mu, std, np.nan, np.nan

    m3 = np.sum(w * xc**3) / wsum
    m4 = np.sum(w * xc**4) / wsum
    skew = m3 / (std**3)
    kurt_excess = m4 / (std**4) - 3.0
    return mu, std, skew, kurt_excess


def radial_profile_volume_weighted(
    q_rtp: np.ndarray,
    r_centers: np.ndarray,
    theta_centers: np.ndarray,
    phi_centers: np.ndarray,
    *,
    r_edges: np.ndarray | None = None,
    theta_edges: np.ndarray | None = None,
    phi_edges: np.ndarray | None = None,
    stats: tuple[str, ...] = ("mean",),
):
    """
    Compute radial profiles of volume-weighted angular statistics for a 3D field q(r,theta,phi).
    Volume from reconstructed spherical grid

    Parameters
    ----------
    q_rtp : (nr, ntheta, nphi) array
        Quantity on a spherical grid (may contain NaNs where bins are empty).
    r_centers, theta_centers, phi_centers : 1D arrays
        Bin centers (theta is colatitude in radians).
    r_edges, theta_edges, phi_edges : 1D arrays, optional
        If not provided, edges are inferred from centers.
    stats : tuple of str
        Any of: "mean", "std", "min", "max", "skewness", "kurtosis".
        "mean/std/skewness/kurtosis" are volume-weighted.
        "min/max" are computed over finite values (unweighted).

    Returns
    -------
    dict with keys: "r", plus each requested stat.
    """
    q = np.asarray(q_rtp, dtype=float)
    if q.ndim != 3:
        raise ValueError("q_rtp must be 3D (nr, ntheta, nphi)")

    nr, ntheta, nphi = q.shape
    r_centers = np.asarray(r_centers, dtype=float)
    theta_centers = np.asarray(theta_centers, dtype=float)
    phi_centers = np.asarray(phi_centers, dtype=float)

    if (r_centers.size, theta_centers.size, phi_centers.size) != (nr, ntheta, nphi):
        raise ValueError("q_rtp shape must match sizes of r/theta/phi centers")

    # edges (needed for Δr, Δθ, Δφ)
    r_edges = _centers_to_edges(r_centers) if r_edges is None else np.asarray(r_edges, dtype=float)
    theta_edges = _centers_to_edges(theta_centers) if theta_edges is None else np.asarray(theta_edges, dtype=float)
    phi_edges = _centers_to_edges(phi_centers) if phi_edges is None else np.asarray(phi_edges, dtype=float)

    dr = np.diff(r_edges)            # (nr,)
    dtheta = np.diff(theta_edges)    # (ntheta,)
    dphi = np.diff(phi_edges)        # (nphi,)

    # Build voxel volume weights: r^2 sinθ Δr Δθ Δφ
    # Shape broadcasting to (nr, ntheta, nphi)
    r2 = (r_centers**2)[:, None, None]
    sinth = np.sin(theta_centers)[None, :, None]
    w = r2 * sinth * dr[:, None, None] * dtheta[None, :, None] * dphi[None, None, :]

    out = {"r": r_centers.copy()}
    want = set(stats)

    mean = np.full(nr, np.nan)
    std = np.full(nr, np.nan)
    skew = np.full(nr, np.nan)
    kurt = np.full(nr, np.nan)
    qmin = np.full(nr, np.nan)
    qmax = np.full(nr, np.nan)

    for i in range(nr):
        qi = q[i, :, :].ravel()
        wi = w[i, :, :].ravel()

        mu, sd, sk, ku = _weighted_moments(qi, wi)
        if "mean" in want: mean[i] = mu
        if "std" in want: std[i] = sd
        if "skewness" in want: skew[i] = sk
        if "kurtosis" in want: kurt[i] = ku

        if "min" in want or "max" in want:
            m = np.isfinite(qi)
            if np.any(m):
                if "min" in want: qmin[i] = np.nanmin(qi[m])
                if "max" in want: qmax[i] = np.nanmax(qi[m])

    if "mean" in want: out["mean"] = mean
    if "std" in want: out["std"] = std
    if "min" in want: out["min"] = qmin
    if "max" in want: out["max"] = qmax
    if "skewness" in want: out["skewness"] = skew
    if "kurtosis" in want: out["kurtosis"] = kurt

    return out

def plot_radial_profiles(
    profiles: dict,
    *,
    x_key: str = "r",
    keys: tuple[str, ...] = ("mean",),
    xlabel: str = "r",
    ylabel: str = "Quantity",
    title: str | None = None,
    logx: bool = False,
    logy: bool = False,
    interpolate_nans: bool = False,
):
    """
    Simple matplotlib plot for one or more radial profile curves.

    If interpolate_nans=True, NaN gaps in the curves are linearly
    interpolated for visualization purposes.
    """
    r = np.asarray(profiles[x_key], float)

    plt.figure()
    for k in keys:
        if k not in profiles:
            raise KeyError(f"'{k}' not in profiles. Available: {list(profiles.keys())}")

        y = np.asarray(profiles[k], float)

        if interpolate_nans:
            m = np.isfinite(r) & np.isfinite(y)
            if m.sum() >= 2:
                y = y.copy()
                y[~np.isfinite(y)] = np.interp(r[~np.isfinite(y)], r[m], y[m])

        plt.plot(r, y, label=k)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")
    if len(keys) > 1:
        plt.legend()
    plt.tight_layout()
    plt.show()

"""
from coconut_tools.visualization_3d.reader import cfmesh_to_binned_spherical_grid
from coconut_tools.visualization_3d.coconut_to_numpy import radial_profile_volume_weighted, plot_radial_profiles

(r, th, ph,
 vr, vlon, vclt,
 rho, temp,
 br, blon, bclt) = cfmesh_to_binned_spherical_grid("corona.CFmesh", nr=80, ntheta=90, nphi=180)

prof = radial_profile_volume_weighted(rho, r, th, ph, stats=("mean", "std", "min", "max", "skewness", "kurtosis"))

plot_radial_profiles(
    prof,
    keys=("mean", "std","min","max"),
    xlabel="r [code units or Rsun depending on your input]",
    ylabel=r"$\mu$n [$m^{-3}$]",
    title="Volume-weighted angular stats of rho",
    logy=True,
)

"""

###########################################
# From .CFmesh, conserving same unstructured grid
###########################################

# ----------------------------
# Geometry: vectorized volumes
# ----------------------------

def _tetra_vol_vec(a, b, c, d):
    """
    Vectorized tetrahedron volume for arrays (N,3).
    V = | (b-a) · ((c-a) x (d-a)) | / 6
    https://www.scribd.com/document/539406847/ProduitMixte
    https://www.nagwa.com/fr/videos/906147402968/
    """
    ba = b - a
    ca = c - a
    da = d - a
    return np.abs(np.einsum("ij,ij->i", ba, np.cross(ca, da))) / 6.0


def prism6_volume_vec(v0, v1, v2, v3, v4, v5):
    """
    Vectorized volume for a 6-node triangular prism (wedge), assuming ordering:
      base triangle: v0,v1,v2
      top  triangle: v3,v4,v5
    
    Decomposition into 3 tetrahedra:
      (0,1,2,3), (1,2,4,3), (2,4,5,3)
    """
    return (
        _tetra_vol_vec(v0, v1, v2, v3)
        + _tetra_vol_vec(v1, v2, v4, v3)
        + _tetra_vol_vec(v2, v4, v5, v3)
    )


# ---------------------------------------
# Weighted radial profiles (volume-weight)
# ---------------------------------------

def _weighted_moments_1d(x, w):
    """Weighted mean, std, skewness, kurtosis(excess). Population-weighted."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan, np.nan, np.nan, np.nan

    x = x[m]
    w = w[m]
    wsum = w.sum()
    if wsum <= 0:
        return np.nan, np.nan, np.nan, np.nan

    mu = np.sum(w * x) / wsum
    xc = x - mu
    m2 = np.sum(w * xc**2) / wsum
    std = np.sqrt(m2)

    if not np.isfinite(std) or std == 0.0:
        return mu, std, np.nan, np.nan

    m3 = np.sum(w * xc**3) / wsum
    m4 = np.sum(w * xc**4) / wsum
    skew = m3 / (std**3)
    kurt_excess = m4 / (std**4) - 3.0
    return mu, std, skew, kurt_excess


def radial_profiles_volume2_weighted(
    r,
    q,
    vol,
    *,
    nr=120,
    r_edges=None,
    stats=("mean",),
):
    """
    Radial shell statistics of q using per-cell volume weights.

    stats: any of ("mean","std","min","max","skewness","kurtosis")
      - mean/std/skewness/kurtosis are volume-weighted
      - min/max are plain extrema among finite values
    """
    r = np.asarray(r, float)
    q = np.asarray(q, float)
    vol = np.asarray(vol, float)

    if r_edges is None:
        rmin = np.nanmin(r)
        rmax = np.nanmax(r)
        r_edges = np.linspace(rmin, rmax, nr + 1)
    else:
        r_edges = np.asarray(r_edges, float)
        nr = len(r_edges) - 1

    rc = 0.5 * (r_edges[:-1] + r_edges[1:])
    out = {"r": rc}
    want = set(stats)
    for k in want:
        out[k] = np.full(nr, np.nan)

    idx = np.digitize(r, r_edges) - 1
    inbin = (idx >= 0) & (idx < nr)

    for i in range(nr):
        m = inbin & (idx == i) & np.isfinite(q) & np.isfinite(vol) & (vol > 0)
        if not np.any(m):
            continue
        qi = q[m]
        wi = vol[m]

        mu, sd, sk, ku = _weighted_moments_1d(qi, wi)
        if "mean" in want: out["mean"][i] = mu
        if "std" in want: out["std"][i] = sd
        if "skewness" in want: out["skewness"][i] = sk
        if "kurtosis" in want: out["kurtosis"][i] = ku

        if "min" in want: out["min"][i] = np.nanmin(qi)
        if "max" in want: out["max"][i] = np.nanmax(qi)

    return out


# ---------------------------------------
# CFmesh reader (generalized + vectorized)
# ---------------------------------------

def read_cfmesh_cells(
    inputfile: str,
    *,
    readstruct_fn,
    extra_field_names: list[str] | None = None,
):
    """
    Generalized CFmesh cell reader for 6-node Prism elements.

    Returns
    -------
    centers : (N,3)
    r : (N,)
    volumes : (N,)
    fields : dict
        rho, p, T, Vx,Vy,Vz, Bx,By,Bz, vr,vlon,vclt, br,blon,bclt, x,y,z
    + extras if necessary
    """
    with open(inputfile, "r") as f:
        lines = f.readlines()

    # readstruct(lines) is from coconut_tools.toheliosphere.create_dat
    idx0, idx1, idx2, idx3, nbelements, nend, comment = readstruct_fn(lines)

    connectivity = np.loadtxt(lines[idx0:idx0 + nbelements], dtype=int)
    coordinates = np.loadtxt(lines[idx1:idx2 - 1], dtype=float)

    nodes = connectivity[:, :6]
    verts = coordinates[nodes]                    # (N,6,3)

    centers = verts.mean(axis=1)
    x, y, z = centers.T
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(z / r)                     # [0, π]
    phi = np.arctan2(y, x)
    phi[phi < 0] += 2*np.pi                      # force into [0, 2π]

    # Vectorized volumes
    v0, v1, v2, v3, v4, v5 = (verts[:, i, :] for i in range(6))
    volumes = prism6_volume_vec(v0, v1, v2, v3, v4, v5)

    # Initial data/state block (same indexing logic as coconut_tools)
    bd = comment[-nend - 1][0] + 1
    bf = comment[-nend][0]
    Init = np.loadtxt(lines[bd:bf], dtype=np.float64)
    ncols = Init.shape[1]

    # Scalars (same scaling as your reader.py)
    rho = Init[:, 0] * 1.67e-13 / 1.67e-27      # n'=rho/mp=mu*n [m^-3]
    Vx  = Init[:, 1] * 480248.0                 # [m/s]
    Vy  = Init[:, 2] * 480248.0
    Vz  = Init[:, 3] * 480248.0
    Bx  = Init[:, 4] * 2.2e-4                   # [T]
    By  = Init[:, 5] * 2.2e-4
    Bz  = Init[:, 6] * 2.2e-4
    p   = Init[:, 7] * 0.03851                  # [Pa]
    T   = p / rho / 2.0 / 1.38e-23              # [K] (mu=0.5 hardcoded)
    phi_div = Init[:, 8] * 480248.0 * 2.2e-4        # [V/m] 

    # Correct spherical projections (consistent with x^2+y^2 = r_xy^2)
    r_xy = np.hypot(x, y)
    eps = 1e-12
    
    # v_r
    vr = (x*Vx + y*Vy + z*Vz) / (r + eps)
    
    # v_phi (longitude)
    vlon = (-y*Vx + x*Vy) / (r_xy + eps)
    
    # v_theta (colatitude direction) :
    # v_theta = (x z Vx + y z Vy - (x^2+y^2) Vz) / (r * sqrt(x^2+y^2))
    vclt = (x*z*Vx + y*z*Vy - (r_xy**2)*Vz) / ((r + eps) * (r_xy + eps))

    br = (x*Bx + y*By + z*Bz) / (r + eps)
    blon = (-y*Bx + x*By) / (r_xy + eps)
    bclt = (x*z*Bx + y*z*By - (r_xy**2)*Bz) / ((r + eps) * (r_xy + eps))

    fields = {
        "rho": rho, "p": p, "T": T, "phi_div": phi_div,
        "Vx": Vx, "Vy": Vy, "Vz": Vz,
        "Bx": Bx, "By": By, "Bz": Bz,
        "vr": vr, "vlon": vlon, "vclt": vclt,
        "br": br, "blon": blon, "bclt": bclt,
        "x": x, "y": y, "z": z,
        "r": r, "theta": theta, "phi": phi,
    }

    # If extra fields in CFmesh
    if ncols > 9:
        extra_field_names = [] if extra_field_names is None else list(extra_field_names)

        for j in range(9, ncols):
            if (j - 9) < len(extra_field_names):
                name = extra_field_names[j - 9]
            else:
                name = f"extra_{j}"
            fields[name] = Init[:, j]

    return centers, r, volumes, fields


"""
from coconut_tools.visualization_3d.coconut_to_numpy import read_cfmesh_cells, radial_profiles_volume2_weighted, plot_radial_profiles

from coconut_tools.toheliosphere.create_dat import readstruct

centers, r, vol, fields = read_cfmesh_cells("corona.CFmesh", readstruct_fn=readstruct)

prof = radial_profiles_volume2_weighted(
    r, fields["rho"], vol,
    nr=120,
    stats=("mean", "std", "min", "max", "skewness", "kurtosis"),
)

plot_radial_profiles(
    prof,
    keys=("mean", "std","min","max"),
    xlabel="r [code units or Rsun depending on your input]",
    ylabel=r"$\mu$n [$m^{-3}$]",
    title="Volume-weighted angular stats of rho",
    logy=True,
    interpolate_nans=True
)

profT = radial_profiles_volume2_weighted(
    r, fields["T"], vol,
    nr=120,
    stats=("mean", "std", "min", "max", "skewness", "kurtosis"),
)

plot_radial_profiles(
    profT,
    keys=("mean", "std"),
    xlabel="r [code units]",
    ylabel="T [K]",
    title="Volume-weighted radial stats of T",
    logy=True,
)
"""

### Compute local gradients on the unstructured CFmesh grid, using least-squares from neighbors defined by shared nodes.

def _read_cfmesh_connectivity(inputfile: str, readstruct_fn):
    with open(inputfile, "r") as f:
        lines = f.readlines()
    idx0, idx1, idx2, idx3, nbelements, nend, comment = readstruct_fn(lines)
    return np.loadtxt(lines[idx0:idx0 + nbelements], dtype=int)


def _cell_neighbors_from_connectivity(
    connectivity: np.ndarray,
    *,
    min_shared_nodes: int = 3,
    max_neighbors: int = 20,
):
    """Build a neighbor list by shared CFmesh node counts."""
    node_to_cells: dict[int, list[int]] = defaultdict(list)
    for cell_index, nodes in enumerate(connectivity[:, :6]):
        for node in nodes:
            node_to_cells[int(node)].append(cell_index)

    neighbors: list[np.ndarray] = []
    for cell_index, nodes in enumerate(connectivity[:, :6]):
        counts: dict[int, int] = {}
        for node in nodes:
            for neighbor_cell in node_to_cells[int(node)]:
                if neighbor_cell == cell_index:
                    continue
                counts[neighbor_cell] = counts.get(neighbor_cell, 0) + 1

        neighbor_cells = [
            neighbor_cell
            for neighbor_cell, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            if count >= min_shared_nodes
        ]
        neighbors.append(np.asarray(neighbor_cells[:max_neighbors], dtype=int))
    return neighbors


def _least_squares_gradients(
    points,
    values,
    neighbors,
    *,
    min_neighbors: int = 4,
    rcond=None,
):
    """Compute local least-squares gradients in Cartesian coordinates."""
    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float)

    if values.ndim == 1:
        values = values[:, None]

    n_cells = points.shape[0]
    n_components = values.shape[1]
    gradients = np.full((n_cells, n_components, 3), np.nan)

    for i, neigh in enumerate(neighbors):
        if neigh.size < min_neighbors:
            continue

        delta_x = points[neigh] - points[i]
        delta_q = values[neigh] - values[i]

        finite_mask = np.isfinite(delta_x).all(axis=1) & np.isfinite(delta_q).all(axis=1)
        if finite_mask.sum() < min_neighbors:
            continue

        A = delta_x[finite_mask]
        B = delta_q[finite_mask]
        if B.ndim == 1:
            B = B[:, None]

        coef, *_ = np.linalg.lstsq(A, B, rcond=rcond)
        gradients[i] = coef.T

    if gradients.shape[1] == 1:
        return gradients[:, 0, :]
    return gradients


def compute_cfmesh_unstructured_cartesian_gradients(
    inputfile: str,
    *,
    readstruct_fn,
    field_names: list[str] | None = None,
    min_shared_nodes: int = 3,
    max_neighbors: int = 20,
    min_neighbors: int = 4,
):
    """Compute least-squares gradients on CFmesh cell centers in Cartesian space.

    The mesh stencil is inferred from shared CFmesh node connectivity.
    Gradients are returned in the global Cartesian basis (d/dx, d/dy, d/dz).
    """
    centers, _, _, fields = read_cfmesh_cells(inputfile, readstruct_fn=readstruct_fn)
    connectivity = _read_cfmesh_connectivity(inputfile, readstruct_fn=readstruct_fn)
    neighbors = _cell_neighbors_from_connectivity(
        connectivity,
        min_shared_nodes=min_shared_nodes,
        max_neighbors=max_neighbors,
    )

    if field_names is None:
        field_names = [
            k for k in fields.keys()
            if k not in ("x", "y", "z", "r", "theta", "phi")
        ]

    gradients = {}
    for name in field_names:
        if name not in fields:
            raise KeyError(f"Field '{name}' not found in CFmesh fields.")
        gradients[name] = _least_squares_gradients(
            centers,
            fields[name],
            neighbors,
            min_neighbors=min_neighbors,
        )

    return centers, gradients


def _spherical_grid_bin_centers_to_cartesian(
    r_centers: np.ndarray,
    theta_centers: np.ndarray,
    phi_centers: np.ndarray,
):
    r = r_centers[:, None, None]
    theta = theta_centers[None, :, None]
    phi = phi_centers[None, None, :]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=-1)


def _spherical_basis_vectors(theta: np.ndarray, phi: np.ndarray):
    """Return orthonormal spherical basis vectors at given angles."""
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    sin_th = np.sin(theta)
    cos_th = np.cos(theta)
    sin_ph = np.sin(phi)
    cos_ph = np.cos(phi)

    e_r = np.stack([sin_th * cos_ph, sin_th * sin_ph, cos_th], axis=-1)
    e_theta = np.stack([cos_th * cos_ph, cos_th * sin_ph, -sin_th], axis=-1)
    e_phi = np.stack([-sin_ph, cos_ph, np.zeros_like(theta)], axis=-1)
    return e_r, e_theta, e_phi


def cartesian_gradients_to_spherical(
    grad_cart: np.ndarray,
    *,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    z: np.ndarray | None = None,
    r: np.ndarray | None = None,
    theta: np.ndarray | None = None,
    phi: np.ndarray | None = None,
):
    """Convert Cartesian gradient components into spherical basis components.

    The returned gradient has the same shape as `grad_cart`, with the last axis
    ordered as (g_r, g_theta, g_phi) in the local spherical basis.
    """
    grad_cart = np.asarray(grad_cart, dtype=float)
    if grad_cart.ndim < 1 or grad_cart.shape[-1] != 3:
        raise ValueError("grad_cart must have shape (..., 3)")

    if x is not None or y is not None or z is not None:
        if x is None or y is None or z is None:
            raise ValueError("x, y, z must be provided together")
        x, y, z = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float))
        theta, phi = _cart_to_spherical(x, y, z)[1:]
    elif r is not None or theta is not None or phi is not None:
        if r is None or theta is None or phi is None:
            raise ValueError("r, theta, phi must be provided together")
        _, theta, phi = np.broadcast_arrays(
            np.asarray(r, dtype=float),
            np.asarray(theta, dtype=float),
            np.asarray(phi, dtype=float),
        )
    else:
        raise ValueError("Either x/y/z or r/theta/phi must be provided")

    basis = _spherical_basis_vectors(theta, phi)
    grad_sph = np.stack([np.sum(grad_cart * b, axis=-1) for b in basis], axis=-1)
    return grad_sph


def compute_cfmesh_unstructured_spherical_gradients(
    inputfile: str,
    *,
    readstruct_fn,
    field_names: list[str] | None = None,
    min_shared_nodes: int = 3,
    max_neighbors: int = 20,
    min_neighbors: int = 4,
):
    """Compute spherical-basis gradients at CFmesh cell centers.

    This wraps `compute_cfmesh_unstructured_cartesian_gradients` and converts
    the returned gradients from global Cartesian into local spherical basis.
    """
    centers, cart_gradients = compute_cfmesh_unstructured_cartesian_gradients(
        inputfile,
        readstruct_fn=readstruct_fn,
        field_names=field_names,
        min_shared_nodes=min_shared_nodes,
        max_neighbors=max_neighbors,
        min_neighbors=min_neighbors,
    )
    x, y, z = centers.T
    spherical_gradients = {
        name: cartesian_gradients_to_spherical(
            grad,
            x=x,
            y=y,
            z=z,
        )
        for name, grad in cart_gradients.items()
    }
    return centers, spherical_gradients


def compute_binned_spherical_grid_spherical_gradients(
    r_centers: np.ndarray,
    theta_centers: np.ndarray,
    phi_centers: np.ndarray,
    q_rtp: np.ndarray,
    *,
    min_neighbors: int = 4,
    periodic_phi: bool = True,
):
    """Compute spherical-basis gradients from binned spherical grid output."""
    cart_grad = compute_binned_spherical_grid_cartesian_gradients(
        r_centers,
        theta_centers,
        phi_centers,
        q_rtp,
        min_neighbors=min_neighbors,
        periodic_phi=periodic_phi,
    )
    r = r_centers[:, None, None]
    theta = theta_centers[None, :, None]
    phi = phi_centers[None, None, :]
    spherical_grad = cartesian_gradients_to_spherical(
        cart_grad,
        r=r,
        theta=theta,
        phi=phi,
    )
    return spherical_grad


def compute_binned_spherical_grid_cartesian_gradients(
    r_centers: np.ndarray,
    theta_centers: np.ndarray,
    phi_centers: np.ndarray,
    q_rtp: np.ndarray,
    *,
    min_neighbors: int = 4,
    periodic_phi: bool = True,
):
    """Compute Cartesian gradients from binned spherical grid output.

    This uses a local least-squares fit over the structured spherical stencil,
    but returns gradients in global Cartesian coordinates.
    """
    q = np.asarray(q_rtp, dtype=float)
    if q.ndim != 3:
        raise ValueError("q_rtp must be a 3D array with shape (nr, ntheta, nphi).")

    coords = _spherical_grid_bin_centers_to_cartesian(r_centers, theta_centers, phi_centers)
    grad = np.full(q.shape + (3,), np.nan)
    nr, ntheta, nphi = q.shape
    offsets = [
        (dr, dt, dphi)
        for dr in (-1, 0, 1)
        for dt in (-1, 0, 1)
        for dphi in (-1, 0, 1)
        if not (dr == dt == dphi == 0)
    ]

    for ir in range(nr):
        for ith in range(ntheta):
            for iph in range(nphi):
                if not np.isfinite(q[ir, ith, iph]):
                    continue

                neighbors = []
                values = []
                for dr, dt, dphi in offsets:
                    jr = ir + dr
                    jth = ith + dt
                    jph = iph + dphi

                    if jr < 0 or jr >= nr or jth < 0 or jth >= ntheta:
                        continue
                    if periodic_phi:
                        jph %= nphi
                    elif jph < 0 or jph >= nphi:
                        continue

                    if not np.isfinite(q[jr, jth, jph]):
                        continue
                    neighbors.append(coords[jr, jth, jph])
                    values.append(q[jr, jth, jph])

                if len(neighbors) < min_neighbors:
                    continue

                A = np.asarray(neighbors, dtype=float) - coords[ir, ith, iph]
                B = np.asarray(values, dtype=float) - q[ir, ith, iph]
                if B.ndim == 1:
                    B = B[:, None]

                coef, *_ = np.linalg.lstsq(A, B, rcond=None)
                grad[ir, ith, iph] = coef[:, 0]

    return grad

######
# 2D histo helper
######

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def hist2d_pdf(
    x,
    y,
    *,
    bins_x=200,
    bins_y=200,
    range_x=None,
    range_y=None,
    logx=False,
    logy=False,
    weights=None,
    density=True,
    drop_nonfinite=True,
):
    """
    Compute a 2D histogram of (x,y) with optional normalization to a PDF.

    Notes
    -----
    - If density=True, returns a *probability density* such that:
        sum(H * dx * dy) ~ 1
      (like numpy.histogram2d / matplotlib.hist2d with density=True).
    - If logx/logy=True, bins are logarithmic (x>0 / y>0 required).
    - weights can be used for volume-weighting (e.g. pass `vol`).

    Returns
    -------
    H : (nx, ny) array
    xedges : (nx+1,) array
    yedges : (ny+1,) array
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()

    if weights is not None:
        weights = np.asarray(weights, float).ravel()
        if weights.shape != x.shape:
            raise ValueError("weights must have same shape as x and y")

    m = np.ones_like(x, dtype=bool)
    if drop_nonfinite:
        m &= np.isfinite(x) & np.isfinite(y)
        if weights is not None:
            m &= np.isfinite(weights)

    x = x[m]
    y = y[m]
    if weights is not None:
        weights = weights[m]

    if x.size == 0:
        raise ValueError("No valid samples after filtering")

    # Bin edges
    def _edges(v, bins, rng, log):
        if rng is None:
            vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
        else:
            vmin, vmax = rng
        if log:
            if vmin <= 0:
                # If user didn't provide range, try to infer from positive values
                vp = v[v > 0]
                if vp.size == 0:
                    raise ValueError("log bins requested but no positive values")
                vmin = float(np.nanmin(vp))
            if vmax <= 0:
                raise ValueError("log bins requested but max <= 0")
            return np.geomspace(vmin, vmax, int(bins) + 1)
        else:
            return np.linspace(vmin, vmax, int(bins) + 1)

    xedges = _edges(x, bins_x, range_x, logx)
    yedges = _edges(y, bins_y, range_y, logy)

    H, xedges, yedges = np.histogram2d(
        x, y,
        bins=[xedges, yedges],
        weights=weights,
        density=density,
    )
    return H, xedges, yedges


def plot_hist2d_pdf(
    H, xedges, yedges,
    *,
    ax=None,
    cmap="viridis",
    lognorm=True,
    vmin=None,
    vmax=None,
    xlabel="x",
    ylabel="y",
    title=None,
    colorbar=True,
):
    """
    Plot a 2D histogram returned by hist2d_pdf using pcolormesh.
    """
    if ax is None:
        fig, ax = plt.subplots()

    norm = None
    if lognorm:
        # Avoid log(0): mask zeros
        H_plot = np.ma.masked_where(H <= 0, H)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        H_plot = H

    mesh = ax.pcolormesh(xedges, yedges, H_plot.T, shading="auto", cmap=cmap, norm=norm)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if colorbar:
        cb = plt.colorbar(mesh, ax=ax, pad=0.02)
        cb.set_label("PDF" if norm is not None else "Counts" )
    return ax

"""
import coconut_tools.visualization_3d.coconut_to_numpy as ctn
from coconut_tools.toheliosphere.create_dat import readstruct
centers, r, vol, fields = ctn.read_cfmesh_cells("corona.CFmesh", readstruct_fn=readstruct)
H, xed, yed = ctn.hist2d_pdf(fields["r"], fields["rho"], bins_x=3000, bins_y=300, weights=vol, density=True,logy=True)
plt.ion()
ctn.plot_hist2d_pdf(H, xed, yed,
                xlabel='r [Rsun]',ylabel=r"$\rho$ [m$^{-3}$]",
                title="Volume-weighted PDF: rho vs r",
                lognorm=True)
plt.yscale('log')
plt.xscale('log')
"""

"""
logger.info("Testing cfmesh_to_binned_spherical_grid.")
    cfmesh_to_binned_spherical_grid(
        inputfile=str(cfmesh_path),
        nr=30,
        ntheta=60,
        nphi=120,
        auto_resolution=False,
    )

    logger.info("Testing _auto_spherical_binning_resolution and _auto_resolution_nn.")
    with open(cfmesh_path, "r") as f:
        lines = f.readlines()
    idx0, idx1, idx2, idx3, nbelements, nend, comment = readstruct(lines)
    connectivity = np.loadtxt(lines[idx0:idx0 + nbelements], dtype=int)
    coordinates = np.loadtxt(lines[idx1:idx2 - 1], dtype=float)
    nodes = connectivity[:, :6]
    centers = coordinates[nodes].mean(axis=1)
    x, y, z = centers.T
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    phi[phi < 0] += 2 * np.pi

    nr, ntheta, nphi = _auto_spherical_binning_resolution(r, theta, phi)
    logger.info("Auto spherical binning: nr=%s ntheta=%s nphi=%s", nr, ntheta, nphi)

    nnr, nntheta, nnphi = _auto_resolution_nn(
        r=r,
        theta=theta,
        phi=phi,
        r_min=float(r.min()),
        r_max=float(r.max()),
    )
    logger.info(
        "Auto NN binning: nr=%s ntheta=%s nphi=%s",
        nnr,
        nntheta,
        nnphi,
    )
"""
