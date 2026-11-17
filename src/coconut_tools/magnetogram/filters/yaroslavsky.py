"""
Low-level local weighted smoothing filter used by Yaroslavsky filtering.

Performs nonlinear weighted averaging of each pixel's neighborhood using Gaussian-like spatial and radiometric terms.
Uses parallel processing for efficiency.

Author: Jose Murteira
Cleaned and modularized by: Luis
"""

import math
import numpy as np
import scipy.integrate
import time
import datetime
from multiprocessing import Pool
from coconut_tools.tools.logger_config import setup_logger

logger = setup_logger(__name__)

def Th(u, i, j, Rn, h, dx, dy):
    y_mask = math.floor(Rn / dy)
    x_mask = math.floor(Rn / dx)

    start_i = max(0, i - y_mask)
    end_i = min(u.shape[0], i + y_mask + 1)
    start_j = max(0, j - x_mask)
    end_j = min(u.shape[1], j + x_mask + 1)

    u_mask = u[start_i:end_i, start_j:end_j]
    T = np.zeros_like(u_mask)
    T_norm = np.zeros_like(u_mask)

    for ii in range(u_mask.shape[0]):
        for jj in range(u_mask.shape[1]):
            r_sq = ((start_i + ii - i)**2 * dy**2 + (start_j + jj - j)**2 * dx**2)
            if r_sq <= Rn**2:
                diff = (u_mask[ii, jj] - u[i, j]) / h
                T_norm[ii, jj] = math.exp(-diff**2)
                T[ii, jj] = T_norm[ii, jj] * u_mask[ii, jj]
    return T, T_norm

def main_loop_integration(u, i, j, Rn, h, dx, dy):
    T_array, T_norm = Th(u, i, j, Rn, h, dx, dy)
    Ix_2 = scipy.integrate.simpson(T_norm, dx=dx)
    N = scipy.integrate.simpson(Ix_2, dx=dy)
    Ix = scipy.integrate.simpson(T_array, dx=dx)

    return scipy.integrate.simpson(Ix, dx=dy) / N if N != 0 else u[i, j]

def filter3(image: np.ndarray, dx: float, dy: float, alpha: float, Rn: float, image_seq=None):
    """
    Apply local nonlinear averaging filter based on spatial and intensity similarity.

    Args:
        image (np.ndarray): Input 2D image.
        dx (float): Spatial resolution in x-direction.
        dy (float): Spatial resolution in y-direction.
        alpha (float): Exponent controlling the radiometric kernel width
            following the article implementation (h = Rn^alpha, before Rn is
            converted from grid-spacing units to physical length).
        Rn (float): Radius of influence (neighborhood size).
        image_seq (list, optional): If provided, intermediate outputs are saved.

    Returns:
        np.ndarray: Smoothed image.
    """
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be positive.")
    if Rn <= 0:
        raise ValueError("Rn must be positive.")
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")

    u = np.asarray(image, dtype=float).copy()
    if u.ndim == 3 and u.shape[2] == 1:
        u = u[:, :, 0]
    if u.ndim != 2:
        raise ValueError("image must be a 2D array.")
    if not np.all(np.isfinite(u)):
        raise ValueError("image must contain only finite values.")
    if image_seq is not None:
        image_seq.append(u.copy())

    h = Rn ** alpha
    Rn *= max(dx, dy)
    u_new = np.zeros_like(u)    

    logger.info(f"Starting local filter with shape {u.shape}, Rn={Rn:.3f}, h={h:.3f}")
    time_start = time.time()

    tasks = [(u, i, j, Rn, h, dx, dy) for i in range(u.shape[0]) for j in range(u.shape[1])]
    try:
        with Pool() as pool:
            results = pool.starmap(main_loop_integration, tasks)
    except OSError as exc:
        logger.warning(f"Multiprocessing unavailable ({exc}); falling back to serial filtering.")
        results = [main_loop_integration(*task) for task in tasks]

    u_new = np.array(results).reshape(u.shape)
    logger.info(f"Filtering completed in {datetime.timedelta(seconds=time.time() - time_start)}")
    return u_new

# ---------------------------------------------------------------------------
# Spherical (physical) Yaroslavsky filter
# ---------------------------------------------------------------------------

YAROSLAVSKY_OFFSET_CHUNK = 256
"""Maximum number of longitude offsets processed at once (memory bound)."""


def spherical_yaroslavsky(
    Br: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    radius_deg: float,
    h_gauss: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Yaroslavsky neighbourhood filter on the sphere.

    For every cell ``i`` the output is the weighted mean of the cells ``k``
    whose centre lies within the great-circle distance ``radius_deg`` (sharp
    disk, as in the original filter, including ``i`` itself):

    ``B_out[i] = sum_k w_ik B[k] / sum_k w_ik``,
    ``w_ik = area_k * exp(-((B[k] - B[i]) / h)^2)``.

    The disk has the same physical size for every product, resolution and
    latitude, is periodic in longitude and continuous across the poles. The
    weights are plain area-weighted sums (no Simpson integration). The filter
    is not flux conserving by nature; use the flux correction after the filter.

    ``h`` (field units, Gauss) is either given, or estimated as the noise
    level ``1.4826 * MAD(B)`` with an area-weighted median, excluding cells
    where the field is exactly zero (e.g. NaN polar caps set to 0).

    Args:
        Br: Radial field ``(theta.size, phi.size)``.
        phi: Uniform increasing longitude centres (radians), endpoint-free.
        theta: Increasing colatitude centres (radians), any spacing.
        radius_deg: Disk radius in heliographic degrees (> 0).
        h_gauss: Radiometric scale in Gauss, or None for the automatic
            noise estimate.

    Returns:
        ``(filtered, info)`` with ``info`` keys ``h_gauss``, ``h_source``
        (``"auto"`` or ``"config"``), ``radius_deg``.

    Raises:
        ValueError: Invalid inputs or non-uniform longitude axis.
    """
    from coconut_tools.magnetogram.core.coordinates import spherical_pixel_areas
    from coconut_tools.magnetogram.filters.nonlinear_diffusion import (
        MAD_TO_SIGMA,
        _weighted_median,
    )

    field = np.asarray(Br, dtype=float)
    phi = np.asarray(phi, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if field.ndim != 2 or field.shape != (theta.size, phi.size):
        raise ValueError("Br must be 2D with shape (theta.size, phi.size).")
    if not (np.all(np.isfinite(field)) and np.all(np.isfinite(phi)) and np.all(np.isfinite(theta))):
        raise ValueError("Br, theta, and phi must contain only finite values.")
    if not np.isfinite(radius_deg) or radius_deg <= 0.0:
        raise ValueError("radius_deg must be a finite positive number.")
    if h_gauss is not None and (not np.isfinite(h_gauss) or h_gauss <= 0.0):
        raise ValueError("h_gauss must be a finite positive number.")

    n_theta, n_phi = field.shape
    dphi = 2.0 * np.pi / n_phi
    if n_phi > 1 and not np.allclose(np.diff(np.unwrap(phi)), dphi, atol=1.0e-9, rtol=0.0):
        raise ValueError("Spherical Yaroslavsky requires a uniform, complete longitude axis.")

    row_area = spherical_pixel_areas(theta, phi, field.shape)[:, 0]
    if h_gauss is None:
        valid = field != 0.0
        if not np.any(valid):
            raise ValueError("Cannot estimate h: the field is identically zero.")
        values = field[valid]
        weights = np.broadcast_to(row_area[:, None], field.shape)[valid]
        centre = _weighted_median(values, weights)
        h = MAD_TO_SIGMA * _weighted_median(np.abs(values - centre), weights)
        if h <= 0.0:
            raise ValueError("Automatic h is zero; give h_gauss explicitly.")
        h_source = "auto"
    else:
        h = float(h_gauss)
        h_source = "config"
    logger.info(
        "Spherical Yaroslavsky: radius = %.4g deg, h = %.6g G (%s)",
        radius_deg, h, h_source,
    )

    radius = np.radians(radius_deg)
    cos_radius = np.cos(min(radius, np.pi))
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    columns = np.arange(n_phi)
    numerator = np.zeros_like(field)
    denominator = np.zeros_like(field)

    for i in range(n_theta):
        centre_values = field[i]
        rows = np.nonzero(np.abs(theta - theta[i]) <= radius)[0]
        for k in rows:
            # cos(d) = cos_i cos_k + sin_i sin_k cos(m dphi) >= cos(R)
            # <=> cos(m dphi) >= threshold: |m| <= m_max (contiguous offsets).
            product = sin_theta[i] * sin_theta[k]
            if product <= 0.0:
                threshold = -np.inf
            else:
                threshold = (cos_radius - cos_theta[i] * cos_theta[k]) / product
            if threshold > 1.0 + 1.0e-12:
                continue
            if threshold <= -1.0:
                offsets = columns - n_phi // 2
            else:
                m_max = int(np.floor(np.arccos(min(threshold, 1.0)) / dphi + 1.0e-9))
                m_max = min(m_max, (n_phi - 1) // 2)
                offsets = np.arange(-m_max, m_max + 1)
            source = field[k]
            for start in range(0, offsets.size, YAROSLAVSKY_OFFSET_CHUNK):
                chunk = offsets[start:start + YAROSLAVSKY_OFFSET_CHUNK]
                neighbour = source[(columns[None, :] + chunk[:, None]) % n_phi]
                weight = row_area[k] * np.exp(-(((neighbour - centre_values[None, :]) / h) ** 2))
                numerator[i] += np.sum(weight * neighbour, axis=0)
                denominator[i] += np.sum(weight, axis=0)

    filtered = numerator / denominator
    return filtered, {"h_gauss": h, "h_source": h_source, "radius_deg": float(radius_deg)}


if __name__ == "__main__":
    test_image = np.random.rand(64, 64)
    result = filter3(test_image, dx=1.0, dy=1.0, alpha=1.5, Rn=5.0)
    logger.info("Test run complete.")
