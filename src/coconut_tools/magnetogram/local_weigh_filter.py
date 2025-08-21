"""
Local Weighted Smoothing Filter

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
from logger_config import setup_logger

logger = setup_logger(__name__)

def Th(u, i, j, Rn, h, dx, dy):
    x_mask = math.floor(Rn / dx)
    y_mask = math.floor(Rn / dy)

    start_i = max(0, i - x_mask)
    end_i = min(u.shape[0], i + x_mask + 1)
    start_j = max(0, j - y_mask)
    end_j = min(u.shape[1], j + y_mask + 1)

    u_mask = u[start_i:end_i, start_j:end_j]
    T = np.zeros_like(u_mask)
    T_norm = np.zeros_like(u_mask)

    for ii in range(u_mask.shape[0]):
        for jj in range(u_mask.shape[1]):
            r_sq = ((start_i + ii - i)**2 * dx + (start_j + jj - j)**2 * dy)
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
        alpha (float): Exponent controlling the kernel width (h = Rn^alpha).
        Rn (float): Radius of influence (neighborhood size).
        image_seq (list, optional): If provided, intermediate outputs are saved.

    Returns:
        np.ndarray: Smoothed image.
    """
    u = np.copy(image)
    if u.ndim == 3 and u.shape[2] == 1:
        u = u[:, :, 0]
    if image_seq is not None:
        image_seq.append(u.copy())

    Rn *= max(dx, dy)
    h = Rn ** alpha
    u_new = np.zeros_like(u)

    logger.info(f"Starting local filter with shape {u.shape}, Rn={Rn:.3f}, h={h:.3f}")
    time_start = time.time()

    with Pool() as pool:
        results = pool.starmap(
            main_loop_integration,
            [(u, i, j, Rn, h, dx, dy) for i in range(u.shape[0]) for j in range(u.shape[1])]
        )

    u_new = np.array(results).reshape(u.shape)
    logger.info(f"Filtering completed in {datetime.timedelta(seconds=time.time() - time_start)}")
    return u_new

if __name__ == "__main__":
    test_image = np.random.rand(64, 64)
    result = filter3(test_image, dx=1.0, dy=1.0, alpha=1.5, Rn=5.0)
    logger.info("Test run complete.")
