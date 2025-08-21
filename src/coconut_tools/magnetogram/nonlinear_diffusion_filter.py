"""
Nonlinear diffusion filter based on Perona-Malik edge-preserving smoothing.

Author: Jose Murteira
Cleaned, modularized and refactored by: Luis
"""
import math
import numpy as np
import scipy.signal
import scipy.sparse
import scipy.sparse.linalg
import scipy.stats
import time
import datetime
from logger_config import setup_logger

logger = setup_logger(__name__)

def nonlinearDiffusionFilter(image: np.ndarray, dx: float, dy: float, iterations: int, tau: float = 1.0, image_seq=None):
    """
    Apply nonlinear isotropic diffusion filtering to a 2D image using Perona-Malik method.

    Args:
        image (np.ndarray): 2D array representing the image to be filtered.
        dx (float): Pixel spacing in x-direction.
        dy (float): Pixel spacing in y-direction.
        iterations (int): Number of iterations.
        tau (float): Time step for the iteration. Default is 1.0.
        image_seq (list, optional): If provided, saves intermediate results here.

    Returns:
        Tuple[np.ndarray, float]: Filtered image and final timestep used.
    """

    def gradU(u):
        gradkernelx = 0.5 / dx * np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
        gradkernely = 0.5 / dy * np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])
        gradx = scipy.signal.convolve2d(u, gradkernelx, boundary='symm')
        grady = scipy.signal.convolve2d(u, gradkernely, boundary='symm')
        return gradx, grady

    def computeDiffusivity(u, lamb):
        gradx, grady = gradU(u)
        gradm2 = gradx ** 2 + grady ** 2
        return 1.0 / np.sqrt(1.0 + gradm2 / (lamb * lamb))

    def computef_matrix(u, g, shape):
        g = np.pad(g, pad_width=1, mode='constant')
        f_matrix = scipy.sparse.lil_matrix((u.shape[0], u.shape[0]))

        rows, cols = shape
        for i in range(cols):
            for j in range(rows):
                k = j * cols + i

                g_ip = math.sqrt(g[j + 1, i] * g[j, i])
                g_in = math.sqrt(g[j - 1, 1] * g[j, i])

                if i == cols - 1:
                    g_pj = math.sqrt(g[j, 0] * g[j, i])
                    f_matrix[k, k - cols - 1] = g_pj * 1 / (dx ** 2)
                else:
                    g_pj = math.sqrt(g[j, i + 1] * g[j, i])
                    f_matrix[k, k + 1] = g_pj * 1 / (dx ** 2)

                if i == 0:
                    g_nj = math.sqrt(g[j, cols - 1] * g[j, i])
                    f_matrix[k, k + cols - 1] = g_nj * 1 / (dx ** 2)
                else:
                    g_nj = math.sqrt(g[j, i - 1] * g[j, i])
                    f_matrix[k, k - 1] = g_nj * 1 / (dx ** 2)

                f_matrix[k, k] = (-g_pj - g_nj) * 1 / (dx ** 2)

                if j != 0:
                    f_matrix[k, k - cols] = g_in * 1 / (dy ** 2)  # from above
                if j != rows - 1:
                    f_matrix[k, k + cols] = g_ip * 1 / (dy ** 2)  # from bellow
                if j != rows - 1 and j != 0:
                    f_matrix[k, k] = f_matrix[k, k] + (-g_ip - g_in) * 1 / (dy ** 2)
                elif j != rows - 1:  # it's in the top boundary
                    f_matrix[k, k] = f_matrix[k, k] + (-g_ip) * 1 / (dy ** 2)
                else:  # it's in the bottom boundary
                    f_matrix[k, k] = f_matrix[k, k] + (-g_in) * 1 / (dy ** 2)
        return f_matrix.tocsr()

    def compute_matrix_system(u, u_0, tau, lamb, f_0, shape):
        g = computeDiffusivity(u.reshape(shape), lamb)
        f = computef_matrix(u, g, shape)
        A = scipy.sparse.identity(u.shape[0]) - tau * 0.5 * f
        b = u_0 + tau * 0.5 * f_0.dot(u_0)
        return A, b, f

    u = image.astype(float).copy()
    if len(u.shape) == 3 and u.shape[2] == 1:
        u = u[:, :, 0]
    if image_seq is not None:
        image_seq.append(u.copy())

    shape = u.shape
    gradx, grady = gradU(u)
    lamb = 1.4826 * abs(np.median([gradx, grady]) - np.median(np.sqrt(gradx ** 2 + grady ** 2)))
    logger.info(f"Lambda estimate: {lamb:.3e}")

    u = u.reshape(-1)
    f_0 = computef_matrix(u, computeDiffusivity(u.reshape(shape), lamb), shape)
    u_0 = u.copy()

    for i in range(iterations):
        logger.info(f"Iteration {i + 1}/{iterations}")
        start = time.time()
        A, b, f_0 = compute_matrix_system(u, u_0, tau, lamb, f_0, shape)
        u, exit_code = scipy.sparse.linalg.gmres(A, b, atol=1e-5)

        res = np.log10(np.linalg.norm(u - u_0))
        if image_seq is not None:
            image_seq.append(u.copy())

        duration = time.time() - start
        eta = datetime.timedelta(seconds=(duration * (iterations - i - 1)))
        logger.info(f"Converged?: {exit_code}, Δt: {duration:.2f}s, ETA: {eta}, residual: {res:.3f}")

        u_0 = u.copy()

    return u.reshape(shape), tau
