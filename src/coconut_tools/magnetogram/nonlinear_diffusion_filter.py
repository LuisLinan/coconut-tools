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
from coconut_tools.tools.logger_config import setup_logger

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
        gradx = scipy.signal.convolve2d(u, gradkernelx, mode="same", boundary="symm")
        grady = scipy.signal.convolve2d(u, gradkernely, mode="same", boundary="symm")
        return gradx, grady

    def computeDiffusivity(u, lamb):
        gradx, grady = gradU(u)
        gradm2 = gradx ** 2 + grady ** 2
        return 1.0 / np.sqrt(1.0 + gradm2 / (lamb * lamb))

    def computef_matrix(u, g, shape):
        f_matrix = scipy.sparse.lil_matrix((u.shape[0], u.shape[0]))

        rows, cols = shape
        inv_dx2 = 1.0 / (dx ** 2)
        inv_dy2 = 1.0 / (dy ** 2)

        for j in range(rows):
            for i in range(cols):
                k = j * cols + i

                i_plus = (i + 1) % cols
                i_minus = (i - 1) % cols

                g_right = math.sqrt(g[j, i] * g[j, i_plus])
                g_left = math.sqrt(g[j, i] * g[j, i_minus])

                k_right = j * cols + i_plus
                k_left = j * cols + i_minus

                f_matrix[k, k_right] = g_right * inv_dx2
                f_matrix[k, k_left] = g_left * inv_dx2

                diagonal = -(g_right + g_left) * inv_dx2

                if j > 0:
                    g_up = math.sqrt(g[j, i] * g[j - 1, i])
                    f_matrix[k, k - cols] = g_up * inv_dy2
                    diagonal -= g_up * inv_dy2

                if j < rows - 1:
                    g_down = math.sqrt(g[j, i] * g[j + 1, i])
                    f_matrix[k, k + cols] = g_down * inv_dy2
                    diagonal -= g_down * inv_dy2

                f_matrix[k, k] = diagonal

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
    gradm = np.sqrt(gradx ** 2 + grady ** 2)
    lamb = 1.4826 * np.median(np.abs(gradm - np.median(gradm)))
    if lamb == 0.0:
        lamb = np.finfo(float).eps    
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
