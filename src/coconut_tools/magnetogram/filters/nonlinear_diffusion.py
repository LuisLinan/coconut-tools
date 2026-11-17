"""
Low-level nonlinear diffusion filter based on Perona-Malik smoothing.

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


# ---------------------------------------------------------------------------
# Spherical (physical) nonlinear diffusion
# ---------------------------------------------------------------------------

MAD_TO_SIGMA = 1.4826
"""Gaussian-consistency factor of the median absolute deviation."""

SPHERICAL_DEFAULT_STEPS = 28
"""Default number of AOS time steps of the spherical NLD (~1 % from the
converged solution on a 360x720 HMI map for a 2 deg diffusion length)."""


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the weighted median of ``values``."""
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def _spherical_gradient_magnitude(
    u: np.ndarray,
    theta: np.ndarray,
    sin_theta: np.ndarray,
    dphi: float,
) -> np.ndarray:
    """Return |grad u| on the unit sphere (field units per radian).

    The colatitude derivative uses second-order differences on the possibly
    non-uniform axis (one-sided at the polar rows); the longitude derivative
    is periodic: ``(u[j+1] - u[j-1]) / (2 dphi sin(theta))``.
    """
    d_theta = np.gradient(u, theta, axis=0)
    d_phi = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dphi)
    return np.sqrt(d_theta**2 + (d_phi / sin_theta[:, None]) ** 2)


def _solve_tridiagonal(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve tridiagonal systems along axis 0, batched over axis 1 (Thomas).

    ``lower[0]`` and ``upper[-1]`` are ignored. The systems handled here are
    strictly diagonally dominant, so no pivoting is needed.
    """
    n = diag.shape[0]
    c_prime = np.empty_like(diag)
    d_prime = np.empty_like(rhs)
    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denominator = diag[i] - lower[i] * c_prime[i - 1]
        c_prime[i] = upper[i] / denominator if i < n - 1 else 0.0
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / denominator
    solution = np.empty_like(rhs)
    solution[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1]
    return solution


def _solve_cyclic_tridiagonal(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve periodic tridiagonal systems along axis 0 (Sherman-Morrison).

    The corner couplings are ``A[0, n-1] = lower[0]`` and
    ``A[n-1, 0] = upper[n-1]``.
    """
    beta, alpha = lower[0], upper[-1]
    gamma = -diag[0]
    modified = diag.copy()
    modified[0] = diag[0] - gamma
    modified[-1] = diag[-1] - alpha * beta / gamma
    x = _solve_tridiagonal(lower, modified, upper, rhs)
    correction_rhs = np.zeros_like(rhs)
    correction_rhs[0] = gamma
    correction_rhs[-1] = alpha
    z = _solve_tridiagonal(lower, modified, upper, correction_rhs)
    factor = (x[0] + beta * x[-1] / gamma) / (1.0 + z[0] + beta * z[-1] / gamma)
    return x - factor * z


def spherical_nonlinear_diffusion(
    Br: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    diffusion_length_deg: float,
    iterations: int = SPHERICAL_DEFAULT_STEPS,
    lambda_g_per_deg: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Nonlinear diffusion of a global radial field on the sphere.

    Solves ``dB/dt = div_s(g(|grad_s B|) grad_s B)`` on the unit sphere with
    the Charbonnier diffusivity of the legacy filter,
    ``g = 1 / sqrt(1 + |grad_s B|^2 / lambda^2)``.

    Discretisation: finite volumes on the (theta, phi) cells with their exact
    areas ``M``, face lengths (``delta_theta`` for east-west faces,
    ``sin(theta_face) * delta_phi`` for north-south faces) and centre
    distances (``sin(theta) * delta_phi`` and ``theta_{i+1} - theta_i``).
    Face diffusivity is the geometric mean of the two cells (as in the legacy
    filter). Longitude is periodic; polar faces have zero length, the polar
    rows are coupled in longitude.

    Time stepping: additive operator splitting (AOS, Weickert et al. 1998),
    semi-implicit with the diffusivity frozen during each step,
    ``B^{n+1} = 1/2 [(M - 2 tau K_theta)^-1 + (M - 2 tau K_phi)^-1] M B^n``.
    Each direction is a batch of (periodic) tridiagonal systems solved
    exactly, so a step costs O(cells). Unconditionally stable, monotone,
    strongly damping (no sign-flipping fine-scale noise as with
    Crank-Nicolson), and the surface integral of ``Br`` is conserved to
    round-off on any grid. The splitting error decreases with the number of
    steps (measured on a 360x720 HMI map, L = 2 deg: 4.5 % RMS from the
    converged solution with 7 steps, 1.0 % with 28).

    Physical scale: in regions where ``g ~ 1`` the result is a Gaussian
    smoothing of standard deviation ``diffusion_length_deg``: the total time is
    ``T = L^2 / 2`` (``L`` in radians) and ``tau = T / iterations``.

    ``lambda`` (field units per heliographic degree) is either given, or
    estimated once from the input as ``1.4826 * MAD(|grad_s B|)`` with an
    area-weighted median, excluding cells where the field is exactly zero
    (e.g. NaN polar caps set to 0).

    Args:
        Br: Radial field ``(theta.size, phi.size)``.
        phi: Uniform increasing longitude centres (radians), endpoint-free.
        theta: Increasing colatitude centres (radians), any spacing.
        diffusion_length_deg: Diffusion length in heliographic degrees (> 0).
        iterations: Number of AOS time steps (>= 1); numerical accuracy only.
        lambda_g_per_deg: Contrast threshold in field units per degree, or
            None for the automatic estimate.

    Returns:
        ``(filtered, info)`` with ``info`` keys ``lambda_g_per_deg``,
        ``lambda_source`` (``"auto"`` or ``"config"``), ``tau_rad2``,
        ``total_time_rad2``, ``steps`` and ``scheme`` (``"AOS"``).

    Raises:
        ValueError: Invalid inputs or non-uniform longitude axis.
    """
    from coconut_tools.magnetogram.core.coordinates import theta_cell_edges

    u = np.asarray(Br, dtype=float)
    phi = np.asarray(phi, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if u.ndim != 2 or u.shape != (theta.size, phi.size):
        raise ValueError("Br must be 2D with shape (theta.size, phi.size).")
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(phi)) and np.all(np.isfinite(theta))):
        raise ValueError("Br, theta, and phi must contain only finite values.")
    if not np.isfinite(diffusion_length_deg) or diffusion_length_deg <= 0.0:
        raise ValueError("diffusion_length_deg must be a finite positive number.")
    if int(iterations) != iterations or iterations < 1:
        raise ValueError("iterations must be a positive integer.")
    if lambda_g_per_deg is not None and (
        not np.isfinite(lambda_g_per_deg) or lambda_g_per_deg <= 0.0
    ):
        raise ValueError("lambda_g_per_deg must be a finite positive number.")

    n_theta, n_phi = u.shape
    if n_theta < 2 or n_phi < 3:
        raise ValueError("Spherical NLD needs at least 2 rows and 3 columns.")
    dphi = 2.0 * np.pi / n_phi
    if not np.allclose(np.diff(np.unwrap(phi)), dphi, atol=1.0e-9, rtol=0.0):
        raise ValueError("Spherical NLD requires a uniform, complete longitude axis.")

    edges = theta_cell_edges(theta)
    sin_theta = np.sin(theta)
    row_area = (np.cos(edges[:-1]) - np.cos(edges[1:])) * dphi
    area = np.broadcast_to(row_area[:, None], u.shape)

    # Geometric factors of the conductances (diffusivity excluded).
    ew_factor = ((edges[1:] - edges[:-1]) / (sin_theta * dphi))[:, None]   # (n_theta, 1)
    ns_factor = (np.sin(edges[1:-1]) * dphi / np.diff(theta))[:, None]     # (n_theta - 1, 1)

    rad_per_deg = np.pi / 180.0
    if lambda_g_per_deg is None:
        gradient = _spherical_gradient_magnitude(u, theta, sin_theta, dphi)
        valid = u != 0.0
        if not np.any(valid):
            raise ValueError("Cannot estimate lambda: the field is identically zero.")
        values = gradient[valid]
        weights = area[valid]
        centre = _weighted_median(values, weights)
        lam = MAD_TO_SIGMA * _weighted_median(np.abs(values - centre), weights)
        if lam <= 0.0:
            raise ValueError("Automatic lambda is zero; give lambda_g_per_deg explicitly.")
        lambda_source = "auto"
    else:
        lam = float(lambda_g_per_deg) / rad_per_deg   # G/deg -> G/rad
        lambda_source = "config"
    logger.info(
        "Spherical NLD (AOS): lambda = %.6g G/deg (%s), length = %.4g deg, %d steps",
        lam * rad_per_deg, lambda_source, diffusion_length_deg, iterations,
    )

    total_time = 0.5 * (np.radians(diffusion_length_deg)) ** 2
    tau = total_time / iterations
    two_tau = 2.0 * tau

    for _ in range(int(iterations)):
        grad = _spherical_gradient_magnitude(u, theta, sin_theta, dphi)
        g = 1.0 / np.sqrt(1.0 + (grad / lam) ** 2)
        # Conductance between (i, j) and (i, j+1), and between (i, j) and (i+1, j).
        c_ew = np.sqrt(g * np.roll(g, -1, axis=1)) * ew_factor
        c_ns = np.sqrt(g[:-1] * g[1:]) * ns_factor
        mass_u = area * u

        # Longitude direction: periodic systems along phi (transposed to axis 0).
        c_west = np.roll(c_ew, 1, axis=1)
        u_phi = _solve_cyclic_tridiagonal(
            (-two_tau * c_west).T.copy(),
            (area + two_tau * (c_ew + c_west)).T.copy(),
            (-two_tau * c_ew).T.copy(),
            mass_u.T.copy(),
        ).T

        # Colatitude direction: systems along theta, no flux through the poles.
        lower = np.zeros_like(u)
        upper = np.zeros_like(u)
        lower[1:] = -two_tau * c_ns
        upper[:-1] = -two_tau * c_ns
        diagonal = np.array(area, copy=True)
        diagonal[:-1] += two_tau * c_ns
        diagonal[1:] += two_tau * c_ns
        u_theta = _solve_tridiagonal(lower, diagonal, upper, mass_u)

        u = 0.5 * (u_phi + u_theta)

    info_out = {
        "lambda_g_per_deg": lam * rad_per_deg,
        "lambda_source": lambda_source,
        "tau_rad2": tau,
        "total_time_rad2": total_time,
        "steps": int(iterations),
        "scheme": "AOS",
    }
    return u, info_out
