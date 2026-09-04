"""Spherical-harmonic projection and reconstruction of radial-field maps."""

import numpy as np
from scipy import special as scisp

from coconut_tools.magnetogram.core.coordinates import spherical_pixel_areas
from coconut_tools.tools.logger_config import setup_logger

logger = setup_logger(__name__)


def spherical_harmonic(
    m: int,
    l: int,
    Phi: np.ndarray,
    Theta: np.ndarray,
) -> np.ndarray:
    """Evaluate spherical harmonics across supported SciPy versions."""
    if hasattr(scisp, "sph_harm"):
        return scisp.sph_harm(m, l, Phi, Theta)
    return scisp.sph_harm_y(l, m, Theta, Phi)


def project_and_reconstruct(Br, Theta, Phi, lmax, amp=1, alpha=0):
    """Project ``Br`` onto spherical harmonics and reconstruct the filtered map."""
    logger.info("Beginning of projection")

    Br = np.asarray(Br)
    Theta = np.asarray(Theta)
    Phi = np.asarray(Phi)

    if Br.shape != Theta.shape or Br.shape != Phi.shape:
        raise ValueError("Br, Theta, and Phi must have the same 2D shape.")
    if Br.ndim != 2:
        raise ValueError("Br, Theta, and Phi must be 2D arrays.")
    if lmax < 1:
        raise ValueError("lmax must be >= 1.")
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")

    nb_modes_tot = int((lmax + 1) * (lmax + 2) / 2 - 1)
    surface_weight = spherical_pixel_areas(
        Theta[:, 0],
        Phi[0, :],
        Br.shape,
    )
    coefbr = np.zeros(nb_modes_tot, dtype=complex)

    mode = 0
    for degree in range(1, lmax + 1):
        logger.info("l = %d", degree)
        damping = 1.0 / (1.0 + alpha * degree**2 * (degree + 1) ** 2)
        for order in range(degree + 1):
            ylm = spherical_harmonic(order, degree, Phi, Theta)
            coefficient = np.sum(Br * np.conj(ylm) * surface_weight)
            coefbr[mode] = damping * coefficient
            mode += 1

    logger.info("End of projection")
    logger.info("Reconstructing Br")
    Br_mode = np.zeros_like(Br, dtype=float)

    mode = 0
    for degree in range(1, lmax + 1):
        logger.info("l = %d", degree)
        for order in range(degree + 1):
            ylm = spherical_harmonic(order, degree, Phi, Theta)
            contribution = np.real(coefbr[mode] * ylm)
            if order > 0:
                contribution *= 2.0
            Br_mode += contribution
            mode += 1

    Br_mode /= 2.2
    Br_mode *= amp
    logger.info("End of reconstructing Br")
    return Br_mode, coefbr
