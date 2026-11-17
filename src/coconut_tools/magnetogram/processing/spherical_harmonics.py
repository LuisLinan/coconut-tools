"""Spherical-harmonic projection and reconstruction of radial-field maps.

The maps handled here live on tensor-product grids: ``Theta`` varies only
along rows and ``Phi`` only along columns (see ``core.coordinates.build_theta_phi``).
The colatitude axis is not necessarily regular (HMI synoptic maps are sampled
uniformly in sine-latitude), so a fast transform library with its own
quadrature grid (e.g. SHTns) would require interpolation and change results.

Instead the transform exploits separability exactly:

``Y_lm(theta, phi) = Y_lm(theta, 0) * exp(i m phi)``

so the azimuthal sums are done once for every order ``m`` with two real
matrix products, and ``Y_lm`` only needs to be evaluated on the 1D colatitude
axis. This is mathematically identical to the former direct 2D summation (same
pixel solid angles, same modes, same damping, same final scaling); only the
floating-point summation order differs. The cost drops from
``O(lmax^2 * Ntheta * Nphi)`` special-function evaluations to
``O(lmax^2 * Ntheta)`` evaluations plus ``O(lmax * Ntheta * Nphi)`` multiply-adds.
"""

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


def _separable_axes(Theta: np.ndarray, Phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the 1D colatitude and longitude axes of a tensor-product grid.

    Raises:
        ValueError: If ``Theta`` varies along columns or ``Phi`` along rows.
            The pixel solid angles are built from ``Theta[:, 0]`` and
            ``Phi[0, :]``, so such grids were never consistently supported.
    """
    theta = Theta[:, 0]
    phi = Phi[0, :]
    if not (np.array_equal(Theta, np.broadcast_to(theta[:, None], Theta.shape))
            and np.array_equal(Phi, np.broadcast_to(phi[None, :], Phi.shape))):
        raise ValueError(
            "Theta must vary only along rows and Phi only along columns "
            "(tensor-product grid as built by build_theta_phi)."
        )
    return theta, phi


def project_and_reconstruct(Br, Theta, Phi, lmax, amp=1, alpha=0):
    """Project ``Br`` onto spherical harmonics and reconstruct the filtered map.

    Args:
        Br: 2D radial field, shape ``(Ntheta, Nphi)``.
        Theta: 2D colatitude grid in radians, same shape as ``Br``.
        Phi: 2D longitude grid in radians, same shape as ``Br``.
        lmax: Highest degree kept (modes ``l = 1 ... lmax``, ``m = 0 ... l``;
            the monopole is excluded).
        amp: Multiplicative factor applied to the reconstructed map.
        alpha: Non-negative damping strength, ``1 / (1 + alpha l^2 (l+1)^2)``.

    Returns:
        ``(Br_mode, coefbr)``: the reconstructed map divided by 2.2 and
        multiplied by ``amp``, and the damped complex coefficients ordered by
        ``l`` then ``m``.
    """
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

    theta, phi = _separable_axes(Theta, Phi)

    nb_modes_tot = int((lmax + 1) * (lmax + 2) / 2 - 1)
    surface_weight = spherical_pixel_areas(
        Theta[:, 0],
        Phi[0, :],
        Br.shape,
    )
    coefbr = np.zeros(nb_modes_tot, dtype=complex)

    # Azimuthal Fourier sums for every order m (real matrix products):
    # G[i, m] = sum_j Br_ij dOmega_ij exp(-i m phi_j)
    orders = np.arange(lmax + 1)
    m_phi = np.outer(phi, orders)
    cos_m_phi = np.cos(m_phi)
    sin_m_phi = np.sin(m_phi)
    weighted_br = np.asarray(Br * surface_weight, dtype=float)
    G = (weighted_br @ cos_m_phi) - 1j * (weighted_br @ sin_m_phi)
    del weighted_br

    # Y_lm(theta, phi=0) on the 1D colatitude axis, reused for reconstruction.
    phi_zero = np.zeros_like(theta)
    ylm_theta = np.zeros((nb_modes_tot, theta.size), dtype=complex)

    mode = 0
    for degree in range(1, lmax + 1):
        logger.info("l = %d", degree)
        damping = 1.0 / (1.0 + alpha * degree**2 * (degree + 1) ** 2)
        for order in range(degree + 1):
            ylm_theta[mode] = spherical_harmonic(order, degree, phi_zero, theta)
            coefficient = np.sum(np.conj(ylm_theta[mode]) * G[:, order])
            coefbr[mode] = damping * coefficient
            mode += 1

    logger.info("End of projection")
    logger.info("Reconstructing Br")

    # F[i, m] = sum_l c_lm Y_lm(theta_i, 0), with m > 0 doubled to account
    # for the m < 0 modes of a real field (c_l,-m Y_l,-m = conj(c_lm Y_lm)).
    F = np.zeros((theta.size, lmax + 1), dtype=complex)
    mode = 0
    for degree in range(1, lmax + 1):
        logger.info("l = %d", degree)
        for order in range(degree + 1):
            contribution = coefbr[mode] * ylm_theta[mode]
            if order > 0:
                contribution *= 2.0
            F[:, order] += contribution
            mode += 1

    # Br_mode = Re(sum_m F[:, m] exp(i m phi))
    Br_mode = (F.real @ cos_m_phi.T) - (F.imag @ sin_m_phi.T)

    Br_mode /= 2.2
    Br_mode *= amp
    logger.info("End of reconstructing Br")
    return Br_mode, coefbr
