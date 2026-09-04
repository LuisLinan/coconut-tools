"""Surface-area-aware magnetic-flux diagnostics and corrections."""

import numpy as np

from coconut_tools.magnetogram.core.coordinates import spherical_pixel_areas
from coconut_tools.tools.logger_config import setup_logger

logger = setup_logger(__name__)


def _regular_phi_from_br(Br: np.ndarray) -> np.ndarray:
    """Return an endpoint-free regular longitude grid matching ``Br`` columns."""
    return np.linspace(0.0, 2.0 * np.pi, Br.shape[1], endpoint=False)


def _pixel_area(
    theta: np.ndarray,
    phi: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Compute spherical pixel areas from colatitude and longitude centers."""
    return spherical_pixel_areas(theta, phi, shape)


def _surface_mean_area(
    theta: np.ndarray,
    phi: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Compute spherical areas used by the surface-mean flux correction."""
    return spherical_pixel_areas(theta, phi, shape)


def _flux_summary(
    Br: np.ndarray,
    pixel_area: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return positive flux, negative flux, net flux, and imbalance percentage."""
    positive_mask = Br > 0
    negative_mask = Br < 0
    flux_positive = float(np.sum(Br[positive_mask] * pixel_area[positive_mask]))
    flux_negative = float(np.sum(Br[negative_mask] * pixel_area[negative_mask]))
    net_flux = flux_positive + flux_negative
    denominator = flux_positive - flux_negative
    imbalance_percent = net_flux / denominator * 100 if denominator != 0 else np.nan
    return flux_positive, flux_negative, net_flux, imbalance_percent


def _log_flux_summary(
    label: str,
    Br: np.ndarray,
    pixel_area: np.ndarray,
) -> None:
    """Log flux-balance diagnostics for a Br map and pixel-area grid."""
    flux_positive, flux_negative, net_flux, imbalance_percent = _flux_summary(
        Br,
        pixel_area,
    )
    logger.info("%s flux positive: %.6e", label, flux_positive)
    logger.info("%s flux negative: %.6e", label, flux_negative)
    logger.info("%s net flux: %.6e", label, net_flux)
    logger.info("%s flux imbalance: %.6e %%", label, imbalance_percent)


def _correct_net_flux_surface_mean(
    Br: np.ndarray,
    pixel_area: np.ndarray,
) -> np.ndarray:
    """Subtract the surface-weighted mean Br from every pixel."""
    mean_br = np.sum(Br * pixel_area) / np.sum(pixel_area)
    logger.info("Net flux correction: subtracting surface mean Br=%.6e", mean_br)
    return Br - mean_br


def _correct_net_flux_polarity_scaling(
    Br: np.ndarray,
    pixel_area: np.ndarray,
) -> np.ndarray:
    """Balance integrated flux by rescaling positive and negative polarities."""
    flux_positive, flux_negative, _, _ = _flux_summary(Br, pixel_area)
    if flux_negative == 0 or flux_positive == 0:
        logger.warning("Flux balancing skipped because one polarity is missing.")
        return Br

    balancing_factor = np.sqrt(flux_positive / abs(flux_negative))
    logger.info("Flux balancing factor: %.6e", balancing_factor)
    return np.where(Br > 0, Br / balancing_factor, Br * balancing_factor)


def correct_net_flux(
    Br: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray | None = None,
    method: str = "surface_mean",
) -> np.ndarray:
    """Reduce or remove net magnetic flux using exact spherical cell areas."""
    method = method.lower()
    phi = _regular_phi_from_br(Br) if phi is None else phi
    if method in {"surface_mean", "mean", "subtract_mean"}:
        pixel_area = _surface_mean_area(theta, phi, Br.shape)
        _log_flux_summary("Before surface_mean correction", Br, pixel_area)
        corrected = _correct_net_flux_surface_mean(Br, pixel_area)
        _log_flux_summary("After surface_mean correction", corrected, pixel_area)
        return corrected
    if method in {"polarity_scaling", "polarity", "multiplicative"}:
        pixel_area = _pixel_area(theta, phi, Br.shape)
        _log_flux_summary("Before polarity_scaling correction", Br, pixel_area)
        corrected = _correct_net_flux_polarity_scaling(Br, pixel_area)
        _log_flux_summary("After polarity_scaling correction", corrected, pixel_area)
        return corrected
    raise ValueError(
        "flux correction method must be 'surface_mean' or 'polarity_scaling'."
    )
