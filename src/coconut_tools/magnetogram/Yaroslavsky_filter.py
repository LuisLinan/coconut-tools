"""
Preprocess magnetograms with a local weighted Yaroslavsky-style filter.

This module uses the shared acquisition, reading, coordinate, rotation, flux,
plotting, and boundary-writing subpackages. Its specific processing step
applies optional Gaussian smoothing followed by the local weighted kernel
implemented in ``filters.yaroslavsky.filter3``.

Author: Jose Murteira
Cleaned and modularized by: Luis
"""

from typing import Any
import os

import numpy as np
import scipy.ndimage

from coconut_tools.magnetogram.core.config import _as_bool
from coconut_tools.magnetogram.filters.gaussian_filter import geodesic_gaussian_filtering
from coconut_tools.magnetogram.filters.yaroslavsky import filter3, spherical_yaroslavsky
from coconut_tools.magnetogram.processing.pipeline import (
    FilterStep,
    run_config,
    run_magnetogram_pipeline,
)
from coconut_tools.tools.logger_config import setup_logger

logger = setup_logger(__name__)

def filter_radial_field_weighted(
    Br: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    alpha_factor: float,
    Rn: float,
    sig: float = 0.0,
    write_gaussian_prepass: bool = False
):
    """Apply optional Gaussian smoothing and local weighted filtering to Br.

    Grid spacing is estimated from the supplied longitude and colatitude vectors
    in radians. Following the article implementation, a single isotropic
    physical spacing ``R_sun * max(delta_theta, delta_phi)`` is passed to the
    local weighted filter.

    Args:
        Br (np.ndarray): Input magnetic field map.
        phi (np.ndarray): 1D array of longitudes in radians.
        theta (np.ndarray): 1D array of colatitudes in radians.
        alpha_factor (float): Alpha controlling kernel sharpness.
        Rn (float): Neighborhood radius in grid-spacing units.
        sig (float): Sigma of optional Gaussian smoothing.
        write_gaussian_prepass (bool): Whether to export Gaussian-smoothed version.

    Returns:
        np.ndarray: Filtered Br field.
    """
    Br = np.asarray(Br)
    phi = np.asarray(phi, dtype=float)
    theta = np.asarray(theta, dtype=float)

    if Br.ndim != 2:
        raise ValueError("Br must be a 2D array.")
    if phi.ndim != 1 or theta.ndim != 1:
        raise ValueError("phi and theta must be 1D arrays.")
    if Br.shape != (theta.size, phi.size):
        raise ValueError(
            f"Br shape {Br.shape} does not match theta/phi sizes "
            f"({theta.size}, {phi.size})."
        )
    if theta.size < 2 or phi.size < 2:
        raise ValueError("theta and phi must contain at least two points.")
    if not (
        np.all(np.isfinite(Br))
        and np.all(np.isfinite(theta))
        and np.all(np.isfinite(phi))
    ):
        raise ValueError("Br, theta, and phi must contain only finite values.")
    if Rn <= 0:
        raise ValueError("Rn must be positive.")
    if alpha_factor < 0:
        raise ValueError("alpha_factor must be non-negative.")
    if sig < 0:
        raise ValueError("sig must be non-negative.")

    theta_steps = np.abs(np.diff(theta))
    phi_steps = np.abs(np.diff(np.unwrap(phi)))
    theta_steps = theta_steps[theta_steps > 0]
    phi_steps = phi_steps[phi_steps > 0]
    if theta_steps.size == 0 or phi_steps.size == 0:
        raise ValueError("theta and phi coordinates must contain at least two distinct values.")

    dtheta = float(np.median(theta_steps))
    dphi = float(np.median(phi_steps))

    R_sun = 696.34e6
    delta_var = R_sun * max(dtheta, dphi)

    Br_smoothed = scipy.ndimage.gaussian_filter(Br, sig) if sig > 0 else Br.copy()

    if write_gaussian_prepass:
        np.save("Br_gaussian_prepass.npy", Br_smoothed)

    logger.info(
        "Running local filter with alpha=%.2f, Rn=%.2f, dtheta=%.6g rad, dphi=%.6g rad, delta=%.2f m",
        alpha_factor,
        Rn,
        dtheta,
        dphi,
        delta_var,
    )
    Br_filtered = filter3(Br_smoothed, delta_var, delta_var, alpha_factor, Rn)

    return Br_filtered


_LEGACY_YAROSLAVSKY_KEYS = ("Rn", "alpha", "sig")
_SPHERICAL_YAROSLAVSKY_KEYS = ("h_gauss", "sig_deg")


def resolve_yaroslavsky_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Select the Yaroslavsky mode and its parameters from a configuration.

    ``radius_deg`` selects the spherical (physical) filter; the legacy pixel
    keys ``Rn``, ``alpha`` and ``sig`` are then rejected. Without it, the
    legacy filter is used exactly as before and the spherical keys are
    rejected.

    Returns:
        Dict with ``mode`` (``"spherical"`` or ``"legacy"``) and its parameters.

    Raises:
        ValueError: On mixed legacy/spherical keys.
    """
    radius_deg = config.get("radius_deg")
    if radius_deg is not None:
        mixed = [key for key in _LEGACY_YAROSLAVSKY_KEYS if config.get(key) is not None]
        if mixed:
            raise ValueError(
                "radius_deg selects the spherical Yaroslavsky filter; remove the "
                f"legacy keys {mixed} (use h_gauss and sig_deg instead)."
            )
        return {
            "mode": "spherical",
            "radius_deg": float(radius_deg),
            "h_gauss": config.get("h_gauss"),
            "sig_deg": float(config.get("sig_deg", 0.0)),
        }
    mixed = [key for key in _SPHERICAL_YAROSLAVSKY_KEYS if config.get(key) is not None]
    if mixed:
        raise ValueError(
            f"{mixed} only apply to the spherical Yaroslavsky filter; set "
            "radius_deg to use it."
        )
    return {
        "mode": "legacy",
        "alpha": config.get("alpha", 1.0),
        "Rn": config.get("Rn", 5.0),
        "sig": config.get("sig", 0.0),
    }


def filter_radial_field_spherical(
    Br: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    radius_deg: float,
    h_gauss: float | None = None,
    sig_deg: float = 0.0,
    write_gaussian_prepass: bool = False,
) -> tuple[np.ndarray, dict]:
    """Physical Yaroslavsky: optional great-circle Gaussian + spherical disk filter.

    Args:
        Br (np.ndarray): 2D radial field.
        phi (np.ndarray): 1D uniform longitude centres (radians).
        theta (np.ndarray): 1D colatitude centres (radians), north to south.
        radius_deg (float): Disk radius in heliographic degrees.
        h_gauss (float | None): Radiometric scale in Gauss, or None for the
            automatic area-weighted noise estimate ``1.4826 * MAD(B)``.
        sig_deg (float): Pre-smoothing width in degrees (0 disables it).
        write_gaussian_prepass (bool): Save the pre-smoothed map to
            ``Br_gaussian_prepass.npy`` (debug, as in the legacy path).

    Returns:
        tuple[np.ndarray, dict]: Filtered field and ``h_gauss``, ``h_source``,
        ``radius_deg``.
    """
    if sig_deg < 0.0:
        raise ValueError("sig_deg must be non-negative.")
    Br_smoothed = (
        geodesic_gaussian_filtering(Br, phi, theta, sig_deg)
        if sig_deg > 0.0
        else np.asarray(Br, dtype=float)
    )
    if write_gaussian_prepass:
        np.save("Br_gaussian_prepass.npy", Br_smoothed)
    return spherical_yaroslavsky(Br_smoothed, phi, theta, radius_deg, h_gauss)


def _resolve_yaroslavsky_step(config: dict[str, Any]) -> dict[str, Any]:
    """Yaroslavsky settings plus the ``write_gaussian_prepass`` debug flag."""
    settings = resolve_yaroslavsky_settings(config)
    settings["write_gaussian_prepass"] = _as_bool(config.get("write_gaussian_prepass", False))
    return settings


def _apply_yaroslavsky(
    Br: np.ndarray,
    Theta: np.ndarray,
    Phi: np.ndarray,
    settings: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Yaroslavsky step of the shared pipeline (legacy or spherical)."""
    if settings["mode"] == "spherical":
        filtered, yaroslavsky_info = filter_radial_field_spherical(
            Br,
            Phi[0, :],
            Theta[:, 0],
            settings["radius_deg"],
            h_gauss=settings["h_gauss"],
            sig_deg=settings["sig_deg"],
            write_gaussian_prepass=settings["write_gaussian_prepass"],
        )
    else:
        filtered = filter_radial_field_weighted(
            Br,
            Phi[0, :],
            Theta[:, 0],
            settings["alpha"],
            settings["Rn"],
            settings["sig"],
            write_gaussian_prepass=settings["write_gaussian_prepass"],
        )
        yaroslavsky_info = None
    return filtered, {
        "yaroslavsky_mode": settings["mode"],
        "yaroslavsky_info": yaroslavsky_info,
    }


YAROSLAVSKY_STEP = FilterStep(
    resolve_settings=_resolve_yaroslavsky_step,
    apply=_apply_yaroslavsky,
)
"""Yaroslavsky step for ``processing.pipeline.run_magnetogram_pipeline``."""


def process_magnetogram_date(
    config: dict[str, Any],
    target_date,
    method_used: str = "Yaroslavsky",
    output_path_fig: str | None = None,
    unique_names: bool = False,
) -> dict[str, Any]:
    """Process one target time with the local weighted filter pipeline.

    Runs the shared pipeline (``processing.pipeline.run_magnetogram_pipeline``):
    read or interpolate the magnetogram, effective time, optional Stonyhurst
    rotation, optional net-flux correction (``flux_correction_stage``),
    Yaroslavsky filter, ``/2.2 * amp``, boundary file and optional figure.

    Args:
        config (dict[str, Any]): Processing configuration: the common keys of
            ``run_magnetogram_pipeline`` plus legacy ``alpha``, ``Rn``, ``sig``
            or spherical ``radius_deg``, ``h_gauss``, ``sig_deg``; and
            ``write_gaussian_prepass``.
        target_date: Requested processing time.
        method_used (str): Method label used in output filenames.
        output_path_fig (str | None): Figure file or directory, resolved with
            ``resolve_figure_path`` at the effective time. If omitted, the
            figure name is built from the effective time.
        unique_names (bool): Append the effective time to the ``.dat`` and
            figure names; set by ``process_config`` when several dates are
            processed so a time series does not overwrite its outputs.

    Returns:
        dict[str, Any]: Processing metadata, including target ``date``,
        ``effective_date``, output paths, selected local file or interpolation
        stencil, optional ``Br_linear``, rotation angle, ``yaroslavsky_mode``,
        ``yaroslavsky_info`` (spherical mode: ``h_gauss``, ``h_source``,
        ``radius_deg``) and ``flux_correction_stage``.
    """
    return run_magnetogram_pipeline(
        config,
        target_date,
        YAROSLAVSKY_STEP,
        method_used=method_used,
        output_path_fig=output_path_fig,
        unique_names=unique_names,
    )


def process_config(config: dict[str, Any], method_used: str = "Yaroslavsky") -> list[dict[str, Any]]:
    """Process all target times described by one Yaroslavsky configuration.

    See ``processing.pipeline.run_config``: one target, or a time series with
    ``cadence_hours`` and ``total_hours`` whose outputs receive the effective
    magnetogram time.

    Args:
        config (dict[str, Any]): Processing configuration.
        method_used (str): Method label used in output filenames.

    Returns:
        list[dict[str, Any]]: Per-date processing results.
    """
    return run_config(config, process_magnetogram_date, method_used)


if __name__ == "__main__":

    base_output_dir = r"C:\Users\luisl\Desktop\testmagnetogram\test"
    label = "Yaroslavsky_1.4_100_1.5"
    output_dir = os.path.join(base_output_dir, label)
    figure_output_dir = os.path.join(base_output_dir, "images")

    configs = [{
            "date": "2011-09-09T01:47:05",
            "custom_magnetogram": r"C:\Users\luisl\Desktop\testmagnetogram\ai_magnetogram\AI_synopt_20260801_162400_TAI.fits",
            "amp": 1,
            "write_map": False,
            "show_map": True,
            "visu_type": "sinlat",
            "carrington": False,
            "rotate_to_stonyhurst": False,
            "interpolation": False,
            "interpolation_order": 2,
            "resize": False,
            "flux_correct": False,
            "flux_correction_method": "surface_mean", #surface_mean' or 'polarity_scaling'
            "flux_correction_stage": "after",  # "after" (default) or "before"
            "map_type": "custom",
            "adapt_map": 6,
            "output_dir": output_dir,
            "download_dir": output_dir,
            "output_path_fig": os.path.join(figure_output_dir, f"{label}.png"),
            "drms_email": "luis.linan@kuleuven.be",
            "alpha": 1.4,
            "Rn" : 100,
            "sig": 1.5,
            # Spherical (physical) Yaroslavsky: set radius_deg and remove "alpha",
            # "Rn" and "sig" (pixel keys).
            # "radius_deg": 2.0,   # great-circle disk radius in degrees (1 deg ~ 12.15 Mm)
            # "h_gauss": None,     # radiometric scale in Gauss; None = automatic area-weighted noise (1.4826 * MAD(B))
            # "sig_deg": 0.5,      # geodesic Gaussian pre-smoothing width in degrees; 0 disables it
        }]

    # for time evolving add : cadence_hours and total_hours to the config dictionary, e.g.:
    # "cadence_hours": 3,
    # "total_hours": 72,

    for config in configs:
        try:
            process_config(config, method_used="Yaroslavsky")
        except Exception as exc:
            logger.warning(
                f'Failed to process {config["date"]} and {config["map_type"]}: {exc}'
            )

