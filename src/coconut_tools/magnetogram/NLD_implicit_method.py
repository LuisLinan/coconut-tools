"""
Preprocess magnetograms with nonlinear diffusion filtering.

This module uses the shared acquisition, reading, coordinate, rotation, flux,
plotting, and boundary-writing subpackages. Its specific processing step
applies optional Gaussian smoothing followed by the Perona-Malik nonlinear
diffusion kernel implemented in ``filters.nonlinear_diffusion``.

Author: Jose Murteira
Cleaned and modularized by: Luis
"""
from typing import Any
import os

import numpy as np
import scipy.ndimage

from coconut_tools.magnetogram.core.config import _as_bool
from coconut_tools.magnetogram.filters.gaussian_filter import geodesic_gaussian_filtering
from coconut_tools.magnetogram.filters.nonlinear_diffusion import (
    SPHERICAL_DEFAULT_STEPS,
    nonlinearDiffusionFilter,
    spherical_nonlinear_diffusion,
)
from coconut_tools.magnetogram.processing.pipeline import (
    FilterStep,
    run_config,
    run_magnetogram_pipeline,
)
from coconut_tools.tools.logger_config import setup_logger

logger = setup_logger(__name__)

def filter_radial_field(
    Br: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    iterations: int = 3,
    apply_gaussian: bool = True,
    gaussian_sigma: float = 1.0,
    dx_override: float = 1.0,
    dy_override: float = 1.0,
    tau: float = 1.0
):
    """Apply optional Gaussian smoothing and nonlinear diffusion to Br.

    ``phi`` and ``theta`` are kept in the signature for consistency with the
    other filters; the numerical spacing used by the nonlinear diffusion solver
    comes from ``dx_override`` and ``dy_override``.

    Args:
        Br (np.ndarray): 2D array of the radial magnetic field.
        phi (np.ndarray): 1D longitude grid in radians.
        theta (np.ndarray): 1D colatitude grid in radians.
        iterations (int): Number of iterations for nonlinear diffusion.
        apply_gaussian (bool): Whether to apply Gaussian filtering before diffusion.
        gaussian_sigma (float): Sigma used in Gaussian smoothing.
        dx_override (float): Spatial resolution in x-direction (default: 1.0).
        dy_override (float): Spatial resolution in y-direction (default: 1.0).
        tau (float): Time step for the nonlinear diffusion.

    Returns:
        tuple[np.ndarray, float]: Filtered Br field and final time step.
    """

    logger.info("Begin Filtering")

    dx = dx_override
    dy = dy_override

    if apply_gaussian:
        Br_smoothed = scipy.ndimage.gaussian_filter(Br, gaussian_sigma)
    else:
        Br_smoothed = Br.copy()

    Br_filtered, timestep = nonlinearDiffusionFilter(Br_smoothed, dx, dy, iterations, tau)

    logger.info("End Filtering")
    return Br_filtered, timestep


_LEGACY_NLD_KEYS = ("tau", "gaussian_sigma", "dx_override", "dy_override")
_SPHERICAL_NLD_KEYS = ("gaussian_sigma_deg", "lambda_g_per_deg")


def resolve_nld_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Select the NLD mode and its parameters from a configuration.

    ``diffusion_length_deg`` selects the spherical (physical) filter; the
    legacy pixel keys ``tau``, ``gaussian_sigma``, ``dx_override`` and
    ``dy_override`` are then rejected. Without it, the legacy pixel filter is
    used exactly as before and the spherical keys are rejected.

    Returns:
        Dict with ``mode`` (``"spherical"`` or ``"legacy"``) and the keyword
        arguments of the corresponding filter function.

    Raises:
        ValueError: On mixed legacy/spherical keys, or a missing
            ``gaussian_sigma_deg`` while the spherical pre-smoothing is on.
    """
    configured_iterations = config.get("iterations")
    apply_gaussian = _as_bool(config.get("apply_gaussian", True))
    diffusion_length_deg = config.get("diffusion_length_deg")
    if diffusion_length_deg is not None:
        mixed = [key for key in _LEGACY_NLD_KEYS if config.get(key) is not None]
        if mixed:
            raise ValueError(
                "diffusion_length_deg selects the spherical NLD; remove the legacy "
                f"pixel keys {mixed} (tau is derived from the diffusion length)."
            )
        gaussian_sigma_deg = config.get("gaussian_sigma_deg")
        if apply_gaussian and gaussian_sigma_deg is None:
            raise ValueError(
                "Spherical NLD with apply_gaussian=True needs gaussian_sigma_deg "
                "(pre-smoothing width in degrees), or set apply_gaussian=False."
            )
        return {
            "mode": "spherical",
            "diffusion_length_deg": float(diffusion_length_deg),
            # Number of AOS time steps; the spherical default only applies to
            # this new mode (legacy keeps 7).
            "iterations": int(
                SPHERICAL_DEFAULT_STEPS
                if configured_iterations is None
                else configured_iterations
            ),
            "apply_gaussian": apply_gaussian,
            "gaussian_sigma_deg": gaussian_sigma_deg,
            "lambda_g_per_deg": config.get("lambda_g_per_deg"),
        }
    mixed = [key for key in _SPHERICAL_NLD_KEYS if config.get(key) is not None]
    if mixed:
        raise ValueError(
            f"{mixed} only apply to the spherical NLD; set diffusion_length_deg "
            "to use it."
        )
    return {
        "mode": "legacy",
        "iterations": 7 if configured_iterations is None else configured_iterations,
        "apply_gaussian": apply_gaussian,
        "gaussian_sigma": config.get("gaussian_sigma", 1.0),
        "dx_override": config.get("dx_override", 1.0),
        "dy_override": config.get("dy_override", 1.0),
        "tau": config.get("tau", 5),
    }


def filter_radial_field_spherical(
    Br: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    diffusion_length_deg: float,
    iterations: int = SPHERICAL_DEFAULT_STEPS,
    apply_gaussian: bool = True,
    gaussian_sigma_deg: float | None = None,
    lambda_g_per_deg: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Physical NLD: great-circle Gaussian pre-smoothing + spherical diffusion.

    Args:
        Br (np.ndarray): 2D radial field.
        phi (np.ndarray): 1D uniform longitude centres (radians).
        theta (np.ndarray): 1D colatitude centres (radians), north to south.
        diffusion_length_deg (float): Diffusion length in degrees.
        iterations (int): Number of AOS time steps (default 28).
        apply_gaussian (bool): Apply the great-circle Gaussian first.
        gaussian_sigma_deg (float | None): Pre-smoothing width in degrees.
        lambda_g_per_deg (float | None): Contrast threshold in G/deg, or None
            for the automatic area-weighted MAD estimate.

    Returns:
        tuple[np.ndarray, dict]: Filtered field and solver information
        (``lambda_g_per_deg``, ``lambda_source``, ``tau_rad2``, ``steps``,
        ``scheme``...).
    """
    logger.info("Begin spherical NLD filtering")
    if apply_gaussian:
        Br = geodesic_gaussian_filtering(Br, phi, theta, gaussian_sigma_deg)
    filtered, info = spherical_nonlinear_diffusion(
        Br, phi, theta, diffusion_length_deg, iterations, lambda_g_per_deg
    )
    logger.info("End spherical NLD filtering")
    return filtered, info


def _apply_nld(
    Br: np.ndarray,
    Theta: np.ndarray,
    Phi: np.ndarray,
    settings: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Nonlinear-diffusion step of the shared pipeline (legacy or spherical)."""
    parameters = {k: v for k, v in settings.items() if k != "mode"}
    if settings["mode"] == "spherical":
        filtered, nld_info = filter_radial_field_spherical(
            Br, Phi[0, :], Theta[:, 0], **parameters
        )
        timestep = nld_info["tau_rad2"]
    else:
        filtered, timestep = filter_radial_field(
            Br, Phi[0, :], Theta[:, 0], **parameters
        )
        nld_info = None
    return filtered, {
        "timestep": timestep,
        "nld_mode": settings["mode"],
        "nld_info": nld_info,
    }


NLD_STEP = FilterStep(resolve_settings=resolve_nld_settings, apply=_apply_nld)
"""Nonlinear-diffusion step for ``processing.pipeline.run_magnetogram_pipeline``."""


def process_magnetogram_date(
    config: dict[str, Any],
    target_date,
    method_used: str = "NLD",
    output_path_fig: str | None = None,
    unique_names: bool = False,
) -> dict[str, Any]:
    """Process one target time with the nonlinear diffusion pipeline.

    Runs the shared pipeline (``processing.pipeline.run_magnetogram_pipeline``):
    read or interpolate the magnetogram, effective time, optional Stonyhurst
    rotation, optional net-flux correction (``flux_correction_stage``),
    nonlinear diffusion, ``/2.2 * amp``, boundary file and optional figure.

    Args:
        config (dict[str, Any]): Processing configuration: the common keys of
            ``run_magnetogram_pipeline`` plus ``iterations`` and
            ``apply_gaussian``; legacy pixel mode: ``tau``, ``gaussian_sigma``,
            ``dx_override``, ``dy_override``; spherical mode (selected by
            ``diffusion_length_deg``): ``gaussian_sigma_deg`` and optional
            ``lambda_g_per_deg``.
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
        stencil, optional ``Br_linear``, final diffusion timestep (pixel units
        in legacy mode, rad^2 in spherical mode), ``nld_mode``, ``nld_info``
        (spherical mode: lambda and solver details), rotation angle and
        ``flux_correction_stage``.
    """
    return run_magnetogram_pipeline(
        config,
        target_date,
        NLD_STEP,
        method_used=method_used,
        output_path_fig=output_path_fig,
        unique_names=unique_names,
    )


def process_config(config: dict[str, Any], method_used: str = "NLD") -> list[dict[str, Any]]:
    """Process all target times described by one nonlinear diffusion config.

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
    label = "nld_25_7"
    output_dir = os.path.join(base_output_dir, label)
    figure_output_dir = os.path.join(base_output_dir, "images")


    configs = [{
        "date": "2011-09-09T01:47:05",
        "custom_magnetogram": r"C:\Users\luisl\Desktop\testmagnetogram\ai_magnetogram\AI_synopt_20260801_162400_TAI.fits",
        "amp": 1,
        "write_map": False,
        "show_map": True,
        "visu_type": "sinlat",
        "alpha": 3 * 10 ** (-6),
        "carrington": False,
        "rotate_to_stonyhurst": True,
        "interpolation": False,
        "interpolation_order": 2,
        "resize": False,
        "flux_correct": False,
        "flux_correction_method": "surface_mean", #surface_mean' or 'polarity_scaling'
        "flux_correction_stage": "after",  # "after" (default) or "before"
        "map_type": "hmi_polfil",
        "output_dir": output_dir,
        "download_dir": output_dir,
        "output_path_fig": os.path.join(figure_output_dir, f"{label}_nld.png"),
        "drms_email": "luis.linan@kuleuven.be",
        "apply_gaussian": True,
        "gaussian_sigma": 1.0,
        "tau": 25,
        "iterations": 7,
        # Spherical (physical) NLD: set diffusion_length_deg and remove "tau",
        # "gaussian_sigma", "dx_override" and "dy_override" (pixel keys).
        # "diffusion_length_deg": 2.0,  # diffusion length L in degrees (1 deg ~ 12.15 Mm)
        # "gaussian_sigma_deg": 0.5,    # geodesic pre-smoothing width in degrees (needed if apply_gaussian=True)
        # "lambda_g_per_deg": None,     # contrast threshold in G/deg; None = automatic area-weighted MAD
        # "iterations": 28,             # AOS steps (accuracy only: ~1 % error with 28)
        }]

    # for time evolving add : cadence_hours and total_hours to the config dictionary, e.g.:
    # "cadence_hours": 3,
    # "total_hours": 72,

    for config in configs:
        try:
            process_config(config, method_used="NLD")
        except Exception as exc:
            logger.warning(
                f'Failed to process {config["date"]} and {config["map_type"]}: {exc}'
            )
            continue

    base_output_dir = r"C:\Users\luisl\Desktop\testmagnetogram\test"
    label = "nld_5_25"
    output_dir = os.path.join(base_output_dir, label)
    figure_output_dir = os.path.join(base_output_dir, "images")


    configs = [{
        "date": "2011-09-09T01:47:05",
        "custom_magnetogram": r"C:\Users\luisl\Desktop\testmagnetogram\ai_magnetogram\AI_synopt_20260801_162400_TAI.fits",
        "amp": 1,
        "write_map": False,
        "show_map": True,
        "visu_type": "sinlat",
        "alpha": 3 * 10 ** (-6),
        "carrington": False,
        "rotate_to_stonyhurst": True,
        "interpolation": False,
        "interpolation_order": 2,
        "resize": False,
        "flux_correct": False,
        "flux_correction_method": "surface_mean", #surface_mean' or 'polarity_scaling'
        "flux_correction_stage": "after",  # "after" (default) or "before"
        "map_type": "hmi_polfil",
        "output_dir": output_dir,
        "download_dir": output_dir,
        "output_path_fig": os.path.join(figure_output_dir, f"{label}_nld.png"),
        "drms_email": "luis.linan@kuleuven.be",
        "apply_gaussian": True,
        "gaussian_sigma": 1.0,
        "tau": 5,
        "iterations": 25
        }]

    # for time evolving add : cadence_hours and total_hours to the config dictionary, e.g.:
    # "cadence_hours": 3,
    # "total_hours": 72,

    for config in configs:
        try:
            process_config(config, method_used="NLD")
        except Exception as exc:
            logger.warning(
                f'Failed to process {config["date"]} and {config["map_type"]}: {exc}'
            )
            continue
