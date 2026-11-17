"""
Preprocess magnetograms with surface-area-weighted Gaussian smoothing.

This module uses the shared acquisition, reading, coordinate, rotation, flux,
plotting, and boundary-writing subpackages. Its specific processing step
applies the spherical Gaussian kernel implemented in
``filters.gaussian_filter``.

Author: Jose Murteira
Cleaned and modularized by: Luis
"""
from typing import Any
import os

import numpy as np

from coconut_tools.magnetogram.filters.gaussian_filter import (
    gaussian_filtering,
    geodesic_gaussian_filtering,
)
from coconut_tools.magnetogram.processing.pipeline import (
    FilterStep,
    run_config,
    run_magnetogram_pipeline,
)
from coconut_tools.tools.logger_config import setup_logger

logger = setup_logger(__name__)


def resolve_gaussian_width(config: dict[str, Any]) -> tuple[int, float | None]:
    """Return ``(template_size, sigma_deg)`` from a configuration.

    ``sigma_deg`` (heliographic degrees) selects the physical great-circle
    Gaussian. Without it, the legacy pixel template ``template_size``
    (default 9) is used exactly as before.

    Raises:
        ValueError: If both ``sigma_deg`` and ``template_size`` are given.
    """
    sigma_deg = config.get("sigma_deg")
    if sigma_deg is not None and config.get("template_size") is not None:
        raise ValueError(
            "Give either sigma_deg (physical Gaussian, degrees) or template_size "
            "(legacy pixel template), not both."
        )
    return config.get("template_size", 9), sigma_deg


def filter_radial_field(
    Br: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    template_size: int = 9,
    sigma_deg: float | None = None,
) -> np.ndarray:
    """Apply Gaussian smoothing to a radial field.

    Args:
        Br (np.ndarray): 2D radial magnetic-field map.
        phi (np.ndarray): 1D endpoint-free longitude centres in radians.
        theta (np.ndarray): 1D colatitude centres in radians, north to south.
        template_size (int): Odd side length of the legacy pixel template,
            used when ``sigma_deg`` is None (historical behaviour).
        sigma_deg (float | None): Standard deviation in heliographic degrees.
            When given, the great-circle Gaussian with source-side (flux
            conserving) normalisation is used and ``template_size`` ignored.

    Returns:
        np.ndarray: Smoothed radial magnetic field.
    """
    logger.info("Begin Gaussian filtering")
    if sigma_deg is not None:
        logger.info("Great-circle Gaussian, sigma = %.4g deg", float(sigma_deg))
        filtered = geodesic_gaussian_filtering(Br, phi, theta, sigma_deg)
    else:
        logger.info("Legacy pixel Gaussian template, template_size = %s", template_size)
        filtered = gaussian_filtering(Br, phi, theta, template_size=template_size)
    logger.info("End Gaussian filtering")
    return filtered


def _resolve_gaussian_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Return the Gaussian widths of a configuration (see ``resolve_gaussian_width``)."""
    template_size, sigma_deg = resolve_gaussian_width(config)
    return {"template_size": template_size, "sigma_deg": sigma_deg}


def _apply_gaussian(
    Br: np.ndarray,
    Theta: np.ndarray,
    Phi: np.ndarray,
    settings: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Gaussian filtering step of the shared pipeline."""
    filtered = filter_radial_field(
        Br,
        Phi[0, :],
        Theta[:, 0],
        template_size=settings["template_size"],
        sigma_deg=settings["sigma_deg"],
    )
    sigma_deg = settings["sigma_deg"]
    return filtered, {
        "template_size": None if sigma_deg is not None else settings["template_size"],
        "sigma_deg": sigma_deg,
    }


GAUSSIAN_STEP = FilterStep(
    resolve_settings=_resolve_gaussian_settings,
    apply=_apply_gaussian,
)
"""Gaussian filter step for ``processing.pipeline.run_magnetogram_pipeline``."""


def process_magnetogram_date(
    config: dict[str, Any],
    target_date,
    method_used: str = "Gaussian",
    output_path_fig: str | None = None,
    unique_names: bool = False,
) -> dict[str, Any]:
    """Process one target time with the Gaussian-smoothing pipeline.

    Runs the shared pipeline (``processing.pipeline.run_magnetogram_pipeline``):
    read or interpolate the magnetogram, effective time, optional Stonyhurst
    rotation, optional net-flux correction (``flux_correction_stage``), Gaussian
    filter, ``/2.2 * amp``, boundary file and optional figure.

    Args:
        config (dict[str, Any]): Processing configuration: the common keys of
            ``run_magnetogram_pipeline`` plus ``template_size`` (legacy pixel
            width) or ``sigma_deg`` (physical width in degrees).
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
        stencil, optional ``Br_linear``, ``template_size``/``sigma_deg``,
        rotation angle and ``flux_correction_stage``.
    """
    return run_magnetogram_pipeline(
        config,
        target_date,
        GAUSSIAN_STEP,
        method_used=method_used,
        output_path_fig=output_path_fig,
        unique_names=unique_names,
    )


def process_config(config: dict[str, Any], method_used: str = "Gaussian") -> list[dict[str, Any]]:
    """Process all target times described by one Gaussian smoothing config.

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
    label = "gaussian_smoothing_ts_35"
    output_dir = os.path.join(base_output_dir, label)
    figure_output_dir = os.path.join(base_output_dir, "images")


    configs =[ {"date": "2011-09-09T01:47:05",
        "custom_magnetogram": r"C:\Users\luisl\Desktop\testmagnetogram\ai_magnetogram\AI_synopt_20260801_162400_TAI.fits",
        "amp": 1,
        "write_map": False,
        "show_map": True,
        "visu_type": "sinlat",
        "carrington": False,
        "rotate_to_stonyhurst": False,
        "interpolation": False,
        "interpolation_order": 2,
        "resize": True,
        "flux_correct": False,
        "flux_correction_method": "surface_mean", #surface_mean' or 'polarity_scaling'
        "flux_correction_stage": "after",  # "after" (default) or "before"
        "map_type": "hmi_polfil",
        "output_dir": output_dir,
        "download_dir": output_dir,
        "output_path_fig": os.path.join(figure_output_dir, f"{label}.png"),
        "drms_email": "luis.linan@kuleuven.be",
        "template_size": 35,
        # "sigma_deg": 2.0,  # geodesic Gaussian width in degrees (1 deg ~ 12.15 Mm); remove "template_size" to use it
        }]

    # for time evolving add : cadence_hours and total_hours to the config dictionary, e.g.:
    # "cadence_hours": 3,
    # "total_hours": 72,

    for config in configs:
        try:
            process_config(config, method_used="Gaussian")
        except Exception as exc:
            logger.warning(
                f'Failed to process {config["date"]} and {config["map_type"]}: {exc}'
            )
            continue
