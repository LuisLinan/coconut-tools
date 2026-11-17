"""Spherical-harmonic magnetogram pipeline entry point.

This launch module contains only pipeline orchestration. Reusable operations
live in the ``core``, ``io``, ``processing``, and ``visualization``
subpackages.
"""

import os
from datetime import datetime
from typing import Any

import numpy as np

from coconut_tools.magnetogram.core.config import _as_bool
from coconut_tools.magnetogram.gaussian_smoothing import (
    filter_radial_field,
    resolve_gaussian_width,
)
from coconut_tools.magnetogram.processing.flux_balance import (
    _flux_summary,
    _surface_mean_area,
)
from coconut_tools.magnetogram.processing.pipeline import (
    FilterStep,
    run_config,
    run_magnetogram_pipeline,
)
from coconut_tools.magnetogram.processing.spherical_harmonics import project_and_reconstruct
from coconut_tools.tools.logger_config import setup_logger

logger = setup_logger(__name__)


def _resolve_sph_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Spherical-harmonic settings; logs the flux-correction notes of this filter."""
    flux_correct = _as_bool(config.get("flux_correct", False))
    flux_correction_method = config.get("flux_correction_method", "surface_mean")
    if config.get("flux_correction_stage") is not None:
        # The projection removes l=0, so Br_mode already has zero net flux:
        # a correction after the filter would be a no-op. Only "before" is
        # meaningful (and only polarity_scaling changes Br_mode).
        logger.info(
            "flux_correction_stage is ignored by the spherical-harmonic filter; "
            "the correction, if any, is applied before the projection."
        )
    if flux_correct and str(flux_correction_method).lower() in {
        "surface_mean", "mean", "subtract_mean"
    }:
        logger.info(
            "surface_mean flux correction does not change Br_mode: the "
            "spherical-harmonic projection already removes the l=0 monopole."
        )
    template_size, sigma_deg = resolve_gaussian_width(config)
    return {
        "lmax": config.get("lmax", 20),
        "amp": config.get("amp", 1),
        "alpha": config.get("alpha", 0),
        "gaussian_filtering": _as_bool(config.get("gaussian_filtering", False)),
        "template_size": template_size,
        "sigma_deg": sigma_deg,
    }


def _apply_sph(
    Br: np.ndarray,
    Theta: np.ndarray,
    Phi: np.ndarray,
    settings: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optional Gaussian pre-filter + spherical-harmonic projection.

    ``project_and_reconstruct`` applies the ``/2.2 * amp`` scaling itself.
    """
    if settings["gaussian_filtering"]:
        Br_filtered = filter_radial_field(
            Br,
            Phi[0, :],
            Theta[:, 0],
            template_size=settings["template_size"],
            sigma_deg=settings["sigma_deg"],
        )
    else:
        Br_filtered = Br
    Br_mode, coefbr = project_and_reconstruct(
        Br_filtered,
        Theta,
        Phi,
        settings["lmax"],
        settings["amp"],
        settings["alpha"],
    )
    return Br_mode, {"coefbr": coefbr}


def _log_br_mode(Br_mode: np.ndarray, Theta: np.ndarray, Phi: np.ndarray) -> None:
    """Log the flux and range diagnostics of the reconstructed field."""
    br_mode_area = _surface_mean_area(Theta[:, 0], Phi[0, :], Br_mode.shape)
    flux_positive, flux_negative, net_flux, imbalance_percent = _flux_summary(
        Br_mode,
        br_mode_area,
    )
    logger.info("Br_mode flux positive: %.6e", flux_positive)
    logger.info("Br_mode flux negative: %.6e", flux_negative)
    logger.info("Br_mode net flux: %.6e", net_flux)
    logger.info("Br_mode flux imbalance: %.6e %%", imbalance_percent)
    logger.info("Br_mode max: %.6e", np.max(Br_mode))
    logger.info("Br_mode mean: %.6e", np.mean(Br_mode))
    logger.info("Br_mode min: %.6e", np.min(Br_mode))


SPH_STEP = FilterStep(
    resolve_settings=_resolve_sph_settings,
    apply=_apply_sph,
    configurable_flux_stage=False,   # correction always before the projection
    scale_output=False,              # /2.2 * amp already applied by project_and_reconstruct
    log_final=_log_br_mode,
)
"""Spherical-harmonic step for ``processing.pipeline.run_magnetogram_pipeline``."""


def process_magnetogram_date(
    config: dict[str, Any],
    target_date: str | datetime,
    method_used: str = "sph",
    output_path_fig: str | None = None,
    unique_names: bool = False,
) -> dict[str, Any]:
    """Process one target time through the spherical-harmonic pipeline.

    Runs the shared pipeline (``processing.pipeline.run_magnetogram_pipeline``)
    with the spherical-harmonic step: optional flux correction before the
    projection, optional Gaussian pre-filter (``gaussian_filtering`` with
    ``template_size`` or ``sigma_deg``), projection up to ``lmax`` with
    damping ``alpha``, ``/2.2 * amp``.

    Args:
        config (dict[str, Any]): Processing configuration.
        target_date (str | datetime): Requested target time.
        method_used (str): Method label used in output filenames.
        output_path_fig (str | None): Figure file or directory, resolved with
            ``resolve_figure_path`` at the effective magnetogram time.
        unique_names (bool): Append the effective time to the ``.dat`` and
            figure names; set by ``process_config`` when several dates are
            processed so a time series does not overwrite its outputs.

    Returns:
        dict[str, Any]: Per-date metadata, including the written paths and
        ``coefbr``.
    """
    return run_magnetogram_pipeline(
        config,
        target_date,
        SPH_STEP,
        method_used=method_used,
        output_path_fig=output_path_fig,
        unique_names=unique_names,
    )


def process_config(
    config: dict[str, Any],
    method_used: str = "sph",
) -> list[dict[str, Any]]:
    """Process every target time described by one filter configuration.

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
    label = "sph_filtering_jose_gon"
    output_dir = os.path.join(base_output_dir, label)
    figure_output_dir = os.path.join(base_output_dir, "images")

    config = {
        "date": "2026-08-13T03:00:00",
        "custom_magnetogram": None,
        "lmax": 20,
        "amp": 1,
        "write_map": False,
        "show_map": True,
        "visu_type": "sinlat",
        "alpha": 3 * 10 ** (-6),
        "carrington": True,
        "rotate_to_stonyhurst": False,
        "interpolation": False,
        "interpolation_order": 2,
        "resize": True,
        "gaussian_filtering": False,
        "template_size": 55,
        "flux_correct": False,
        "flux_correction_method": "surface_mean",
        "map_type": "gong_mrzqs",
        "output_dir": output_dir,
        "download_dir": output_dir,
        "output_path_fig": os.path.join(figure_output_dir, f"{label}.png"),
        "drms_email": "luis.linan@kuleuven.be",
    }
    process_config(config, method_used="sph")
    r"""
    # Apply the spherical-harmonic filter to every supported non-custom, non-WSO map.
    reference_output_dir = r"C:\Users\luisl\Desktop\testmagnetogram\test_all\carrington"
    map_types = (
        "HMI_SYNC",
        "HMI_hourly",
        "ADAPT",
        "HMI_polfil",
        "HMI_small",
        "HMI_fdt",
        "GONG",
        "GONG_mrbqj",
        "GONG_mrbqs",
        "GONG_mrmqs",
        "GONG_mrnqs",
        "GONG_mrzqs",
    )
    completed_map_types = []
    failed_map_types = []
    for map_type in map_types:
        config = {
            "date": "2026-09-01T00:00:00",
            "map_type": map_type,
            "lmax": 20,
            "write_map": False,
            "show_map": True,
            "resize": True,
            "carrington": True,
            "rotate_to_stonyhurst": True,
            "interpolation": False,
            "gaussian_filtering": False,
            "output_dir": reference_output_dir,
            "download_dir": reference_output_dir,
            "output_path_fig": os.path.join(
                reference_output_dir,
            ),
            "drms_email": "luis.linan@kuleuven.be",
        }
        logger.info("Processing reference magnetogram: %s", map_type)
        try:
            process_config(config, method_used="sph")
        except Exception:
            failed_map_types.append(map_type)
            logger.exception(
                "Reference processing failed for %s at %s; continuing with the other map types.",
                map_type,
                config["date"],
            )
        else:
            completed_map_types.append(map_type)

    logger.info(
        "Reference processing finished: %d/%d map types completed. Successful types: %s",
        len(completed_map_types),
        len(map_types),
        ", ".join(completed_map_types) or "none",
    )
    if failed_map_types:
        logger.warning("Failed map types: %s", ", ".join(failed_map_types))
    """
