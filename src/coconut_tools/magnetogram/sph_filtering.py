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
from coconut_tools.magnetogram.io.downloads import (
    build_output_name,
    build_processing_dates,
    default_figure_path,
    generate_output_and_interpolation_map_names,
    generate_output_and_map_names,
    is_gong_temporal_map_type,
    magnetogram_display_date,
    magnetogram_effective_date,
    normalize_map_type,
    parse_iso_datetime,
    resolve_figure_path,
)
from coconut_tools.magnetogram.io.readers import (
    read_interpolated_magnetogram,
    read_magnetogram,
)
from coconut_tools.magnetogram.io.writers import write_bc_file
from coconut_tools.magnetogram.processing.flux_balance import (
    _flux_summary,
    _surface_mean_area,
    correct_net_flux,
)
from coconut_tools.magnetogram.processing.longitude import apply_configured_longitude_rotation
from coconut_tools.magnetogram.processing.spherical_harmonics import project_and_reconstruct
from coconut_tools.magnetogram.visualization.plotting import plot_maps
from coconut_tools.tools.logger_config import setup_logger

logger = setup_logger(__name__)


def process_magnetogram_date(
    config: dict[str, Any],
    target_date: str | datetime,
    method_used: str = "sph",
    output_path_fig: str | None = None,
) -> dict[str, Any]:
    """Process one target time through the spherical-harmonic pipeline."""
    custom_magnetogram = config.get("custom_magnetogram")
    map_type = (
        "custom"
        if custom_magnetogram is not None
        else normalize_map_type(config["map_type"])
    )
    if custom_magnetogram is not None:
        custom_magnetogram = os.fspath(custom_magnetogram)
    output_dir = config.get("output_dir", "../")
    download_dir = config.get("download_dir", output_dir)
    lmax = config.get("lmax", 20)
    amp = config.get("amp", 1)
    r_st = config.get("r_st", 1.0)
    adapt_map = config.get("adapt_map", 6)
    write_map = _as_bool(config.get("write_map", True))
    show_map = _as_bool(config.get("show_map", True))
    visu_type = config.get("visu_type", "sinlat")
    alpha = config.get("alpha", 0)
    interpolation_order = config.get(
        "interpolation_order",
        config.get("Interp_order", 2),
    )
    requested_interpolation = _as_bool(
        config.get(
            "interpolation",
            is_gong_temporal_map_type(map_type) or map_type == "ADAPT",
        )
    )
    use_interpolation = requested_interpolation and custom_magnetogram is None
    if custom_magnetogram is not None and requested_interpolation:
        logger.info("Temporal interpolation is disabled for a custom magnetogram.")
    rotate_to_stonyhurst = _as_bool(config.get("rotate_to_stonyhurst", True))
    flux_correction_method = config.get("flux_correction_method", "surface_mean")
    drms_email = config.get("drms_email", config.get("jsoc_email"))
    resize = _as_bool(config.get("resize", False))

    interpolated = use_interpolation and (
        is_gong_temporal_map_type(map_type)
        or map_type in {"ADAPT", "HMI_hourly", "HMI_fdt"}
    )

    if custom_magnetogram is not None:
        output_name = build_output_name(
            map_type,
            output_dir,
            method_used=method_used,
        )
        local_file = custom_magnetogram
        Br, Theta, Phi = read_magnetogram(
            local_file,
            map_type,
            adapt_map,
            resize=resize,
        )
        Br_linear = None
        selection = None
    elif interpolated:
        output_name, local_files, selection = (
            generate_output_and_interpolation_map_names(
                target_date,
                map_type,
                output_dir,
                method_used=method_used,
                download_dir=download_dir,
                drms_email=drms_email,
            )
        )
        Br, Theta, Phi, Br_linear = read_interpolated_magnetogram(
            local_files,
            map_type,
            selection,
            adapt_map=adapt_map,
            interpolation_order=interpolation_order,
            resize=resize,
        )
        local_file = local_files
    else:
        output_name, local_file = generate_output_and_map_names(
            target_date,
            map_type,
            output_dir,
            method_used=method_used,
            drms_email=drms_email,
        )
        Br, Theta, Phi = read_magnetogram(
            local_file,
            map_type,
            adapt_map,
            resize=resize,
        )
        Br_linear = None
        selection = None

    source_file = local_file[0] if isinstance(local_file, list) else local_file
    effective_date = magnetogram_effective_date(
        source_file,
        map_type,
        target_date,
        interpolated=interpolated,
    )
    logger.info(
        "Magnetogram timing: target_time: %s, effective_time: %s",
        parse_iso_datetime(target_date).isoformat(),
        effective_date.isoformat(),
    )
    figure_date = magnetogram_display_date(
        source_file,
        map_type,
        target_date,
        interpolated=interpolated,
    )

    Br, Br_linear, rotation_angle = apply_configured_longitude_rotation(
        Br,
        Br_linear,
        local_file,
        map_type,
        target_date,
        use_interpolation,
        rotate_to_stonyhurst,
        effective_date=effective_date,
        resize=resize,
    )

    if _as_bool(config.get("flux_correct", False)):
        Br = correct_net_flux(
            Br,
            Theta[:, 0],
            Phi[0, :],
            method=flux_correction_method,
        )

    Br_mode, coefbr = project_and_reconstruct(Br, Theta, Phi, lmax, amp, alpha)

    if write_map:
        write_bc_file(output_name, Br_mode, Theta[:, 0], Phi[0, :], r_st)

    if show_map:
        figure_path = output_path_fig or default_figure_path(
            output_dir,
            map_type,
            effective_date,
        )
        plot_maps(
            Br,
            Br_mode,
            Theta[:, 0],
            Phi[0, :],
            map_type,
            visu_type,
            output_path=figure_path,
            date=figure_date,
        )
    else:
        figure_path = None

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

    return {
        "date": parse_iso_datetime(target_date),
        "effective_date": effective_date,
        "magnetogram_date": figure_date,
        "output_name": output_name,
        "local_file": local_file,
        "figure_path": figure_path,
        "selection": selection,
        "Br_linear": Br_linear,
        "coefbr": coefbr,
        "rotation_angle": rotation_angle,
    }


def process_config(
    config: dict[str, Any],
    method_used: str = "sph",
) -> list[dict[str, Any]]:
    """Process every target time described by one filter configuration."""
    target_dates = build_processing_dates(
        config["date"],
        cadence_hours=config.get("cadence_hours", config.get("candence")),
        total_hours=config.get("total_hours"),
    )
    output_path_fig = config.get("output_path_fig")
    use_unique_figures = len(target_dates) > 1
    figure_map_type = (
        "custom"
        if config.get("custom_magnetogram") is not None
        else config["map_type"]
    )
    results = []
    for target_date in target_dates:
        figure_path = (
            resolve_figure_path(
                output_path_fig,
                config.get("output_dir", "../"),
                figure_map_type,
                target_date,
                use_unique_name=use_unique_figures,
            )
            if output_path_fig
            else None
        )
        results.append(
            process_magnetogram_date(
                config,
                target_date,
                method_used=method_used,
                output_path_fig=figure_path,
            )
        )
    return results


if __name__ == "__main__":
    base_output_dir = r"C:\Users\luisl\Desktop\testmagnetogram\hyunji"
    label = "AI_final_not_rotated"
    output_dir = os.path.join(base_output_dir, label)
    figure_output_dir = os.path.join(base_output_dir, "images")

    config = {
        "date": "2026-05-09T01:47:05",
        "custom_magnetogram": r"C:\Users\luisl\Desktop\AI_synopt_20260801_162400_TAI.fits",
        "lmax": 20,
        "amp": 1,
        "write_map": True,
        "show_map": True,
        "visu_type": "sinlat",
        "alpha": 3 * 10 ** (-6),
        "rotate_to_stonyhurst": False,
        "interpolation": False,
        "interpolation_order": 2,
        "resize": True,
        "flux_correct": False,
        "flux_correction_method": "surface_mean",
        "map_type": "hmi_hourly",
        "output_dir": output_dir,
        "download_dir": output_dir,
        "output_path_fig": os.path.join(figure_output_dir, f"{label}.png"),
        "drms_email": "luis.linan@kuleuven.be",
    }
    process_config(config, method_used="sph")
