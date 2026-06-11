"""
Local Weighted Filtering Interface for Radial Magnetogram Field (Br)

This script applies a local adaptive filter based on spatial and radiometric
proximity, using a Gaussian kernel controlled by `alpha` and `Rn`.

Author: Jose Murteira
Cleaned and modularized by: Luis
"""

import numpy as np
import scipy.ndimage
import logging
from typing import Any

from coconut_tools.magnetogram.local_weigh_filter import filter3
from coconut_tools.magnetogram.sph_filtering import (
    apply_configured_longitude_rotation,
    build_processing_dates,
    correct_net_flux,
    generate_output_and_interpolation_map_names,
    generate_output_and_map_names,
    magnetogram_display_date,
    parse_iso_datetime,
    plot_maps,
    read_magnetogram,
    read_interpolated_magnetogram,
    resolve_figure_path,
    write_bc_file,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def filter_radial_field_weighted(
    Br: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    alpha_factor: float,
    Rn: float,
    sig: float = 0.0,
    write_gaussian_prepass: bool = False
):
    """
    Apply local weighted filtering to radial magnetic field map.

    Args:
        Br (np.ndarray): Input magnetic field map.
        phi (np.ndarray): 1D array of longitudes in radians.
        theta (np.ndarray): 1D array of latitudes in radians.
        alpha_factor (float): Alpha controlling kernel sharpness.
        Rn (float): Radius of neighborhood in physical units.
        sig (float): Sigma of optional Gaussian smoothing.
        write_gaussian_prepass (bool): Whether to export Gaussian-smoothed version.

    Returns:
        np.ndarray: Filtered Br field.
    """
    dlong = phi[1] / (2 * np.pi)
    dlat = theta[1] / np.pi
    R_sun = 696.34e6

    dx = dlong * R_sun
    dy = dlat * R_sun
    delta_var = max(dx, dy)

    Br_smoothed = scipy.ndimage.gaussian_filter(Br, sig) if sig > 0 else Br.copy()

    if write_gaussian_prepass:
        np.save("Br_gaussian_prepass.npy", Br_smoothed)

    logger.info("Running local filter with alpha=%.2f, Rn=%.2f, dx=%.2f, dy=%.2f", alpha_factor, Rn, dx, dy)
    Br_filtered = filter3(Br_smoothed, delta_var, delta_var, alpha_factor, Rn)

    return Br_filtered


def _as_bool(value: Any) -> bool:
    """Convert common config values to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def process_magnetogram_date(
    config: dict[str, Any],
    target_date,
    method_used: str = "Yaroslavsky",
    output_path_fig: str | None = None,
) -> dict[str, Any]:
    """Process one target date with the local weighted filter.

    Args:
        config (dict[str, Any]): Processing configuration.
        target_date: Date to process.
        method_used (str): Method label included in output filenames.
        output_path_fig (str | None): Diagnostic figure path.

    Returns:
        dict[str, Any]: Paths and processing metadata.
    """
    map_type = config["map_type"]
    output_dir = config.get("output_dir", "../")
    download_dir = config.get("download_dir", output_dir)
    lmax = config.get("lmax", 20)
    r_st = config.get("r_st", 1.0)
    adapt_map = config.get("adapt_map", 6)
    write_map = _as_bool(config.get("write_map", True))
    show_map = _as_bool(config.get("show_map", True))
    visu_type = config.get("visu_type", "sinlat")
    alpha = config.get("alpha", 1.0)
    Rn = config.get("Rn", 5.0)
    sig = config.get("sig", 0.0)
    interpolation_order = config.get("interpolation_order", config.get("Interp_order", 2))
    use_interpolation = _as_bool(config.get("interpolation", map_type in {"GONG", "ADAPT"}))
    rotate_to_stonyhurst = _as_bool(config.get("rotate_to_stonyhurst", True))

    if use_interpolation and map_type in {"GONG", "ADAPT"}:
        output_name, local_files, selection = generate_output_and_interpolation_map_names(
            target_date,
            map_type,
            output_dir,
            lmax,
            method_used=method_used,
            download_dir=download_dir,
        )
        Br, Theta, Phi, Br_linear = read_interpolated_magnetogram(
            local_files,
            map_type,
            selection,
            adapt_map=adapt_map,
            interpolation_order=interpolation_order,
        )
        local_file = local_files
    else:
        output_name, local_file = generate_output_and_map_names(
            target_date,
            map_type,
            output_dir,
            lmax,
            method_used=method_used,
        )
        Br, Theta, Phi = read_magnetogram(local_file, map_type, adapt_map)
        Br_linear = None
        selection = None

    figure_date = magnetogram_display_date(
        local_file[0] if isinstance(local_file, list) else local_file,
        map_type,
        target_date,
        interpolated=use_interpolation and map_type in {"GONG", "ADAPT"},
    )
    Br, Br_linear, rotation_angle = apply_configured_longitude_rotation(
        Br,
        Br_linear,
        local_file,
        map_type,
        target_date,
        use_interpolation,
        rotate_to_stonyhurst,
    )

    if _as_bool(config.get("flux_correct", False)):
        Br = correct_net_flux(Br, Theta[:, 0])

    Br_filtered = filter_radial_field_weighted(
        Br,
        Phi[0, :],
        Theta[:, 0],
        alpha,
        Rn,
        sig,
        write_gaussian_prepass=_as_bool(config.get("write_gaussian_prepass", False)),
    )

    if write_map:
        write_bc_file(output_name, Br_filtered, Theta[:, 0], Phi[0, :], r_st)

    if show_map:
        figure_path = resolve_figure_path(
            output_path_fig,
            output_dir,
            map_type,
            target_date,
        )
        plot_maps(
            Br,
            Br_filtered,
            Theta[:, 0],
            Phi[0, :],
            map_type,
            visu_type,
            output_path=figure_path,
            date=figure_date,
        )
    else:
        figure_path = None

    return {
        "date": parse_iso_datetime(target_date),
        "magnetogram_date": figure_date,
        "output_name": output_name,
        "local_file": local_file,
        "figure_path": figure_path,
        "selection": selection,
        "Br_linear": Br_linear,
        "rotation_angle": rotation_angle,
    }


def process_config(config: dict[str, Any], method_used: str = "Yaroslavsky") -> list[dict[str, Any]]:
    """Process a single-date or multi-date Yaroslavsky configuration.

    Args:
        config (dict[str, Any]): Processing configuration.
        method_used (str): Method label included in output filenames.

    Returns:
        list[dict[str, Any]]: Per-date processing results.
    """
    target_dates = build_processing_dates(
        config["date"],
        cadence_hours=config.get("cadence_hours", config.get("candence")),
        total_hours=config.get("total_hours"),
    )
    output_path_fig = config.get("output_path_fig")
    use_unique_figures = len(target_dates) > 1
    results = []
    for target_date in target_dates:
        figure_path = resolve_figure_path(
            output_path_fig,
            config.get("output_dir", "../"),
            config["map_type"],
            target_date,
            use_unique_name=use_unique_figures,
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
    # Multi-date example:
    # config = {
    #     "date": "2025-10-09T18:00:00",
    #     "map_type": "GONG",
    #     "cadence_hours": 3,
    #     "total_hours": 72,
    #     "interpolation": True,
    #     "interpolation_order": 2,
    #     "flux_correct": True,
    #     "lmax": 20,
    #     "alpha": 1.4,
    #     "Rn": 2,
    #     "sig": 1.5,
    #     "write_map": True,
    #     "show_map": True,
    #     "visu_type": "sinlat",
    #     "output_dir": "../COCONUT/",
    #     "download_dir": "../raw/",
    # }
    # process_config(config, method_used="Yaroslavsky")
    """
    configs = [
        {
            "date": '2020-12-07T15:00:00', "map_type": 'HMI_small',
            "lmax": 15, "amp": 1, "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "../", "output_path_fig": "../hmi_20201207_weighted.png",
            "alpha": 1.4, "Rn": 5.0, "sig": 1.0
        },
        {
            "date": '2022-03-11T12:00:00', "map_type": 'GONG',
            "lmax": 15, "amp": 0.8, "write_map": False, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "../", "output_path_fig": "../gong_20220311_weighted.png",
            "alpha": 1.2, "Rn": 6.0, "sig": 1.0
        },
        {
            "date": '2023-08-15T00:00:00', "map_type": 'ADAPT', "adapt_map": 4,
            "lmax": 15, "amp": 1.2, "write_map": True, "show_map": False,
            "visu_type": "sinlat",
            "output_dir": "../", "output_path_fig": "../adapt_20230815_weighted.png",
            "alpha": 1.0, "Rn": 5.0, "sig": 1.5
        },
        {
            "date": '2024-09-12T06:00:00', "map_type": 'WSO',
            "lmax": 15, "amp": 1.0, "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "../", "output_path_fig": "../wso_20240912_weighted.png",
            "alpha": 1.1, "Rn": 7.0, "sig": 0.0
        }
    ]
    """
    configs = [
        {
            "date": '2013-03-13T12:00:00', "map_type": 'GONG',
            "lmax": 15, "amp": 1, "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/euhforia/magnetogram/yaroslavsky/", "output_path_fig": "E:/euhforia/magnetogram/yaroslavsky/GONG_20130313T120000.png",
            "alpha": 1.4, "Rn" : 2, "sig": 1.5
        }]

    for config in configs:
        try:
            process_config(config, method_used="Yaroslavsky")
        except Exception as exc:
            logger.warning(
                f'Failed to process {config["date"]} and {config["map_type"]}: {exc}'
            )
