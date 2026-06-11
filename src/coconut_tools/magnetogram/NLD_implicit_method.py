"""
Filtering interface for radial magnetic field map (Br)

Applies a nonlinear diffusion filter followed by optional Gaussian smoothing.
Encapsulates preprocessing for Perona-Malik filtering of magnetogram data.

Author: Jose Murteira
Cleaned and modularized by: Luis
"""
import logging
from typing import Any

import numpy as np
import scipy.ndimage
from coconut_tools.magnetogram.nonlinear_diffusion_filter import nonlinearDiffusionFilter
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
    """
    Apply nonlinear diffusion filtering to the Br magnetogram field.

    Args:
        Br (np.ndarray): 2D array of the radial magnetic field.
        phi (np.ndarray): Array of longitudes (in radians).
        theta (np.ndarray): Array of latitudes (in radians).
        iterations (int): Number of iterations for nonlinear diffusion.
        apply_gaussian (bool): Whether to apply Gaussian filtering before diffusion.
        gaussian_sigma (float): Sigma used in Gaussian smoothing.
        dx_override (float): Spatial resolution in x-direction (default: 1.0).
        dy_override (float): Spatial resolution in y-direction (default: 1.0).
        tau (float): Time step for the nonlinear diffusion.

    Returns:
        Tuple[np.ndarray, float]: Filtered Br field and final time step.
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
    method_used: str = "NLD",
    output_path_fig: str | None = None,
) -> dict[str, Any]:
    """Process one target date with the nonlinear diffusion filter.

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
    interpolation_order = config.get("interpolation_order", config.get("Interp_order", 2))
    use_interpolation = _as_bool(config.get("interpolation", map_type in {"GONG", "ADAPT"}))
    rotate_to_stonyhurst = _as_bool(config.get("rotate_to_stonyhurst", True))

    tau = config.get("tau", 5)
    iterations = config.get("iterations", 7)
    apply_gaussian = _as_bool(config.get("apply_gaussian", True))
    gaussian_sigma = config.get("gaussian_sigma", 1.0)
    dx_override = config.get("dx_override", 1.0)
    dy_override = config.get("dy_override", 1.0)

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

    Br_filtered, timestep = filter_radial_field(
        Br,
        Phi[0, :],
        Theta[:, 0],
        iterations=iterations,
        apply_gaussian=apply_gaussian,
        gaussian_sigma=gaussian_sigma,
        dx_override=dx_override,
        dy_override=dy_override,
        tau=tau,
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
        "timestep": timestep,
        "rotation_angle": rotation_angle,
    }


def process_config(config: dict[str, Any], method_used: str = "NLD") -> list[dict[str, Any]]:
    """Process a single-date or multi-date nonlinear diffusion configuration.

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
    # --- Example runs ---
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
    #     "tau": 5,
    #     "iterations": 7,
    #     "apply_gaussian": True,
    #     "gaussian_sigma": 1.0,
    #     "write_map": True,
    #     "show_map": True,
    #     "visu_type": "sinlat",
    #     "output_dir": "../COCONUT/",
    #     "download_dir": "../raw/",
    # }
    # process_config(config, method_used="NLD")

    configs = [
        {
            "date": '2017-09-04T18:00:00', "map_type": 'GONG',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/euhforia/magnetogram/2017/nld/",
            "output_path_fig": "E:/euhforia/magnetogram/2017/nld/GONG.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2017-09-04T18:00:00', "map_type": 'HMI_polfil',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/euhforia/magnetogram/2017/nld/",
            "output_path_fig": "E:/euhforia/magnetogram/2017/nld/HMI_polfil.png",
            "tau": 5, "iterations": 7
        },
    ]

    configs = [
        {
        "date": "2012-07-13T00:00:00",
        "map_type": "GONG",
        "cadence_hours": 3,
        "total_hours": 240,
        "interpolation": True,
        "interpolation_order": 2,
        "flux_correct": False,
        "write_map": True,
        "show_map": True,
        "visu_type": "sinlat",
        "output_dir": "E:/COCONUT/2012/nld/",
        "output_path_fig": "E:/COCONUT/2012/nld/image/",
        "download_dir": "E:/COCONUT/2012/nld/raw/",
        "tau": 5, "iterations": 7
    }]

    for config in configs:
        try:
            process_config(config, method_used="NLD")
        except Exception as exc:
            logger.warning(
                f'Failed to process {config["date"]} and {config["map_type"]}: {exc}'
            )
            continue
