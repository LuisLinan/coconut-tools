"""
Preprocess magnetograms with nonlinear diffusion filtering.

This module shares the magnetogram download, reading, temporal interpolation,
effective-time handling, Stonyhurst rotation, flux correction, plotting, and
COCONUT boundary writing utilities from ``sph_filtering``. Its specific
processing step applies optional Gaussian smoothing followed by the
Perona-Malik nonlinear diffusion filter implemented in
``nonlinear_diffusion_filter``.

Author: Jose Murteira
Cleaned and modularized by: Luis
"""
from typing import Any
import os

import numpy as np
import scipy.ndimage
from coconut_tools.magnetogram.magnetogram_download import (
    build_processing_dates,
    generate_output_and_interpolation_map_names,
    generate_output_and_map_names,
    is_gong_temporal_map_type,
    magnetogram_effective_date,
    magnetogram_display_date,
    normalize_map_type,
    parse_iso_datetime,
    resolve_figure_path,
)
from coconut_tools.magnetogram.nonlinear_diffusion_filter import nonlinearDiffusionFilter
from coconut_tools.magnetogram.sph_filtering import (
    _as_bool,
    apply_configured_longitude_rotation,
    correct_net_flux,
    plot_maps,
    read_magnetogram,
    read_interpolated_magnetogram,
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


def process_magnetogram_date(
    config: dict[str, Any],
    target_date,
    method_used: str = "NLD",
    output_path_fig: str | None = None,
) -> dict[str, Any]:
    """Process one target time with the nonlinear diffusion pipeline.

    The function downloads or reuses a magnetogram, optionally interpolates a
    four-map stencil, computes and logs the effective magnetogram time,
    optionally rotates to Stonyhurst, optionally balances net flux, applies the
    nonlinear diffusion filter, writes the boundary file, and optionally saves
    a diagnostic figure.

    Args:
        config (dict[str, Any]): Processing configuration. Common keys are
            ``map_type``, ``output_dir``, ``download_dir``, ``r_st``,
            ``amp``, ``adapt_map``, ``write_map``, ``show_map``, ``visu_type``,
            ``interpolation_order``, ``interpolation``,
            ``rotate_to_stonyhurst``, ``flux_correct``,
            ``flux_correction_method``, ``drms_email`` or ``jsoc_email``,
            ``tau``, ``iterations``, ``apply_gaussian``, ``gaussian_sigma``,
            ``dx_override``, and ``dy_override``.
        target_date: Requested processing time.
        method_used (str): Method label used in output filenames.
        output_path_fig (str | None): Explicit diagnostic figure path. If
            omitted, the figure name is built from the effective time.

    Returns:
        dict[str, Any]: Processing metadata, including target ``date``,
        ``effective_date``, output paths, selected local file or interpolation
        stencil, optional ``Br_linear``, final diffusion timestep, and rotation
        angle.
    """
    map_type = normalize_map_type(config["map_type"])
    output_dir = config.get("output_dir", "../")
    download_dir = config.get("download_dir", output_dir)
    r_st = config.get("r_st", 1.0)
    amp = config.get("amp", 1)
    adapt_map = config.get("adapt_map", 6)
    write_map = _as_bool(config.get("write_map", True))
    show_map = _as_bool(config.get("show_map", True))
    visu_type = config.get("visu_type", "sinlat")
    interpolation_order = config.get("interpolation_order", config.get("Interp_order", 2))
    use_interpolation = _as_bool(
        config.get("interpolation", is_gong_temporal_map_type(map_type) or map_type == "ADAPT")
    )
    rotate_to_stonyhurst = _as_bool(config.get("rotate_to_stonyhurst", True))
    flux_correction_method = config.get("flux_correction_method", "surface_mean")
    drms_email = config.get("drms_email", config.get("jsoc_email"))
    resize = _as_bool(config.get("resize", False))

    tau = config.get("tau", 5)
    iterations = config.get("iterations", 7)
    apply_gaussian = _as_bool(config.get("apply_gaussian", True))
    gaussian_sigma = config.get("gaussian_sigma", 1.0)
    dx_override = config.get("dx_override", 1.0)
    dy_override = config.get("dy_override", 1.0)

    interpolated = use_interpolation and (
        is_gong_temporal_map_type(map_type) or map_type == "ADAPT"
    )

    if interpolated:
        output_name, local_files, selection = generate_output_and_interpolation_map_names(
            target_date,
            map_type,
            output_dir,
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
            method_used=method_used,
            drms_email=drms_email,
        )
        Br, Theta, Phi = read_magnetogram(local_file, map_type, adapt_map, resize=resize)
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
        resize=resize and not interpolated,
    )

    if _as_bool(config.get("flux_correct", False)):
        Br = correct_net_flux(
            Br,
            Theta[:, 0],
            Phi[0, :],
            method=flux_correction_method,
        )

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

    Br_filtered = Br_filtered / 2.2
    Br_filtered *= amp

    if write_map:
        write_bc_file(output_name, Br_filtered, Theta[:, 0], Phi[0, :], r_st)

    if show_map:
        figure_path = resolve_figure_path(
            output_path_fig,
            output_dir,
            map_type,
            effective_date,
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
        "effective_date": effective_date,
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
    """Process all target times described by one nonlinear diffusion config.

    With only ``date`` set, one target time is processed. With
    ``cadence_hours`` and ``total_hours``, the function builds a time sequence
    and processes each target independently. When ``output_path_fig`` is omitted
    each figure is named from the effective magnetogram time.

    Args:
        config (dict[str, Any]): Processing configuration.
        method_used (str): Method label used in output filenames.

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
        figure_path = (
            resolve_figure_path(
                output_path_fig,
                config.get("output_dir", "../"),
                config["map_type"],
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

    base_output_dir = r"C:\Users\luisl\Desktop\testmagnetogram"
    label = "NLD"
    output_dir = os.path.join(base_output_dir, label)
    figure_output_dir = os.path.join(base_output_dir, "images")


    configs = [
        {
        "date": "2026-07-01T06:17:00",
        "amp": 1,
        "write_map": True,
        "show_map": True,
        "visu_type": "sinlat",
        "rotate_to_stonyhurst": True,
        "interpolation": False,
        "interpolation_order": 2,
        "resize": False,
        "flux_correct": False,
        "flux_correction_method": "surface_mean", #surface_mean' or 'polarity_scaling'
        "map_type": "GONG_mrbqs",
        "adapt_map": 6,
        "output_dir": output_dir,
        "download_dir": output_dir,
        "output_path_fig": os.path.join(figure_output_dir, f"{label}.png"),
        "drms_email": "luis.linan@kuleuven.be",
        "apply_gaussian": True,
        "gaussian_sigma": 1.0,
        "tau": 5,
        "iterations": 7
        },
    ]

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
