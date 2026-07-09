"""
Preprocess magnetograms with a local weighted Yaroslavsky-style filter.

This module shares the magnetogram download, reading, temporal interpolation,
effective-time handling, Stonyhurst rotation, flux correction, plotting, and
COCONUT boundary writing utilities from ``sph_filtering``. Its specific
processing step applies optional Gaussian smoothing followed by the local
weighted filter implemented in ``local_weigh_filter.filter3``.

Author: Jose Murteira
Cleaned and modularized by: Luis
"""

import numpy as np
import scipy.ndimage
from typing import Any
import os

from coconut_tools.tools.logger_config import setup_logger
from coconut_tools.magnetogram.local_weigh_filter import filter3
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
from coconut_tools.magnetogram.sph_filtering import (
    _as_bool,
    apply_configured_longitude_rotation,
    correct_net_flux,
    plot_maps,
    read_magnetogram,
    read_interpolated_magnetogram,
    write_bc_file,
)

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


def process_magnetogram_date(
    config: dict[str, Any],
    target_date,
    method_used: str = "Yaroslavsky",
    output_path_fig: str | None = None,
) -> dict[str, Any]:
    """Process one target time with the local weighted filter pipeline.

    The function downloads or reuses a magnetogram, optionally interpolates a
    four-map stencil, computes and logs the effective magnetogram time,
    optionally rotates to Stonyhurst, optionally balances net flux, applies the
    local weighted filter, writes the boundary file, and optionally saves a
    diagnostic figure.

    Args:
        config (dict[str, Any]): Processing configuration. Common keys are
            ``map_type``, ``output_dir``, ``download_dir``, ``r_st``,
            ``amp``, ``adapt_map``, ``write_map``, ``show_map``, ``visu_type``,
            ``alpha``, ``Rn``, ``sig``, ``interpolation_order``,
            ``interpolation``, ``rotate_to_stonyhurst``, ``flux_correct``,
            ``flux_correction_method``, ``drms_email`` or ``jsoc_email``, and
            ``write_gaussian_prepass``.
        target_date: Requested processing time.
        method_used (str): Method label used in output filenames.
        output_path_fig (str | None): Explicit diagnostic figure path. If
            omitted, the figure name is built from the effective time.

    Returns:
        dict[str, Any]: Processing metadata, including target ``date``,
        ``effective_date``, output paths, selected local file or interpolation
        stencil, optional ``Br_linear``, and rotation angle.
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
    alpha = config.get("alpha", 1.0)
    Rn = config.get("Rn", 5.0)
    sig = config.get("sig", 0.0)
    interpolation_order = config.get("interpolation_order", config.get("Interp_order", 2))
    use_interpolation = _as_bool(
        config.get("interpolation", is_gong_temporal_map_type(map_type) or map_type == "ADAPT")
    )
    rotate_to_stonyhurst = _as_bool(config.get("rotate_to_stonyhurst", True))
    flux_correction_method = config.get("flux_correction_method", "surface_mean")
    drms_email = config.get("drms_email", config.get("jsoc_email"))
    resize = _as_bool(config.get("resize", False))

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

    Br_filtered = filter_radial_field_weighted(
        Br,
        Phi[0, :],
        Theta[:, 0],
        alpha,
        Rn,
        sig,
        write_gaussian_prepass=_as_bool(config.get("write_gaussian_prepass", False)),
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
        "rotation_angle": rotation_angle,
    }


def process_config(config: dict[str, Any], method_used: str = "Yaroslavsky") -> list[dict[str, Any]]:
    """Process all target times described by one Yaroslavsky configuration.

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
    label = "yaroslavsky_gong"
    output_dir = os.path.join(base_output_dir, label)
    figure_output_dir = os.path.join(base_output_dir, "images")

    configs = [{
            "date": "2024-07-01T06:17:00",
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
            "map_type": "GONG_mrzqs",
            "adapt_map": 6,
            "output_dir": output_dir,
            "download_dir": output_dir,
            "output_path_fig": os.path.join(figure_output_dir, f"{label}.png"),
            "drms_email": "luis.linan@kuleuven.be",
            "alpha": 1.4,
            "Rn" : 2,
            "sig": 1.5
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
