"""Shared orchestration of the magnetogram filter pipelines.

The four entry scripts (``sph_filtering``, ``gaussian_smoothing``,
``NLD_implicit_method``, ``Yaroslavsky_filter``) used to carry their own copy
of the same chain. That chain now lives here once:

1. read the common configuration keys;
2. read the magnetogram (custom file, temporal interpolation, or single map);
3. determine the effective and display dates, and the output names;
4. rotate to Stonyhurst when requested;
5. optional net-flux correction before the filter;
6. **the filter itself** (supplied by the entry script as a ``FilterStep``);
7. optional net-flux correction after the filter;
8. ``/ 2.2 * amp`` scaling (unless the filter applies it itself);
9. write the boundary file, plot, and return the metadata dictionary.

The entry scripts keep their public functions (``process_magnetogram_date``,
``process_config``, filter wrappers) with unchanged signatures and returned
keys, and their ``if __name__ == "__main__":`` blocks remain the place where a
run is configured.
"""

from dataclasses import dataclass
import os
from typing import Any, Callable

import numpy as np

from coconut_tools.magnetogram.core.config import _as_bool
from coconut_tools.magnetogram.io.downloads import (
    build_output_name,
    build_processing_dates,
    generate_output_and_interpolation_map_names,
    generate_output_and_map_names,
    is_gong_temporal_map_type,
    log_shared_output_files,
    magnetogram_display_date,
    magnetogram_effective_date,
    normalize_map_type,
    parse_iso_datetime,
    resolve_boundary_output_name,
    resolve_figure_path,
    resolve_temporal_interpolation,
)
from coconut_tools.magnetogram.io.readers import (
    read_interpolated_magnetogram,
    read_magnetogram,
)
from coconut_tools.magnetogram.io.writers import write_bc_file
from coconut_tools.magnetogram.processing.flux_balance import (
    correct_net_flux,
    log_flux_summary,
    resolve_flux_correction_stage,
)
from coconut_tools.magnetogram.processing.longitude import (
    apply_configured_longitude_rotation,
)
from coconut_tools.magnetogram.visualization.plotting import plot_maps
from coconut_tools.tools.logger_config import setup_logger

logger = setup_logger(__name__)

OUTPUT_SCALE_DIVISOR = 2.2
"""Historical empirical scaling of the written boundary field (``Br / 2.2``)."""


def log_boundary_flux(Br: np.ndarray, Theta: np.ndarray, Phi: np.ndarray) -> None:
    """Log the flux summary of the final boundary map (non-spectral filters)."""
    log_flux_summary("Final boundary Br", Br, Theta[:, 0], Phi[0, :])


@dataclass(frozen=True)
class FilterStep:
    """Filter-specific part of a pipeline.

    Attributes:
        resolve_settings: ``config -> settings``. Called before any file is
            read, so configuration errors are raised early.
        apply: ``(Br, Theta, Phi, settings) -> (filtered, extra_metadata)``.
            ``Br`` is the rotated (and possibly flux-corrected) field;
            ``extra_metadata`` is merged into the returned dictionary.
        configurable_flux_stage: True when ``flux_correction_stage``
            (``"before"``/``"after"``) applies. False means the correction, if
            requested, is always applied before the filter and the key is not
            returned (spherical harmonics).
        scale_output: True when the pipeline applies ``/ 2.2 * amp`` to the
            filtered field. False when ``apply`` already does it.
        log_final: Optional ``(final_field, Theta, Phi) -> None`` diagnostic
            called on the final (scaled) field.
    """

    resolve_settings: Callable[[dict[str, Any]], Any]
    apply: Callable[[np.ndarray, np.ndarray, np.ndarray, Any], tuple[np.ndarray, dict[str, Any]]]
    configurable_flux_stage: bool = True
    scale_output: bool = True
    log_final: Callable[[np.ndarray, np.ndarray, np.ndarray], None] | None = log_boundary_flux


def run_magnetogram_pipeline(
    config: dict[str, Any],
    target_date,
    step: FilterStep,
    method_used: str,
    output_path_fig: str | None = None,
    unique_names: bool = False,
) -> dict[str, Any]:
    """Process one target time with the given filter step.

    Args:
        config: Processing configuration (common keys: ``map_type`` or
            ``custom_magnetogram``, ``output_dir``, ``download_dir``,
            ``r_st``, ``amp``, ``adapt_map``, ``write_map``, ``show_map``,
            ``visu_type``, ``interpolation_order``, ``interpolation``,
            ``resize``, ``rotate_to_stonyhurst``, ``carrington``,
            ``flux_correct``, ``flux_correction_method``,
            ``flux_correction_stage``, ``drms_email`` or ``jsoc_email``; plus
            the filter-specific keys read by ``step.resolve_settings``).
        target_date: Requested processing time.
        step: Filter-specific part of the pipeline.
        method_used: Method label used in output filenames.
        output_path_fig: Figure file or directory, resolved with
            ``resolve_figure_path`` at the effective time.
        unique_names: Append the effective time to the ``.dat`` and figure
            names (set by ``run_config`` for a time series).

    Returns:
        Metadata dictionary: ``date``, ``effective_date``,
        ``magnetogram_date``, ``output_name``, ``local_file``,
        ``figure_path``, ``selection``, ``Br_linear``, the filter's extra
        metadata, ``rotation_angle`` and, when configurable,
        ``flux_correction_stage``.
    """
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
    r_st = config.get("r_st", 1.0)
    amp = config.get("amp", 1)
    adapt_map = config.get("adapt_map", 6)
    write_map = _as_bool(config.get("write_map", True))
    show_map = _as_bool(config.get("show_map", True))
    visu_type = config.get("visu_type", "sinlat")
    interpolation_order = config.get("interpolation_order", config.get("Interp_order", 2))
    requested_interpolation = _as_bool(
        config.get("interpolation", is_gong_temporal_map_type(map_type) or map_type == "ADAPT")
    )
    use_interpolation = requested_interpolation and custom_magnetogram is None
    if custom_magnetogram is not None and requested_interpolation:
        logger.info("Temporal interpolation is disabled for a custom magnetogram.")
    rotate_to_stonyhurst = _as_bool(config.get("rotate_to_stonyhurst", True))
    carrington = _as_bool(config.get("carrington", False))
    physical_longitude = (carrington or rotate_to_stonyhurst) and map_type.lower() != "wso"
    flux_correction_method = config.get("flux_correction_method", "surface_mean")
    flux_correct = _as_bool(config.get("flux_correct", False))
    flux_correction_stage = (
        resolve_flux_correction_stage(config.get("flux_correction_stage"))
        if step.configurable_flux_stage
        else None
    )
    drms_email = config.get("drms_email", config.get("jsoc_email"))
    resize = _as_bool(config.get("resize", False))
    settings = step.resolve_settings(config)

    interpolated = resolve_temporal_interpolation(map_type, use_interpolation)

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
            carrington=physical_longitude,
        )
        Br_linear = None
        selection = None
    elif interpolated:
        output_name, local_files, selection = generate_output_and_interpolation_map_names(
            target_date,
            map_type,
            output_dir,
            method_used=method_used,
            download_dir=download_dir,
            drms_email=drms_email,
        )
        Br, Theta, Phi, Br_linear = read_interpolated_magnetogram(
            local_files,
            map_type,
            selection,
            adapt_map=adapt_map,
            interpolation_order=interpolation_order,
            resize=resize,
            carrington=physical_longitude,
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
            carrington=physical_longitude,
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
    output_name = resolve_boundary_output_name(
        output_name, effective_date, use_unique_name=unique_names
    )
    figure_date = magnetogram_display_date(
        source_file,
        map_type,
        target_date,
        interpolated=interpolated,
    )
    Br, Br_linear, Phi, rotation_angle = apply_configured_longitude_rotation(
        Br,
        Br_linear,
        local_file,
        map_type,
        target_date,
        use_interpolation,
        rotate_to_stonyhurst,
        effective_date=effective_date,
        resize=resize,
        Phi=Phi,
    )

    correct_before = flux_correct and (
        not step.configurable_flux_stage or flux_correction_stage == "before"
    )
    if correct_before:
        Br = correct_net_flux(
            Br,
            Theta[:, 0],
            Phi[0, :],
            method=flux_correction_method,
        )

    Br_filtered, extra_metadata = step.apply(Br, Theta, Phi, settings)

    if flux_correct and step.configurable_flux_stage and flux_correction_stage == "after":
        # Correct the filtered field so the written boundary map has zero net
        # flux (the filter may reintroduce some). Independent of the /2.2*amp
        # scaling below (mean removal and polarity balancing commute with a
        # positive constant factor).
        Br_filtered = correct_net_flux(
            Br_filtered,
            Theta[:, 0],
            Phi[0, :],
            method=flux_correction_method,
        )

    if step.scale_output:
        Br_filtered = Br_filtered / OUTPUT_SCALE_DIVISOR
        Br_filtered *= amp
    if step.log_final is not None:
        step.log_final(Br_filtered, Theta, Phi)

    if write_map:
        write_bc_file(output_name, Br_filtered, Theta[:, 0], Phi[0, :], r_st)

    if show_map:
        figure_path = resolve_figure_path(
            output_path_fig,
            output_dir,
            map_type,
            effective_date,
            use_unique_name=unique_names,
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

    result = {
        "date": parse_iso_datetime(target_date),
        "effective_date": effective_date,
        "magnetogram_date": figure_date,
        "output_name": output_name,
        "local_file": local_file,
        "figure_path": figure_path,
        "selection": selection,
        "Br_linear": Br_linear,
    }
    result.update(extra_metadata)
    result["rotation_angle"] = rotation_angle
    if step.configurable_flux_stage:
        result["flux_correction_stage"] = flux_correction_stage
    return result


def run_config(
    config: dict[str, Any],
    process_date: Callable[..., dict[str, Any]],
    method_used: str,
) -> list[dict[str, Any]]:
    """Process every target time described by one configuration.

    With only ``date`` set, one target time is processed and the historical
    output names are kept. With ``cadence_hours`` and ``total_hours``, every
    target is processed independently and the ``.dat`` and figure names
    receive the effective magnetogram time, so the outputs of a time series
    are not overwritten. Targets resolving to the same magnetogram share one
    file (identical content); this is logged.

    Args:
        config: Processing configuration.
        process_date: The entry script's ``process_magnetogram_date``.
        method_used: Method label used in output filenames.

    Returns:
        Per-date processing results.
    """
    target_dates = build_processing_dates(
        config["date"],
        cadence_hours=config.get("cadence_hours", config.get("candence")),
        total_hours=config.get("total_hours"),
    )
    unique_names = len(target_dates) > 1
    results = [
        process_date(
            config,
            target_date,
            method_used=method_used,
            output_path_fig=config.get("output_path_fig"),
            unique_names=unique_names,
        )
        for target_date in target_dates
    ]
    log_shared_output_files(results)
    return results
