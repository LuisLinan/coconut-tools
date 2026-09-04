"""Physical-coordinate diagnostic plots for processed magnetograms."""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from coconut_tools.magnetogram.core.coordinates import theta_cell_edges
from coconut_tools.magnetogram.io.downloads import parse_iso_datetime
from coconut_tools.tools.logger_config import setup_logger

logger = setup_logger(__name__)

PLOT_COLOR_LIMIT_PERCENTILE = 99.0
_PLOT_COLOR_LIMIT_PERCENTILE = PLOT_COLOR_LIMIT_PERCENTILE


def _symmetric_color_limit(
    values,
    percentile=PLOT_COLOR_LIMIT_PERCENTILE,
):
    """Return a robust symmetric colorbar limit for signed magnetogram data."""
    finite_abs = np.abs(np.asarray(values, dtype=float))
    finite_abs = finite_abs[np.isfinite(finite_abs)]
    finite_abs = finite_abs[finite_abs > 0.0]
    if finite_abs.size == 0:
        return 1.0

    limit = float(np.nanpercentile(finite_abs, percentile))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = float(np.nanmax(finite_abs))
    return limit if limit > 0.0 else 1.0


def _colorbar_extend(values, limit):
    """Return the Matplotlib colorbar extension needed for clipped values."""
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return "neither"

    extend_min = np.nanmin(finite_values) < -limit
    extend_max = np.nanmax(finite_values) > limit
    if extend_min and extend_max:
        return "both"
    if extend_min:
        return "min"
    if extend_max:
        return "max"
    return "neither"


def _center_edges(values: np.ndarray, lower=None, upper=None) -> np.ndarray:
    """Extrapolate monotonic cell-center coordinates to plotting edges."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Plot coordinates must be a non-empty finite 1D array.")
    if values.size == 1:
        edges = np.array([values[0] - 0.5, values[0] + 0.5])
    else:
        differences = np.diff(values)
        if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
            raise ValueError("Plot coordinates must be strictly monotonic.")
        edges = np.empty(values.size + 1, dtype=float)
        edges[1:-1] = 0.5 * (values[:-1] + values[1:])
        edges[0] = values[0] - 0.5 * differences[0]
        edges[-1] = values[-1] + 0.5 * differences[-1]
    if lower is not None:
        edges = np.maximum(edges, lower)
    if upper is not None:
        edges = np.minimum(edges, upper)
    return edges


def _plot_magnetogram_axis(
    ax,
    values,
    longitude,
    latitude,
    visu_type,
    limit,
):
    """Plot one magnetogram on its requested latitude coordinate."""
    longitude_edges = _center_edges(longitude)
    theta = np.radians(90.0 - np.asarray(latitude, dtype=float))
    latitude_edges = 90.0 - np.degrees(theta_cell_edges(theta))
    if visu_type == "lat":
        artist = ax.pcolormesh(
            longitude_edges,
            latitude_edges,
            values,
            shading="flat",
            cmap="seismic",
            vmin=-limit,
            vmax=limit,
        )
        return artist, "Latitude"

    sinlat = np.sin(np.radians(latitude))
    sine_differences = np.diff(sinlat)
    uniform_sine = sine_differences.size <= 1 or np.allclose(
        sine_differences,
        np.median(sine_differences),
        atol=1.0e-12,
        rtol=1.0e-8,
    )
    sine_edges = np.cos(theta_cell_edges(theta))
    if not uniform_sine:
        artist = ax.pcolormesh(
            longitude_edges,
            sine_edges,
            values,
            shading="flat",
            cmap="seismic",
            vmin=-limit,
            vmax=limit,
        )
        return artist, "Sine Latitude"

    artist = ax.imshow(
        values[::-1],
        aspect="auto",
        origin="lower",
        cmap="seismic",
        extent=[
            longitude_edges[0],
            longitude_edges[-1],
            sine_edges[-1],
            sine_edges[0],
        ],
        vmin=-limit,
        vmax=limit,
    )
    return artist, "Sine Latitude"


def plot_maps(
    Br,
    Br_mode,
    theta,
    phi,
    map_type,
    visu_type,
    output_path="output_map.png",
    date: str | datetime | None = None,
):
    """Save a two-panel diagnostic figure for input and processed Br maps."""
    logger.info("Plotting maps")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    if date is not None:
        date_label = parse_iso_datetime(date).strftime("%Y-%m-%d %H:%M:%S")
        fig.suptitle(f"{map_type} magnetogram - {date_label} UTC", fontsize=14)

    latitude = 90.0 - 180.0 * theta / np.pi
    longitude = 180.0 * phi / np.pi
    vmax1 = _symmetric_color_limit(Br)
    vmax2 = _symmetric_color_limit(Br_mode)

    def log_stats(name, values):
        logger.info(
            "%s min %.6e max %.6e absmax %.6e p99 abs %.6e mean abs %.6e",
            name,
            np.nanmin(values),
            np.nanmax(values),
            np.nanmax(np.abs(values)),
            np.nanpercentile(np.abs(values), 99),
            np.nanmean(np.abs(values)),
        )

    log_stats("original", Br)
    log_stats("processed", Br_mode)

    im1, ylabel = _plot_magnetogram_axis(
        ax1,
        Br,
        longitude,
        latitude,
        visu_type,
        vmax1,
    )
    ax1.set_ylabel(ylabel, fontsize=14)
    ax1.set_title("Original magnetogram", fontsize=16)
    ax1.set_xticks(np.arange(0.0, 360.0, 60.0))
    ax1.tick_params(axis="both", which="major", labelsize=12)
    cbar1 = plt.colorbar(im1, ax=ax1, extend=_colorbar_extend(Br, vmax1))
    cbar1.set_label("Br [G]", fontsize=14)
    cbar1.ax.tick_params(labelsize=12)

    im2, ylabel = _plot_magnetogram_axis(
        ax2,
        Br_mode,
        longitude,
        latitude,
        visu_type,
        vmax2,
    )
    ax2.set_title("Processed input magnetogram", fontsize=16)
    ax2.set_ylabel(ylabel, fontsize=14)
    ax2.set_xlabel("Longitude", fontsize=14)
    ax2.tick_params(axis="both", which="major", labelsize=12)
    cbar2 = plt.colorbar(im2, ax=ax2, extend=_colorbar_extend(Br_mode, vmax2))
    cbar2.set_label("Br [G/2.2]", fontsize=14)
    cbar2.ax.tick_params(labelsize=12)

    if date is not None:
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path)
    plt.close()
