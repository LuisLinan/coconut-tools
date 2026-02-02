"""
Visualize Br maps stored in a COCONUT photospheric boundary condition (BC) file.

This module reads a COCONUT BC .dat file that contains rows of:
    x y z Br

It reconstructs longitude and latitude from Cartesian coordinates and displays
Br as a 2D map with:
- X axis: longitude in degrees in [0, 360)
- Y axis: latitude in degrees or sin(latitude), depending on configuration
- Color: Br values (same units as in the BC file)

No command line interface is used. The script is driven by a list of configs
defined in the __main__ block.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from coconut_tools.logger_config import setup_logger


logger = setup_logger(__name__)


@dataclass(frozen=True)
class BrMap:
    """Container for a gridded Br map.

    Attributes:
        lon_deg: 1D array of longitudes in degrees, sorted ascending in [0, 360).
        lat_deg: 1D array of latitudes in degrees, sorted ascending.
        br: 2D array of Br values with shape (n_lat, n_lon).
    """

    lon_deg: np.ndarray
    lat_deg: np.ndarray
    br: np.ndarray


def load_br_map_from_bcfile(
    bc_file: str | Path,
    rounding_decimals: int = 8,
) -> BrMap:
    """Load and grid Br data from a COCONUT BC file.

    The BC file is expected to contain:
        first line: an integer (often "1")
        second line: a comment line (often starts with "!PHOTOSPHERE ...")
        subsequent lines: x y z Br

    Longitude and latitude are reconstructed from (x, y, z) as:
        lon = atan2(y, x) in degrees, mapped to [0, 360)
        lat = 90 - theta, where theta = arccos(z / r)

    A regular grid is inferred from unique (lat, lon) values after rounding.

    Args:
        bc_file: Path to the BC file.
        rounding_decimals: Number of decimals used to infer unique lat and lon
            values. This mitigates floating point noise in coordinate reconstruction.

    Returns:
        BrMap: Gridded map with longitude and latitude axes and a 2D Br array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no numeric data rows are parsed.
    """
    bc_path = Path(bc_file)
    if not bc_path.exists():
        raise FileNotFoundError(f"BC file not found: {bc_path}")

    logger.info("Reading BC file: %s", bc_path)

    with bc_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 3:
        raise ValueError("BC file has too few lines to contain data.")

    rows: List[List[float]] = []
    for line_no, line in enumerate(lines[2:], start=3):
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            logger.debug("Skipping non-numeric line %d: %s", line_no, line.strip())

    if not rows:
        raise ValueError("No numeric data rows were parsed from the BC file.")

    arr = np.asarray(rows, dtype=float)
    x = arr[:, 0]
    y = arr[:, 1]
    z = arr[:, 2]
    br = arr[:, 3]

    r = np.sqrt(x * x + y * y + z * z)
    valid = r > 0.0
    if not np.all(valid):
        logger.warning("Found %d points with r=0, they will be ignored.", int(np.sum(~valid)))
        x, y, z, r, br = x[valid], y[valid], z[valid], r[valid], br[valid]

    lon = np.degrees(np.arctan2(y, x))
    lon = np.mod(lon, 360.0)

    cos_theta = np.clip(z / r, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    lat = 90.0 - np.degrees(theta)

    lon_r = np.round(lon, rounding_decimals)
    lat_r = np.round(lat, rounding_decimals)

    lon_unique = np.unique(lon_r)
    lat_unique = np.unique(lat_r)
    lon_unique.sort()
    lat_unique.sort()

    n_lat = lat_unique.size
    n_lon = lon_unique.size
    logger.info("Inferred grid size: n_lat=%d, n_lon=%d", n_lat, n_lon)

    lon_index: Dict[float, int] = {v: i for i, v in enumerate(lon_unique.tolist())}
    lat_index: Dict[float, int] = {v: i for i, v in enumerate(lat_unique.tolist())}

    br_grid = np.full((n_lat, n_lon), np.nan, dtype=float)
    for lo, la, val in zip(lon_r, lat_r, br, strict=False):
        br_grid[lat_index[la], lon_index[lo]] = val

    filled = int(np.isfinite(br_grid).sum())
    total = int(br_grid.size)
    if filled == 0:
        raise ValueError("Failed to grid the BC data: all grid cells are NaN.")
    if filled < total:
        logger.warning(
            "Grid has missing values: filled=%d of %d cells. Plot will show gaps.",
            filled,
            total,
        )

    return BrMap(lon_deg=lon_unique, lat_deg=lat_unique, br=br_grid)


def plot_br_map(
    br_map: BrMap,
    visu_type: str = "lat",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "Br map from COCONUT BC file",
    colorbar_label: str = "Br",
    output_path_fig: Optional[str | Path] = None,
    show_map: bool = True,
) -> None:
    """Plot a gridded Br map.

    Args:
        br_map: BrMap container with longitude, latitude, and Br grid.
        visu_type: "lat" for latitude (deg) or "sinlat" for sin(latitude).
        vmin: Minimum value for the color scale. If None, uses nanmin(Br).
        vmax: Maximum value for the color scale. If None, uses nanmax(Br).
        title: Figure title.
        colorbar_label: Colorbar label.
        output_path_fig: If provided, saves the figure to this path.
        show_map: If True, displays the figure interactively.

    Raises:
        ValueError: If visu_type is not "lat" or "sinlat".
    """
    if visu_type not in {"lat", "sinlat"}:
        raise ValueError("visu_type must be either 'lat' or 'sinlat'.")

    lon = br_map.lon_deg
    lat = br_map.lat_deg
    br = br_map.br

    y = lat if visu_type == "lat" else np.sin(np.deg2rad(lat))
    y_label = "Latitude (deg)" if visu_type == "lat" else "sin(Latitude)"

    vmin_use = float(np.nanmin(br)) if vmin is None else float(vmin)
    vmax_use = float(np.nanmax(br)) if vmax is None else float(vmax)

    Lon, Y = np.meshgrid(lon, y)

    plt.figure(figsize=(10, 5))
    im = plt.pcolormesh(Lon, Y, br, shading="auto", vmin=vmin_use, vmax=vmax_use)

    plt.xlabel("Longitude (deg)")
    plt.ylabel(y_label)
    plt.title(title)

    cb = plt.colorbar(im)
    cb.set_label(colorbar_label)

    plt.xlim(0.0, 360.0)
    plt.xticks(np.arange(0.0, 361.0, 60.0))
    plt.grid(True)

    if output_path_fig is not None:
        out_path = Path(output_path_fig)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        logger.info("Saved figure to: %s", out_path)

    if show_map:
        plt.show()
    else:
        plt.close()


def visualize_bc_br_from_config(config: Dict[str, object]) -> None:
    """Load a BC file from config and visualize its Br map.

    Expected config keys:
        - bc_file (str): Path to the BC file to read.
        - visu_type (str, optional): "lat" or "sinlat" (default: "lat")
        - vmin (float, optional): Color scale minimum (default: Br min)
        - vmax (float, optional): Color scale maximum (default: Br max)
        - show_map (bool, optional): Whether to display the plot (default: True)
        - output_path_fig (str, optional): If set, save the figure to this path
        - title (str, optional): Figure title (default: derived from filename)

    Args:
        config: Configuration dictionary.

    Raises:
        KeyError: If required keys are missing.
    """
    bc_file = Path(str(config["bc_file"]))

    visu_type = str(config.get("visu_type", "lat"))
    vmin = config.get("vmin", None)
    vmax = config.get("vmax", None)
    show_map = bool(config.get("show_map", True))
    output_path_fig = config.get("output_path_fig", None)

    title = str(config.get("title", f"Br map from {bc_file.name}"))

    br_map = load_br_map_from_bcfile(bc_file)

    plot_br_map(
        br_map=br_map,
        visu_type=visu_type,
        vmin=None if vmin is None else float(vmin),
        vmax=None if vmax is None else float(vmax),
        title=title,
        colorbar_label="Br",
        output_path_fig=None if output_path_fig is None else str(output_path_fig),
        show_map=show_map,
    )


if __name__ == "__main__":
    configs = [
        {
            "bc_file": "C:/Users/luisl/Documents/Travail/coconut-tools/src/coconut_tools/testmap_hmi_polfil_lmax20_cr2219_sph.dat",
            "visu_type": "lat",
            "vmin": None,
            "vmax": None,
            "show_map": True,
            "output_path_fig": "C:/Users/luisl/Documents/Travail/coconut-tools/src/coconut_tools/test/hmi_20201207.png",
            "title": "Br map from COCONUT BC file (example)",
        }
    ]

    for cfg in configs:
        logger.info("Processing config for bc_file=%s", cfg.get("bc_file"))
        visualize_bc_br_from_config(cfg)
