#!/usr/bin/env python3
"""SolarMACH helper: plot and list positions of planets/spacecraft.

This module provides thin wrappers around `solarmach` to:
- create a publication-ready SolarMACH plot, and
- export the position table to CSV.

Notes
-----
- SolarMACH expects UTC timestamps (e.g. ``"YYYY-MM-DD HH:MM:SS"``).
- Coordinate systems supported here: ``"Carrington"`` and ``"Stonyhurst"``.
- We import `solarmach` lazily inside functions so Sphinx/RTD can import
  this module even if `solarmach` is not installed.

Examples
--------
As a module::

    from coconut_tools.solarmach_plot import run_solarmach, DEFAULT_BODIES

    run_solarmach(
        date="2025-08-13 12:00:00",
        bodies=DEFAULT_BODIES,
        outfile="fig_solarmach.png",
        csv_out="positions.csv",
        coords="Carrington",
    )

From the command line (when executing this file directly)::

    python solarmach_plot.py
"""

from __future__ import annotations

from datetime import datetime
from typing import List

import pandas as pd  # ok to import eagerly

# Keep a single definition (remove duplicates)
DEFAULT_BODIES = [
    "Mercury", "Venus", "Earth", "Mars", "Jupiter",
    "PSP", "SOHO", "Solar Orbiter", "STEREO-A", "BepiColombo",
]


def list_bodies() -> None:
    """Print a subset of body keys supported by SolarMACH."""
    try:
        # Lazy import so RTD can import this module without solarmach installed
        from solarmach import print_body_list
        print("Available body keys (subset):")
        print(print_body_list().index)
    except Exception as e:
        print("Could not retrieve body list:", e)


def run_solarmach(
    date: str,
    bodies: List[str] | None = None,
    outfile: str | None = "solarmach_plot.png",
    csv_out: str | None = "positions.csv",
    coords: str = "Carrington",
    reference_long: float | None = 180,
    reference_lat: float | None = 0,
    plot_spirals: bool = True,
    plot_sun_body_line: bool = True,
    reference_vsw: float = 400.0,
    long_offset: float = 0.0,
):
    """Create a SolarMACH plot and dump positions to CSV.

    Args
    ----
    date:
        UTC datetime string, e.g. ``"2025-08-13 12:00:00"``.
    bodies:
        Bodies/spacecraft to include. If ``None``, uses ``DEFAULT_BODIES``.
    outfile:
        PNG path to save the plot. If ``None``, the figure is not saved.
    csv_out:
        CSV path to save the coordinate table. If ``None``, not saved.
    coords:
        Coordinate system, ``"Carrington"`` or ``"Stonyhurst"``.
    reference_long, reference_lat:
        Reference longitude/latitude passed to SolarMACH (degrees).
    plot_spirals, plot_sun_body_line:
        Toggle Parker spirals and Sun–body line.
    reference_vsw:
        Reference solar-wind speed (km/s) used for spirals.
    long_offset:
        Longitude offset (degrees) added to plotted positions.

    Returns
    -------
    pandas.DataFrame
        The coordinate table returned by SolarMACH (also printed).
    """
    # Lazy import to avoid RTD import errors
    from solarmach import SolarMACH

    if bodies is None:
        bodies = list(DEFAULT_BODIES)

    # Since v0.4+, vsw_list can be empty to auto-fetch from spacecraft when possible
    vsw_list: List[float] = []

    sm = SolarMACH(
        date,
        bodies,
        vsw_list,
        reference_long,
        reference_lat,
        coords,
    )

    sm.plot(
        plot_spirals=plot_spirals,
        plot_sun_body_line=plot_sun_body_line,
        reference_vsw=reference_vsw,
        transparent=False,
        markers="numbers",
        long_offset=long_offset,
        return_plot_object=False,
        outfile=outfile,
    )

    df = sm.coord_table.copy()

    # Prefer a compact selection if present
    cols_pref = ["body", "lon", "lat", "r", "vsw", "lon_footpoint", "lat_footpoint"]
    cols = [c for c in cols_pref if c in df.columns]
    if cols:
        df = df[cols]

    with pd.option_context("display.max_columns", None, "display.width", 140):
        print("\nPositions @", date, f"(coords={coords})")
        print(df.sort_values("body") if "body" in df.columns else df)

    if csv_out:
        df.to_csv(csv_out, index=False)
        print(f"\nSaved positions to: {csv_out}")

    if outfile:
        print(f"Saved plot to: {outfile}")

    return df


def main(
    date: str | None = None,
    bodies: List[str] | None = None,
    coords: str = "Carrington",
    outfile: str | None = "solarmach_plot.png",
    csv_out: str | None = "positions.csv",
    plot_spirals: bool = True,
    plot_sun_body_line: bool = True,
):
    """Convenience wrapper to call :func:`run_solarmach` without CLI parsing."""
    if date is None:
        date = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    return run_solarmach(
        date=date,
        bodies=bodies or DEFAULT_BODIES,
        outfile=outfile,
        csv_out=csv_out,
        coords=coords,
        plot_spirals=plot_spirals,
        plot_sun_body_line=plot_sun_body_line,
    )


if __name__ == "__main__":
    # Run with defaults and current UTC time when executed directly
    main(
        date="2025-08-13 12:00:00",
        bodies=DEFAULT_BODIES,
        outfile="fig_solarmach.png",
        csv_out="positions.csv",
        coords="Carrington",
    )
