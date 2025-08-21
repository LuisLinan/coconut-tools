#!/usr/bin/env python3
"""
Plot and list positions of planets + spacecraft (incl. PSP and SOHO) using `solarmach`.

Usage examples (no CLI parsing)
-------------------------------
# as a module
from plot_positions_with_solarmach import run_solarmach, DEFAULT_BODIES
run_solarmach(
    date="2025-08-13 12:00:00",              # UTC string
    bodies=DEFAULT_BODIES,                    # e.g. ["Mercury","Venus","Earth","Mars","Jupiter","PSP","SOHO","Solar Orbiter","STEREO-A","BepiColombo"]
    outfile="fig_solarmach.png",             # PNG path, or None to skip saving
    csv_out="positions.csv",                 # CSV path, or None
    coords="Carrington",                     # or "Stonyhurst"
)

# or run this file directly (uses now-UTC and defaults)
python plot_positions_with_solarmach.py

Notes
-----
* `solarmach` expects times in UTC. Strings like "YYYY-MM-DD HH:MM:SS" are fine.
* Customize the list of bodies by passing a Python list.
* Switch to Stonyhurst with `coords="Stonyhurst"`.
"""

from __future__ import annotations

from datetime import datetime
from typing import List

import pandas as pd
from solarmach import SolarMACH, print_body_list


DEFAULT_BODIES = [
    # Planets (safe subset widely used in docs/examples)
    "Mercury", "Venus", "Earth", "Mars", "Jupiter",
    # Spacecraft
    "PSP", "SOHO", "Solar Orbiter", "STEREO-A", "BepiColombo",
]

DEFAULT_BODIES = [
    # Planets (safe subset widely used in docs/examples)
     "Earth",
    # Spacecraft
    "PSP", "Solar Orbiter",
]


def list_bodies() -> None:
    """Print body keys that SolarMACH supports (selection)."""
    try:
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
    """
    Create a SolarMACH plot and dump positions to CSV.

    Parameters
    ----------
    date : str
        UTC datetime string, e.g. "2025-08-13 12:00:00".
    bodies : list[str] | None
        Bodies/spacecraft to include. If None, uses DEFAULT_BODIES.
    outfile : str | None
        Path to save PNG. If None, no image is saved.
    csv_out : str | None
        Path to save coordinates CSV. If None, no CSV is saved.
    coords : {"Carrington","Stonyhurst"}
        Coordinate system for plotting and table.

    Returns
    -------
    pandas.DataFrame
        Table of coordinates from SolarMACH (also printed to console).
    """
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

    # A compact selection (if present); keep all otherwise
    cols_pref = [
        "body", "lon", "lat", "r", "vsw", "lon_footpoint", "lat_footpoint",
    ]
    cols = [c for c in cols_pref if c in df.columns]
    if cols:
        df = df[cols]

    # Pretty print to console
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
    """Convenience wrapper to call `run_solarmach` without CLI parsing."""
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
        date="2025-08-13 12:00:00",  # UTC string
        bodies=DEFAULT_BODIES,
        # e.g. ["Mercury","Venus","Earth","Mars","Jupiter","PSP","SOHO","Solar Orbiter","STEREO-A","BepiColombo"]
        outfile="fig_solarmach.png",  # PNG path, or None to skip saving
        csv_out="positions.csv",  # CSV path, or None
        coords="Stonyhurst",  # or "Stonyhurst"
    )