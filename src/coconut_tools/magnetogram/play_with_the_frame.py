"""Synoptic magnetogram utilities (header/filename parsing + aligned plotting).

This module provides small, focused helpers to:

- Read longitude/latitude world coordinates from FITS metadata, including the
  WCS inversion required by standards-compliant CEA latitude axes.
- Detect unit quirks in synoptic products (e.g., *micro‑degrees* in some GONG
  files) and convert to degrees when appropriate.
- Infer the starting Carrington longitude from either header keywords or from
  GONG‑style filenames, then align a synoptic map so the left edge lands at
  0° Carrington.
- Produce a diagnostic plot that clearly reports which longitude was used and
  why (header vs filename).

The public entry point is :func:`plot_synoptic_aligned`, which returns the
``(fig, ax, info)`` triple for further customization.

Notes
-----
- Linear FITS axes follow the world‑coordinate relation::

    world = CRVALn + (i - CRPIXn) * CDELTn,  with i starting at 1

  Angular CEA latitude axes are inverted with Astropy/WCSLIB instead.

- Longitude wrapping uses the ``[0, 360)`` convention.
- Latitude is returned in degrees. If the header encodes sine‑latitude (CSLT),
  it is converted to degrees.

Examples
--------
Basic usage:

>>> from coconut_tools.tools.solarmach_plot import plot_synoptic_aligned
>>> fig, ax, info = plot_synoptic_aligned("/path/to/file.fits", vmin=-100, vmax=100)
>>> info["lon0_used"]
123.0

"""

import csv
import re
from pathlib import Path

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

from coconut_tools.magnetogram.coordinates import (
    fits_latitude_axis as _decode_fits_latitude_axis,
)


GONG_FILE_IDS = {"mrzqs", "mrbqs", "mrbqj", "mrmqs", "mrnqs"}


def gong_file_id_from_name(fname):
    """Return the GONG file id encoded in a filename, if supported."""
    name = Path(str(fname)).name.lower()
    prefix = name[:5]
    return prefix if prefix in GONG_FILE_IDS else None


def is_carrington_longitude(header, fname=""):
    """Heuristically decide whether the x axis is Carrington longitude.

    Args:
        header (collections.abc.Mapping): FITS header.
        fname (str, optional): Filename used as a weak hint.

    Returns:
        tuple[bool, dict]: ``(is_carrington, info)`` where ``info`` contains a
        ``reason`` string plus the relevant header fields.
    """
    ctype1 = str(header.get("CTYPE1", "")).strip().upper()
    ctype2 = str(header.get("CTYPE2", "")).strip().upper()
    cunit1 = str(header.get("CUNIT1", "")).strip().lower()
    name = str(fname).upper()

    # Strong indicators used by many Carrington or synoptic map products
    carr_keys = [
        "CAR_ROT", "CARROT", "CARR_ROT", "CARR_ROTATION",
        "CRLN_OBS", "L0", "CRLN", "CARRINGTON"
    ]
    present_keys = [k for k in carr_keys if k in header]

    # 1) WCS axis type explicitly says Carrington longitude
    # Common: CRLN-CEA (Carrington lon, cylindrical equal area)
    if "CRLN" in ctype1:
        return True, {
            "reason": "CTYPE1 contains CRLN (Carrington longitude axis).",
            "CTYPE1": ctype1, "CTYPE2": ctype2, "present_keys": present_keys, "CUNIT1": cunit1
        }

    # 2) Header contains Carrington rotation number or explicit Carrington metadata
    # CAR_ROT / CARROT is a very strong hint it is Carrington mapped data
    if any(k in header for k in ("CAR_ROT", "CARROT", "CARR_ROT", "CARR_ROTATION")):
        return True, {
            "reason": "Header contains Carrington rotation metadata (CAR_ROT/CARROT...).",
            "CTYPE1": ctype1, "CTYPE2": ctype2, "present_keys": present_keys, "CUNIT1": cunit1
        }

    # 3) Some products label heliographic longitude as HGLN.
    # That can be Stonyhurst or Carrington, so this alone is NOT enough.
    # But if HGLN is present AND there is CRLN_OBS or L0, it usually means Carrington context.
    if "HGLN" in ctype1 and any(k in header for k in ("CRLN_OBS", "L0")):
        return True, {
            "reason": "CTYPE1 is HGLN and header has CRLN_OBS or L0, likely Carrington context.",
            "CTYPE1": ctype1, "CTYPE2": ctype2, "present_keys": present_keys, "CUNIT1": cunit1
        }

    # 4) Product-specific filename hints for known synoptic/Carrington maps.
    if gong_file_id_from_name(fname) is not None and re.search(r"c\d{4}", name, flags=re.IGNORECASE):
        return True, {
            "reason": "Filename matches supported GONG synoptic/CAR product.",
            "CTYPE1": ctype1, "CTYPE2": ctype2, "present_keys": present_keys, "CUNIT1": cunit1
        }

    if "ADAPT" in name:
        return True, {
            "reason": "Filename suggests ADAPT Carrington synoptic product.",
            "CTYPE1": ctype1, "CTYPE2": ctype2, "present_keys": present_keys, "CUNIT1": cunit1
        }

    if "HMI.SYNOPTIC" in name or "MRDAILYSYNFRAME" in name:
        return True, {
            "reason": "Filename suggests HMI synoptic Carrington product.",
            "CTYPE1": ctype1, "CTYPE2": ctype2, "present_keys": present_keys, "CUNIT1": cunit1
        }

    # 5) Filename hint (weak)
    # Some synoptic Carrington products include 'carr' / 'synoptic' in filename.
    if any(tag in name for tag in ("CARR", "SYNOPTIC", "CRLN")):
        return True, {
            "reason": "Filename suggests Carrington/synoptic product (weak heuristic).",
            "CTYPE1": ctype1, "CTYPE2": ctype2, "present_keys": present_keys, "CUNIT1": cunit1
        }

    # Otherwise assume it is not a Carrington longitude map
    return False, {
        "reason": "No strong Carrington indicators found (likely image plane or non Carrington heliographic).",
        "CTYPE1": ctype1, "CTYPE2": ctype2, "present_keys": present_keys, "CUNIT1": cunit1
    }


def fits_latitude_axis(
    header,
    data=None,
    *,
    output="deg",          # "deg" ou "sin"
    one_based_fits=True,
    force_degrees=False,
    force_sine=False,
):
    """Build a physical latitude axis using the shared FITS decoder.

    The result is ordered from south to north.  See
    :func:`coconut_tools.magnetogram.coordinates.fits_latitude_axis` for the
    header-priority and CEA-inversion rules.
    """
    return _decode_fits_latitude_axis(
        header,
        data,
        output=output,
        one_based_fits=one_based_fits,
        force_degrees=force_degrees,
        force_sine=force_sine,
    )



def detect_instrument(h, fname=""):
    """Detect the likely instrument/vendor from header or filename.

    Args:
        h (collections.abc.Mapping): FITS header.
        fname (str, optional): Filename used as a fallback hint.

    Returns:
        str: ``"HMI"``, ``"GONG"``, or ``"UNKNOWN"``.
    """
    tel = str(h.get("TELESCOP", "")).upper()
    inst = str(h.get("INSTRUME", "")).upper()
    name = fname.upper()
    if "HMI" in tel or "HMI" in inst or "HMI" in name:
        return "HMI"
    if "GONG" in tel or name.startswith("MR") or "GONG" in name:
        return "GONG"
    return "UNKNOWN"


# -------------------------------------------------------------
# Start longitude discovery
# -------------------------------------------------------------

def start_lon_from_filename(fname):
    """Extract starting Carrington longitude from a GONG-style filename.

    The function parses filenames like
    ``mrzqs170404t1814c2189_268.fits.gz`` and returns the trailing ``268`` value.

    Args:
        fname (str): Input filename.

    Returns:
        int | None: Parsed degree value if found, otherwise ``None``.
    """
    name = Path(str(fname)).name
    match = re.search(r"_([0-9]{1,3})(?:\.fits|\.fts|\.fit)", name, flags=re.IGNORECASE)
    if match is None:
        return None
    try:
        # e.g., mrzqs170404t1814c2189_268.fits.gz → 268
        return int(match.group(1))
    except Exception:
        return None

def fits_longitude_deg(header, nx, *, one_based_fits=True):
    """Compute the x-axis longitudes in degrees from FITS WCS-like keywords.

    Args:
        header (collections.abc.Mapping): FITS header (e.g. ``hdul[0].header``).
        nx (int): Number of pixels along x (``data.shape[-1]``).
        one_based_fits (bool, optional): FITS WCS convention uses 1-based pixel
            coordinates for CRPIX. Keep True unless CRPIX1 is 0-based.

    Returns:
        tuple[np.ndarray, dict]: ``(x_deg, meta)`` where ``x_deg`` has shape
        ``(nx,)``. ``meta`` includes ``crpix1``, ``cdelt1_deg``, ``crval1_deg``,
        ``cunit1``, and ``detected_scale``.

    Raises:
        KeyError: If required WCS keywords are missing from the header.
    """
    # Required WCS-ish keywords with safe defaults
    crpix1 = float(header.get("CRPIX1", 1.0))
    crval1 = float(header.get("CRVAL1", 0.0))

    # CDELT1 is standard, but some files use CD1_1
    if "CDELT1" in header and header["CDELT1"] not in (None, ""):
        cdelt1 = float(header["CDELT1"])
    elif "CD1_1" in header and header["CD1_1"] not in (None, ""):
        cdelt1 = float(header["CD1_1"])
    else:
        raise KeyError("Missing CDELT1 (or CD1_1) in FITS header.")

    cunit1 = str(header.get("CUNIT1", "")).strip().lower()

    # Convert metadata to degrees if needed
    detected_scale = "deg"
    if cunit1.startswith("deg"):
        cdelt1_deg = cdelt1
        crval1_deg = crval1
        detected_scale = "deg_from_cunit1"
    else:
        # Heuristic: typical deg/px ~ 0.5..2, so if step is huge, it is likely micro-deg
        if abs(cdelt1) > 1000:
            cdelt1_deg = cdelt1 / 1e6
            crval1_deg = crval1 / 1e6
            detected_scale = "microdeg_from_cdelt"
        else:
            cdelt1_deg = cdelt1
            crval1_deg = crval1
            detected_scale = "unknown_assumed_deg"

    # Pixel coordinate vector
    # FITS: world = (pix - CRPIX) * CDELT + CRVAL, with pix typically 1..nx
    if one_based_fits:
        pix = np.arange(1, nx + 1, dtype=float)
    else:
        pix = np.arange(nx, dtype=float)

    x_raw = (pix - crpix1) * cdelt1_deg + crval1_deg

    # Final safety heuristic on the resulting coordinate magnitude
    # If not explicitly degrees and values are huge, it is likely micro-deg encoded
    if not cunit1.startswith("deg"):
        if np.nanmax(np.abs(x_raw)) > 1e4:
            x_raw = x_raw / 1e6
            detected_scale = detected_scale + "+microdeg_from_range"

    meta = dict(
        crpix1=crpix1,
        cdelt1_deg=cdelt1_deg,
        crval1_deg=crval1_deg,
        cunit1=cunit1,
        detected_scale=detected_scale,
    )
    return x_raw.astype(float), meta



def fits_longitude_deg(header, *, one_based_fits=True):
    """Compute the x-axis longitude vector in degrees from a FITS header.

    Args:
        header (collections.abc.Mapping): FITS header (``hdul[0].header``).
        one_based_fits (bool, optional): FITS convention uses 1-based pixel
            coordinates for CRPIX.

    Returns:
        tuple[np.ndarray, dict]: ``(x_deg, meta)`` where ``x_deg`` has shape
        ``(NAXIS1,)`` and ``meta`` contains diagnostics on units and scaling.

    Raises:
        KeyError: If required WCS keywords are missing from the header.
    """
    # Number of pixels along x
    if "NAXIS1" not in header:
        raise KeyError("NAXIS1 missing from FITS header.")
    nx = int(header["NAXIS1"])

    # WCS-like keywords
    crpix1 = float(header.get("CRPIX1", 1.0))
    crval1 = float(header.get("CRVAL1", 0.0))

    if "CDELT1" in header and header["CDELT1"] not in (None, ""):
        cdelt1 = float(header["CDELT1"])
    elif "CD1_1" in header and header["CD1_1"] not in (None, ""):
        cdelt1 = float(header["CD1_1"])
    else:
        raise KeyError("Missing CDELT1 (or CD1_1) in FITS header.")

    cunit1 = str(header.get("CUNIT1", "")).strip().lower()

    # --- unit handling ---
    detected_scale = "deg"

    # If unit explicitly says degrees, trust it
    if cunit1.startswith("deg"):
        cdelt1_deg = cdelt1
        crval1_deg = crval1
        detected_scale = "deg_from_cunit1"
    else:
        # Heuristic: very large step means micro-degrees
        if abs(cdelt1) > 1000:
            cdelt1_deg = cdelt1 / 1e6
            crval1_deg = crval1 / 1e6
            detected_scale = "microdeg_from_cdelt"
        else:
            cdelt1_deg = cdelt1
            crval1_deg = crval1
            detected_scale = "unknown_assumed_deg"

    # Pixel coordinates
    if one_based_fits:
        pix = np.arange(1, nx + 1, dtype=float)
    else:
        pix = np.arange(nx, dtype=float)

    x_raw = (pix - crpix1) * cdelt1_deg + crval1_deg

    # Final safety check on magnitude
    if not cunit1.startswith("deg"):
        if np.nanmax(np.abs(x_raw)) > 1e4:
            x_raw = x_raw / 1e6
            detected_scale += "+microdeg_from_range"

    meta = dict(
        nx=nx,
        crpix1=crpix1,
        cdelt1_deg=cdelt1_deg,
        crval1_deg=crval1_deg,
        cunit1=cunit1,
        detected_scale=detected_scale,
    )

    return x_raw.astype(float), meta

import numpy as np

def order_carrington_0_360(lon_native, data, *, tol=1e-6):
    """Reorder a Carrington map so longitude runs from 0 to 360 degrees.

    If the native longitude axis is decreasing (common for HMI synoptic maps),
    apply the physical transform phi = (360 - lon) mod 360.

    Args:
        lon_native (np.ndarray): Native Carrington longitudes, shape (nx,).
        data (np.ndarray | np.ma.MaskedArray): Magnetogram data, shape (ny, nx).
        tol (float, optional): Tolerance used to stabilize roll detection.

    Returns:
        tuple[np.ndarray, np.ndarray | np.ma.MaskedArray]:
            (lon_out, data_out) with lon_out in [0, 360).
    """
    lon = np.asarray(lon_native, dtype=float)
    data_out = data

    if lon.ndim != 1:
        raise ValueError("lon_native must be 1D.")
    if data.shape[-1] != lon.size:
        raise ValueError("data and lon_native size mismatch.")

    dlon_med = float(np.nanmedian(np.diff(lon)))

    # Physical longitude for spatial BCs
    if dlon_med < 0:
        lon_phys = np.mod(360.0 - lon, 360.0)
    else:
        lon_phys = np.mod(lon, 360.0)

    if tol is not None:
        lon_phys = np.round(lon_phys / tol) * tol

    # Roll so that we start near 0°
    idx0 = int(np.nanargmin(lon_phys))
    lon_out = np.roll(lon_phys, -idx0)
    data_out = np.roll(data_out, -idx0, axis=-1)

    # Final safety: enforce increasing longitude to the right
    if float(np.nanmedian(np.diff(lon_out))) < 0:
        lon_out = lon_out[::-1]
        data_out = data_out[..., ::-1]

    return lon_out, data_out

# -------------------------------------------------------------
# Diagnostics + plotting
# -------------------------------------------------------------

def plot_synoptic_imshow(
    data,
    lon,
    lat,
    header,
    *,
    inst="UNKNOWN",
    lon_meta=None,          # dict from fits_longitude_deg
    lat_mode="unknown",     # string from your lat decoder
    lon0_file=None,         # from start_lon_from_filename for GONG
    lon0_used=None,         # optional, if you apply a specific choice
    vmin=None,
    vmax=None,
    cmap="RdBu_r",
):
    """Plot a synoptic magnetogram with ``imshow`` using lon/lat vectors.

    Assumes ``lon`` and ``lat`` are 1D vectors matching data columns/rows, the
    data are ordered so that ``lon`` is increasing in ``[0, 360)``, and ``lat``
    is increasing (south to north).

    Args:
        data (np.ndarray | np.ma.MaskedArray): Magnetogram data, shape
            ``(ny, nx)``.
        lon (np.ndarray): Longitudes in degrees, shape ``(nx,)``.
        lat (np.ndarray): Latitudes in degrees, shape ``(ny,)``.
        header (collections.abc.Mapping): FITS header.
        inst (str, optional): Instrument label.
        lon_meta (dict | None, optional): Metadata from ``fits_longitude_deg``.
        lat_mode (str, optional): Latitude decoding mode string.
        lon0_file (float | None, optional): Filename-encoded starting longitude.
        lon0_used (float | None, optional): Longitude used for alignment.
        vmin (float | None, optional): ``imshow`` vmin.
        vmax (float | None, optional): ``imshow`` vmax.
        cmap (str, optional): Matplotlib colormap name.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, dict]:
        ``(fig, ax, info)`` with diagnostic information.
    """

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    ny, nx = data.shape[-2], data.shape[-1]
    if lon.size != nx:
        raise ValueError(f"lon size {lon.size} does not match data nx {nx}.")
    if lat.size != ny:
        raise ValueError(f"lat size {lat.size} does not match data ny {ny}.")

    # Ensure monotonic increasing for imshow extent + origin="lower"
    # If not, flip consistently here rather than relying on caller.
    flipped_x = False
    flipped_y = False

    if lon[0] > lon[-1]:
        flipped_x = True
        lon = lon[::-1]
        data = data[..., ::-1]

    if lat[0] > lat[-1]:
        flipped_y = True
        lat = lat[::-1]
        data = data[..., ::-1, :]

    # Extent: imshow assumes pixel centers span the extent.
    # Better: build edges so that the axis ticks correspond to centers.
    def _edges_from_centers(x):
        dx = np.median(np.diff(x))
        edges = np.empty(x.size + 1, dtype=float)
        edges[1:-1] = 0.5 * (x[:-1] + x[1:])
        edges[0] = x[0] - 0.5 * dx
        edges[-1] = x[-1] + 0.5 * dx
        return edges

    lon_edges = _edges_from_centers(lon)
    lat_edges = _edges_from_centers(lat)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(
        data,
        origin="lower",
        extent=[lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]],
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(f"Br [{header.get('BUNIT', 'G')}]")

    # Labels
    ax.set_xlabel("Carrington longitude [deg]")
    ax.set_ylabel("Latitude [deg]")

    # X limits in 0..360, with a tiny pad to avoid cutting off the last column
    ax.set_xlim(0, 360)

    # Diagnostics
    cr = header.get("CAR_ROT", header.get("CARROT", ""))

    ctype1 = str(header.get("CTYPE1", "")).upper()
    cunit1 = str(header.get("CUNIT1", "")).lower()
    ctype2 = str(header.get("CTYPE2", "")).upper()
    cunit2 = str(header.get("CUNIT2", "")).lower()

    cdelt1_deg = None
    lon0_header = float(lon[0]) if lon.size else np.nan

    if lon_meta and "cdelt1_deg" in lon_meta:
        cdelt1_deg = float(lon_meta["cdelt1_deg"])

    # If lon0_used is not provided, interpret it as the first column after ordering
    if lon0_used is None:
        lon0_used = lon0_header

    diag = [
        f"Inst={inst}",
        f"CTYPE1={ctype1 or '(none)'}",
        f"CUNIT1={cunit1 or '(none)'}",
    ]
    if cdelt1_deg is not None:
        diag.append(f"Δx={cdelt1_deg:.6f}°/px")
    diag.append(f"lon0(used)={lon0_used:.3f}°")
    diag.append(f"CTYPE2={ctype2 or '(none)'}")
    diag.append(f"CUNIT2={cunit2 or '(none)'}")
    diag.append(f"lat axis: {lat_mode}")

    if lon0_file is not None:
        d = ((lon0_used - lon0_file + 540) % 360) - 180
        diag.append(f"lon0(file)={float(lon0_file):.3f}°  Δ={d:.3f}°")

    if flipped_x or flipped_y:
        diag.append(f"flip(x)={flipped_x} flip(y)={flipped_y}")

    title = f"Synoptic Br"
    if cr != "":
        title += f" — CR {cr}"
    ax.set_title(title + "\n" + "  |  ".join(diag))

    info = {
        "instrument": inst,
        "shape": (ny, nx),
        "ctype1": ctype1,
        "cunit1": cunit1,
        "ctype2": ctype2,
        "cunit2": cunit2,
        "cdelt1_deg": cdelt1_deg,
        "lon0_header_after_order": lon0_header,
        "lon0_filename": lon0_file,
        "lon0_used": lon0_used,
        "flipped_x": flipped_x,
        "flipped_y": flipped_y,
        "lon_minmax": (float(np.nanmin(lon)), float(np.nanmax(lon))),
        "lat_minmax": (float(np.nanmin(lat)), float(np.nanmax(lat))),
    }

    return fig, ax, info

def read_synoptic_fits(fits_path):
    """Read a synoptic FITS file and return the first valid 2D image HDU.

    Args:
        fits_path (str | pathlib.Path): Path to the FITS file.

    Returns:
        tuple[collections.abc.Mapping, np.ndarray, int]: ``(header, data, hdu_index)``
        where ``data`` is a 2D float array.

    Raises:
        RuntimeError: If no valid 2D image HDU is found.
    """
    with fits.open(fits_path) as hdul:
        for i, hdu in enumerate(hdul):
            data = hdu.data
            if data is None:
                continue

            data = np.squeeze(data)
            if data.ndim != 2:
                continue

            header = hdu.header
            if "NAXIS1" not in header or "NAXIS2" not in header:
                continue

            return header, data.astype(float), i

    raise RuntimeError("No valid 2D image HDU found in FITS file.")

def plot_synoptic_aligned(
    fits_path,
    prefer_filename_for_gong=True,
    vmin=None,
    vmax=None,
    cmap="RdBu_r",
    force_degrees=False,
    force_sine=False,
):
    """Plot a synoptic map aligned so the left edge is 0° Carrington.

    Workflow:
        1) Read data and header from ``fits_path``.
        2) Build the native longitude axis and infer a start longitude from the
           header and, for GONG files, the filename.
        3) Choose which start longitude to use (header by default; filename for
           GONG if ``prefer_filename_for_gong=True``).
        4) Shift/wrap columns so the left edge is at 0°, and build a latitude
           axis in degrees.
        5) Render a diagnostic plot and return ``(fig, ax, info)``.

    Args:
        fits_path (str | pathlib.Path): Path to the synoptic FITS file.
        prefer_filename_for_gong (bool, optional): If True, prefer the
            filename-encoded longitude for GONG products while still reporting
            both header and filename values.
        vmin (float | None, optional): Color scaling lower bound.
        vmax (float | None, optional): Color scaling upper bound.
        cmap (str, optional): Matplotlib colormap name.
        force_degrees (bool, optional): Force latitude axis to be degrees.
        force_sine (bool, optional): Force latitude axis to be sine-latitude.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, dict]:
        ``(fig, ax, info)`` where ``info`` includes diagnostic keys such as
        ``instrument``, ``cdelt1_deg``, ``lon0_header``, ``lon0_filename``, and
        ``lon0_used``.

    Raises:
        RuntimeError: If the map is not in Carrington longitude.

    Notes:
        - The x-extent is set to ``[0, 360]`` after reordering by longitude.
        - Missing/invalid data are masked with ``np.ma.masked_invalid``.
    """
    h, data, hdu_index = read_synoptic_fits(fits_path)


    data = np.ma.masked_invalid(data)
    inst = detect_instrument(h, fname=fits_path)

    # Longitude native depuis le header
    lon_native, lon_meta = fits_longitude_deg(h)

    # Carrington ou non
    is_carrington, proof = is_carrington_longitude(h, fname=fits_path)
    if not is_carrington:
        raise RuntimeError("Map is not in Carrington longitude")

    # Remise en ordre 0–360
    lon_sorted, data_sorted = order_carrington_0_360(lon_native, data)

    # GONG: origine depuis le filename (diagnostic / fallback)
    lon0_file = None
    if inst == "GONG":
        lon0_file = start_lon_from_filename(fits_path)
        if lon0_file is not None:
            print(
                f"GONG start longitude: filename={lon0_file} deg; "
                f"header-derived={lon_native[0] % 360:.3f} deg"
            )

    # Latitude physique en degrés, axe croissant
    lat_deg, data_sorted2, meta_lat = fits_latitude_axis(
        h,
        data_sorted,
        output="deg",   # si tu gardes cette API, sinon tu enlèves output
        force_degrees=force_degrees,
        force_sine=force_sine
    )

    # Plot
    fig, ax, info = plot_synoptic_imshow(
        data_sorted2,
        lon_sorted,
        lat_deg,
        h,
        inst=inst,
        lon_meta=lon_meta,
        lat_mode=meta_lat["detected_mode"],
        lon0_file=lon0_file,
        lon0_used=lon_sorted[0],
        vmin=vmin,
        vmax=vmax,
        cmap="RdBu_r",
    )
        
    return fig, ax, info


def read_synoptic_header(fits_path):
    """Read the first FITS header with longitude/latitude image axes."""
    with fits.open(fits_path) as hdul:
        for i, hdu in enumerate(hdul):
            header = hdu.header
            if "NAXIS1" in header and "NAXIS2" in header:
                return header, i
    raise RuntimeError("No FITS HDU with NAXIS1/NAXIS2 found.")


def diagnose_magnetogram_frame(fits_path, label=None):
    """Return longitude-frame diagnostics for one magnetogram FITS file."""
    fits_path = Path(fits_path)
    header, hdu_index = read_synoptic_header(fits_path)
    inst = detect_instrument(header, fname=str(fits_path))
    is_carrington, proof = is_carrington_longitude(header, fname=str(fits_path))
    lon0_file = start_lon_from_filename(fits_path)

    info = {
        "label": label or fits_path.parent.name,
        "path": str(fits_path),
        "file": fits_path.name,
        "hdu_index": hdu_index,
        "instrument": inst,
        "is_carrington_longitude": is_carrington,
        "carrington_reason": proof.get("reason", ""),
        "ctype1": str(header.get("CTYPE1", "")).strip(),
        "ctype2": str(header.get("CTYPE2", "")).strip(),
        "cunit1": str(header.get("CUNIT1", "")).strip(),
        "cunit2": str(header.get("CUNIT2", "")).strip(),
        "car_rot": header.get("CAR_ROT", header.get("CARROT", "")),
        "crln_obs": header.get("CRLN_OBS", header.get("L0", "")),
        "lon0_filename": lon0_file,
        "longitude_axis_available": False,
        "longitude_axis_error": "",
    }

    try:
        lon_native, lon_meta = fits_longitude_deg(header)
        dlon = np.diff(lon_native)
        info.update(
            {
                "longitude_axis_available": True,
                "lon0_header": float(lon_native[0]),
                "lon_min": float(np.nanmin(lon_native)),
                "lon_max": float(np.nanmax(lon_native)),
                "lon_step_median": float(np.nanmedian(dlon)) if dlon.size else np.nan,
                "lon_is_increasing": bool(float(np.nanmedian(dlon)) > 0.0) if dlon.size else "",
                "lon_detected_scale": lon_meta.get("detected_scale", ""),
                "lon_cdelt1_deg": lon_meta.get("cdelt1_deg", ""),
                "lon_crval1_deg": lon_meta.get("crval1_deg", ""),
            }
        )
    except Exception as exc:
        info["longitude_axis_error"] = str(exc)

    return info


def is_magnetogram_fits(path):
    """Return True for FITS-like magnetogram filenames, including .gz files."""
    name = Path(path).name.lower()
    return name.endswith((".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fts.gz"))


def find_first_magnetogram_file(directory):
    """Find the first FITS-like file under a directory."""
    directory = Path(directory)
    if not directory.exists():
        return None
    files = sorted(path for path in directory.rglob("*") if path.is_file() and is_magnetogram_fits(path))
    return files[0] if files else None


def write_frame_diagnostics_csv(rows, output_path):
    """Write frame diagnostics rows to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "file",
        "path",
        "hdu_index",
        "instrument",
        "is_carrington_longitude",
        "carrington_reason",
        "ctype1",
        "ctype2",
        "cunit1",
        "cunit2",
        "car_rot",
        "crln_obs",
        "lon0_filename",
        "longitude_axis_available",
        "longitude_axis_error",
        "lon0_header",
        "lon_min",
        "lon_max",
        "lon_step_median",
        "lon_is_increasing",
        "lon_detected_scale",
        "lon_cdelt1_deg",
        "lon_crval1_deg",
        "figure_path",
        "figure_error",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_frame_diagnostic_figure(fits_path, output_path):
    """Save a diagnostic longitude-frame figure when the FITS image is 2D."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax, info = plot_synoptic_aligned(
        fits_path,
        vmin=-100,
        vmax=100,
        force_sine=True,
        prefer_filename_for_gong=True,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return info


# -------------------------------------------------------------
# Demo / CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    base_output_dir = Path(r"C:\Users\luisl\Desktop\testmagnetogram\test_all")
    figure_output_dir = base_output_dir / "frame_diagnostics_images"
    labels = [
        "GONG_mrzqs",
        "GONG_mrbqs",
        "GONG_mrbqj",
        "GONG_mrmqs",
        "GONG_mrnqs",
        *(f"ADAPT_{adapt_map:02d}" for adapt_map in range(1, 12)),
        "HMI_polfil",
        "HMI_SYNC",
        "HMI_small",
    ]

    rows = []
    for label in labels:
        fits_path = find_first_magnetogram_file(base_output_dir / label)
        if fits_path is None:
            rows.append(
                {
                    "label": label,
                    "path": str(base_output_dir / label),
                    "file": "",
                    "is_carrington_longitude": "",
                    "carrington_reason": "No FITS-like magnetogram file found.",
                }
            )
            print(f"{label}: missing FITS file")
            continue

        try:
            info = diagnose_magnetogram_frame(fits_path, label=label)
        except Exception as exc:
            info = {
                "label": label,
                "path": str(fits_path),
                "file": fits_path.name,
                "is_carrington_longitude": "",
                "carrington_reason": f"Diagnostic failed: {exc}",
            }

        figure_path = figure_output_dir / f"{label}.png"
        try:
            save_frame_diagnostic_figure(fits_path, figure_path)
            info["figure_path"] = str(figure_path)
            info["figure_error"] = ""
        except Exception as exc:
            info["figure_path"] = str(figure_path)
            info["figure_error"] = str(exc)

        rows.append(info)
        print(
            f"{label}: Carrington={info.get('is_carrington_longitude')} "
            f"| instrument={info.get('instrument', '')} "
            f"| reason={info.get('carrington_reason', '')} "
            f"| figure_error={info.get('figure_error', '')}"
        )

    csv_path = base_output_dir / "frame_diagnostics.csv"
    write_frame_diagnostics_csv(rows, csv_path)
    print(f"\nFrame diagnostics written to: {csv_path}")
