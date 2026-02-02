"""Synoptic magnetogram utilities (header/filename parsing + aligned plotting).

This module provides small, focused helpers to:

- Read longitude/latitude world coordinates directly from a FITS header (no
  third‑party WCS objects required).
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
- The FITS header is accessed directly; the formulas follow the simple linear
  world‑coordinate relation::

    world = CRVALn + (i - CRPIXn) * CDELTn,  with i starting at 1

- Longitude wrapping uses the ``[0, 360)`` convention.
- Latitude is returned in degrees. If the header encodes sine‑latitude (CSLT),
  it is converted to degrees.

Examples
--------
Basic usage:

>>> from coconut_tools.solarmach_plot import plot_synoptic_aligned
>>> fig, ax, info = plot_synoptic_aligned("/path/to/file.fits", vmin=-100, vmax=100)
>>> info["lon0_used"]
123.0

"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt


def is_carrington_longitude(header, fname=""):
    """
    Heuristic to decide whether the x axis is Carrington longitude.

    Returns
    -------
    is_car : bool
    info : dict
        reason + key header fields used.
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

    # 4) Filename hint (weak)
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
    """
    Build latitude axis from FITS header. Supports degrees-lat and sine-lat.

    Parameters
    ----------
    header : mapping
        FITS header.
    data : ndarray or masked array, optional
        If provided, will be flipped consistently when latitude is reversed.
        Expected shape (..., ny, nx) or (ny, nx).
    output : {"deg","sin"}
        Requested output axis: degrees latitude or sine(latitude).
    one_based_fits : bool
        FITS WCS CRPIX convention is 1-based.
    force_degrees : bool
        Force interpretation of axis values as degrees latitude.
    force_sine : bool
        Force interpretation of axis values as sine-latitude.
    Returns
    -------
    lat_out : ndarray, shape (ny,)
        Latitude axis in requested output.
    data_out : ndarray or masked array or None
        Flipped data if needed, otherwise original data. None if data is None.
    meta : dict
        Diagnostics: detected_mode, flipped, ctype2, cunit2, ny, crpix2, cdelt2_used, crval2_used.
    """
    if "NAXIS2" not in header:
        raise KeyError("NAXIS2 missing from FITS header.")
    ny = int(header["NAXIS2"])

    crpix2 = float(header.get("CRPIX2", 1.0))
    crval2 = float(header.get("CRVAL2", 0.0))

    if "CDELT2" in header and header["CDELT2"] not in (None, ""):
        cdelt2 = float(header["CDELT2"])
    elif "CD2_2" in header and header["CD2_2"] not in (None, ""):
        cdelt2 = float(header["CD2_2"])
    else:
        raise KeyError("Missing CDELT2 (or CD2_2) in FITS header.")

    ctype2 = str(header.get("CTYPE2", "")).strip().upper()
    cunit2 = str(header.get("CUNIT2", "")).strip().lower()

    if one_based_fits:
        pix = np.arange(1, ny + 1, dtype=float)
    else:
        pix = np.arange(ny, dtype=float)

    # Step 1: build raw axis using linear WCS
    # Handle potential micro-degrees in cdelt/crval if unit is not explicit
    cdelt2_used = cdelt2
    crval2_used = crval2
    scale_note = "native"

    if cunit2.startswith("deg"):
        scale_note = "deg_from_cunit2"
    else:
        if abs(cdelt2) > 1000:
            cdelt2_used = cdelt2 / 1e6
            crval2_used = crval2 / 1e6
            scale_note = "microdeg_from_cdelt2"

    y_raw = (pix - crpix2) * cdelt2_used + crval2_used

    # Step 2: decide whether y_raw is degrees-lat or sine-lat
    detected_mode = None
    if force_degrees and force_sine:
        raise ValueError("Choose at most one of force_degrees or force_sine.")

    if force_degrees or (("CRLT" in ctype2) and not force_sine):
        y_deg = y_raw.astype(float)
        detected_mode = "degrees"
    elif force_sine or ("CSLT" in ctype2) or ("sine" in cunit2):
        y_deg = np.degrees(np.arcsin(np.clip(y_raw.astype(float), -1.0, 1.0)))
        detected_mode = "sine_to_degrees"
    else:
        # Heuristic: sine-lat is typically within [-1, 1]
        if (np.nanmin(y_raw) >= -1.05) and (np.nanmax(y_raw) <= 1.05) and (not cunit2.startswith("deg")):
            y_deg = np.degrees(np.arcsin(np.clip(y_raw.astype(float), -1.0, 1.0)))
            detected_mode = "heuristic_sine_to_degrees"
        else:
            y_deg = y_raw.astype(float)
            detected_mode = "degrees_fallback"

    # Step 3: choose output axis
    if output.lower() == "deg":
        lat_out = y_deg
    elif output.lower() == "sin":
        lat_out = np.sin(np.radians(y_deg))
    else:
        raise ValueError('output must be "deg" or "sin".')

    # Step 4: enforce increasing latitude upward
    flipped = False
    data_out = data
    if lat_out[0] > lat_out[-1]:
        flipped = True
        lat_out = lat_out[::-1]
        if data is not None:
            data_out = data[..., ::-1, :]

    meta = dict(
        ny=ny,
        ctype2=ctype2,
        cunit2=cunit2,
        detected_mode=detected_mode,
        flipped=flipped,
        crpix2=crpix2,
        cdelt2_used=float(cdelt2_used),
        crval2_used=float(crval2_used),
        scale_note=scale_note,
    )
    return lat_out, data_out, meta



def detect_instrument(h, fname=""):
    """Detect the likely instrument/vendor from header or filename.

    Args
    ----
    h : collections.abc.Mapping
        FITS header.
    fname : str, optional
        Filename used as a fallback hint.

    Returns
    -------
    str
        ``"HMI"``, ``"GONG"`` or ``"UNKNOWN"``.
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
    """Extract starting Carrington longitude from a GONG‑style filename.

    The function parses filenames like ``mrzqs170404t1814c2189_268.fits.gz`` and
    returns the trailing ``268`` value as a degree quantity.

    Args
    ----
    fname : str
        Input filename.

    Returns
    -------
    int | None
        Parsed degree value if found, otherwise ``None``.
    """
    name = fname.split('/')[-1]
    try:
        # e.g., mrzqs170404t1814c2189_268.fits.gz → 268
        return int(name[22:25])
    except Exception:
        return None

def fits_longitude_deg(header, nx, *, one_based_fits=True):
    """
    Compute the x-axis longitudes in degrees from FITS WCS-like keywords.

    Parameters
    ----------
    header : mapping
        FITS header (e.g. hdul[0].header).
    nx : int
        Number of pixels along x (data.shape[-1]).
    one_based_fits : bool, default True
        FITS WCS convention uses 1-based pixel coordinates for CRPIX.
        Keep True unless you know your CRPIX1 is 0-based.

    Returns
    -------
    x_deg : np.ndarray
        Longitudes in degrees, shape (nx,).
    meta : dict
        Useful diagnostics: crpix1, cdelt1_deg, crval1_deg, cunit1, detected_scale.
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
    """
    Compute the x-axis longitude vector in degrees from a FITS header.

    Parameters
    ----------
    header : mapping
        FITS header (hdul[0].header).
    one_based_fits : bool, default True
        FITS convention uses 1-based pixel coordinates for CRPIX.

    Returns
    -------
    x_deg : np.ndarray
        Longitude in degrees, shape (NAXIS1,).
    meta : dict
        Diagnostics on detected units and scaling.
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

def order_carrington_0_360(lon_native, data, *, tol=1e-6):
    """
    Reorder a Carrington map so that longitude runs from 0 to 360 degrees.

    Parameters
    ----------
    lon_native : ndarray, shape (nx,)
        Native Carrington longitudes (can be in any range, any start).
    data : ndarray or masked array, shape (ny, nx)
        Magnetogram data.
    tol : float
        Tolerance used to stabilize sorting against numerical noise.

    Returns
    -------
    lon_sorted : ndarray, shape (nx,)
        Carrington longitude in [0, 360), sorted.
    data_sorted : ndarray or masked array, shape (ny, nx)
        Data reordered along x.
    """
    lon = np.asarray(lon_native, dtype=float)

    # Normalize to [0, 360)
    lon_mod = np.mod(lon, 360.0)

    # Stabilize ordering if needed
    if tol is not None:
        lon_mod = np.round(lon_mod / tol) * tol

    # Sorting indices
    idx = np.argsort(lon_mod)


    lon_sorted = lon_mod[idx]
    data_sorted = data[..., idx]

    return lon_sorted, data_sorted

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
    """
    Plot a synoptic magnetogram with imshow using lon/lat vectors.

    Assumptions
    ----------
    lon and lat are 1D vectors matching data columns/rows.
    data is already reordered so that lon is increasing and in [0, 360).
    lat is increasing (south to north).

    Returns
    -------
    fig, ax, info
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

def plot_synoptic_aligned(
    fits_path,
    prefer_filename_for_gong=True,
    vmin=None,
    vmax=None,
    cmap="RdBu_r",
    force_degrees=False,
    force_sine=False,
):
    """Plot a synoptic map aligned so the **left edge** is 0° Carrington.

    Workflow
    --------
    1. Read data and header from ``fits_path``.
    2. Build the native longitude axis from the header and infer the starting
       longitude both from the header and, if applicable, from a GONG‑style
       filename.
    3. Choose which starting longitude to use (header by default; filename for
       GONG if ``prefer_filename_for_gong=True``).
    4. Shift/wrap columns so the left edge is at 0°; construct the latitude
       axis in degrees.
    5. Render a diagnostic plot and return ``(fig, ax, info)``.

    Args
    ----
    fits_path : str | pathlib.Path
        Path to the synoptic FITS file.
    prefer_filename_for_gong : bool, optional
        If ``True`` (default), prefer the filename‑encoded longitude for GONG
        products, while still reporting both header and filename values.
    vmin, vmax : float, optional
        Color scaling bounds passed to ``imshow``.
    cmap : str, optional
        Matplotlib colormap name. Default ``"RdBu_r"``.
    force_degrees : bool, optional
        Force latitude axis to be treated as degrees.
    force_sine : bool, optional
        Force latitude axis to be treated as sine‑latitude.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, dict]
        ``(fig, ax, info)`` where ``info`` contains diagnostic keys such as
        ``instrument``, ``cdelt1_deg``, ``lon0_header``, ``lon0_filename`` and
        ``lon0_used``.

    Notes
    -----
    - The x‑extent is set to ``[0, 360]`` after reordering the columns by
      increasing longitude.
    - Missing / invalid data are masked with ``np.ma.masked_invalid``.

    Examples
    --------
    >>> fig, ax, info = plot_synoptic_aligned("/path/to/file.fits", vmin=-100, vmax=100)
    >>> info["instrument"]
    'HMI'
    """
    with fits.open(fits_path) as hdul:
        h = hdul[0].header
        data = np.squeeze(hdul[0].data).astype(float)

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
                f"GONG lon0: filename={lon0_file}°  "
                f"header/table={lon_native[0] % 360:.3f}°"
            )

    # Latitude physique en degrés, axe croissant
    lat_deg, data_sorted2, meta_lat = fits_latitude_axis(
        h,
        data_sorted,
        output="deg"   # si tu gardes cette API, sinon tu enlèves output
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
        vmin=vmax,
        vmax=vmin,
        cmap="RdBu_r",
    )
        
    return fig, ax, info


# -------------------------------------------------------------
# Demo / CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    hmi_file = r"C:/Users/luisl/Documents/Travail/COCONUT/AI_magnetogram/hmi.Synoptic_Mr_small.2219.fits"
    gong_file = r"C:/Users/luisl/Documents/Travail/COCONUT/AI_magnetogram/mrzqs190702t1204c2219_260.fits.gz"

    # --- HMI ---
    fig, ax, info = plot_synoptic_aligned(
        hmi_file,
        vmin=-20,
        vmax=20,
        force_sine=True,
        prefer_filename_for_gong=False,  # inutile pour HMI
    )

    plt.show()
    print("HMI diagnostics:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # --- GONG ---
    fig, ax, info = plot_synoptic_aligned(
        gong_file,
        vmin=-20,
        vmax=20,
        force_sine=True,
        prefer_filename_for_gong=True,
    )
    print("\nGONG diagnostics:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    plt.show()
