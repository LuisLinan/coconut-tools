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


def _read_lon_wcs_in_deg(h):
    """Return longitude WCS triplet ``(CRPIX1, CDELT1_deg, CRVAL1_deg)``.

    The function inspects the header to determine whether the longitude axis is
    provided in **degrees** or **micro‑degrees** (a quirk present in some GONG
    synoptic products). If the unit field explicitly mentions degrees, it is
    trusted. Otherwise, a simple magnitude‑based heuristic on ``CDELT1`` is used
    to detect micro‑degrees.

    Args
    ----
    h : collections.abc.Mapping
        FITS header (e.g., ``hdul[0].header``) providing CRPIX1/CDELT1/CRVAL1
        and CUNIT1.

    Returns
    -------
    tuple[float, float, float]
        ``(crpix1, cdelt1_deg, crval1_deg)`` where both step and reference value
        are **in degrees**.
    """
    crpix1 = float(h.get("CRPIX1", 1.0))
    cdelt1 = float(h.get("CDELT1", 1.0))
    crval1 = float(h.get("CRVAL1", 0.0))
    cunit1 = str(h.get("CUNIT1", "")).lower()

    # If unit explicitly says degrees, trust it
    if cunit1.startswith("deg"):
        return crpix1, cdelt1, crval1

    # Otherwise detect micro-degrees by magnitude of the step (heuristic)
    # Typical deg/px ~ 0.5 .. 2. If >> 1000, likely micro-deg
    if abs(cdelt1) > 1000:
        return crpix1, cdelt1 / 1e6, crval1 / 1e6

    # Fall back: assume already in degrees
    return crpix1, cdelt1, crval1


def _axis_from_header(h, n):
    """Compute a 1‑D world coordinate vector from a FITS header.

    Uses the linear relation
    ``coord = CRVALn + (i - CRPIXn) * CDELTn`` with pixel index ``i`` starting
    at **1** as per FITS/WCS conventions.

    Args
    ----
    h : collections.abc.Mapping
        FITS header.
    n : int
        Axis index (1‑based).

    Returns
    -------
    numpy.ndarray
        The world coordinate values (in the *raw* units encoded in the header).
    """
    n = int(n)
    nx = int(h[f"NAXIS{n}"])
    crpix = float(h.get(f"CRPIX{n}", nx / 2 + 0.5))
    cdelt = float(h.get(f"CDELT{n}", 1.0))
    crval = float(h.get(f"CRVAL{n}", 0.0))
    i = np.arange(nx, dtype=float) + 1.0
    return (i - crpix) * cdelt + crval


def _lon_deg_from_header(h):
    """Return longitude vector in **degrees** (unwrapped) from a FITS header.

    If the header unit is degrees it is used directly; otherwise a simple
    magnitude check is used to detect micro‑degree encodings and convert them to
    degrees.

    Args
    ----
    h : collections.abc.Mapping
        FITS header.

    Returns
    -------
    numpy.ndarray
        Longitudes in degrees (not wrapped to 0–360).
    """
    x_raw = _axis_from_header(h, 1)
    cunit1 = str(h.get("CUNIT1", "")).lower()
    if cunit1.startswith("deg"):
        return x_raw.astype(float)
    if np.nanmax(np.abs(x_raw)) > 1e4:  # likely micro-deg encoding
        return (x_raw / 1e6).astype(float)
    return x_raw.astype(float)


def _lat_deg_from_header(h, force_degrees=False, force_sine=False):
    """Return latitude vector in **degrees** from a FITS header.

    The function supports two common encodings of the latitude axis:

    - Degrees (``CRLT``), returned unchanged.
    - Sine‑latitude (``CSLT``), converted to degrees via ``arcsin``.

    You can override detection with ``force_degrees`` or ``force_sine``.

    Args
    ----
    h : collections.abc.Mapping
        FITS header.
    force_degrees : bool, optional
        Force interpretation as degrees.
    force_sine : bool, optional
        Force interpretation as sine‑latitude.

    Returns
    -------
    tuple[numpy.ndarray, str]
        The latitude array in degrees and a short description of the chosen
        mode (for diagnostics).
    """
    y_raw = _axis_from_header(h, 2)
    ctype2 = str(h.get("CTYPE2", "")).upper()
    cunit2 = str(h.get("CUNIT2", "")).lower()

    if force_degrees or ("CRLT" in ctype2 and not force_sine):
        lat = y_raw
        mode = "degrees (CRLT)"
    elif force_sine or ("CSLT" in ctype2 or "sine" in cunit2):
        lat = np.degrees(np.arcsin(np.clip(y_raw, -1.0, 1.0)))
        mode = "sine-lat → degrees"
    else:
        if y_raw.min() >= -1.05 and y_raw.max() <= 1.05 and not cunit2:
            lat = np.degrees(np.arcsin(np.clip(y_raw, -1.0, 1.0)))
            mode = "heuristic sine-lat → degrees"
        else:
            lat = y_raw
            mode = "degrees (fallback)"

    if lat[0] > lat[-1]:  # ensure latitude increases upward
        lat = lat[::-1]
    return lat, mode


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

def start_lon_from_header(h):
    """Compute the left‑edge Carrington longitude from header (degrees).

    The returned longitude is wrapped to ``[0, 360)``. The pixel index for the
    first column is assumed to be 1, as per WCS.

    Args
    ----
    h : collections.abc.Mapping
        FITS header.

    Returns
    -------
    tuple[float, float]
        ``(lon_start_deg, cdelt1_deg)``.
    """
    crpix1, cdelt1_deg, crval1_deg = _read_lon_wcs_in_deg(h)
    lon_start = crval1_deg + (1.0 - crpix1) * cdelt1_deg  # i=1 for first column
    return float(lon_start % 360.0), cdelt1_deg


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


# -------------------------------------------------------------
# Diagnostics + plotting
# -------------------------------------------------------------

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

    # Longitudes
    lon_native = _lon_deg_from_header(h)
    lon0_header, cdelt1_deg = start_lon_from_header(h)

    lon0_used = lon0_header
    lon0_file = None
    if inst == "GONG":
        lon0_file = start_lon_from_filename(fits_path)
        if lon0_file is not None and prefer_filename_for_gong:
            lon0_used = float(lon0_file)

    # Align: make left edge 0° by subtracting chosen lon0 and wrapping
    lon = (lon_native - lon0_used) % 360.0

    # Sort columns by longitude increasing (ensures extent [0,360])
    order = np.argsort(lon)
    lon = lon[order]
    data = data[:, order]

    # Latitude
    lat, lat_mode = _lat_deg_from_header(h, force_degrees=force_degrees, force_sine=force_sine)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(
        data,
        origin="lower",
        extent=[lon[0], lon[-1], lat[0], lat[-1]],
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(f"Br [{h.get('BUNIT', 'G')}]")

    # Title and labels
    cr = h.get('CAR_ROT', '')
    ax.set_xlabel("Carrington longitude [deg]")
    ax.set_ylabel("Latitude [deg]")

    # Diagnostics string
    ctype1 = str(h.get("CTYPE1", "")).upper()
    cunit1 = str(h.get("CUNIT1", "")).lower()
    diag = [
        f"Inst={inst}  CTYPE1={ctype1}  CUNIT1={cunit1 or '(none)'}  Δpix={cdelt1_deg:.6f}°/px",
        f"lon0(header)={lon0_header:.3f}°",
    ]
    if lon0_file is not None:
        diag.append(f"lon0(filename)={lon0_file:.3f}°  Δ={((lon0_header - lon0_file + 540)%360-180):.3f}°")
    diag.append(f"lon0(used)={lon0_used:.3f}°")

    ax.set_title(
        f"Synoptic Br — CR {cr}  | lat axis: {lat_mode}\n" +
        "  |  ".join(diag)
    )

    ax.set_xlim(0, 360)
    ax.set_ylim(lat[0], lat[-1])

    info = {
        "instrument": inst,
        "ctype1": ctype1,
        "cunit1": cunit1,
        "cdelt1_deg": cdelt1_deg,
        "lon0_header": lon0_header,
        "lon0_filename": lon0_file,
        "lon0_used": lon0_used,
    }
    return fig, ax, info


# -------------------------------------------------------------
# Demo / CLI
# -------------------------------------------------------------
if __name__ == "__main__":
    # Example files (edit paths as needed)
    hmi_file = "C:/Users/luisl/Desktop/hmi.Synoptic_Mr_small.2134.fits"
    gong_file = "mrzqs170404t1814c2189_268.fits.gz"

    # --- HMI (header-only) ---
    fig, ax, info = plot_synoptic_aligned(hmi_file, vmin=-100, vmax=100, force_sine=True)
    print("HMI diagnostics:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    plt.show()

    # --- GONG (compare filename vs header; use filename for alignment) ---
    fig, ax, info = plot_synoptic_aligned(
        gong_file,
        vmin=-100,
        vmax=100,
        force_sine=True,
        prefer_filename_for_gong=True,
    )
    print("\nGONG diagnostics:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    plt.show()
