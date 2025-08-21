from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt


def _read_lon_wcs_in_deg(h):
    """Return (crpix1, cdelt1_deg, crval1_deg) for longitude axis.
    Converts micro-degrees to degrees when needed (common in some GONG files).
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
    """Return 1D pixel→world coordinate for axis n using FITS WCS (raw units).
    coord = CRVALn + (i - CRPIXn) * CDELTn, with i starting at 1.
    """
    n = int(n)
    nx = int(h[f"NAXIS{n}"])
    crpix = float(h.get(f"CRPIX{n}", nx/2 + 0.5))
    cdelt = float(h.get(f"CDELT{n}", 1.0))
    crval = float(h.get(f"CRVAL{n}", 0.0))
    i = np.arange(nx, dtype=float) + 1.0
    return (i - crpix) * cdelt + crval


def _lon_deg_from_header(h):
    """Longitude axis in degrees (unwrapped). Handles micro-degrees if needed."""
    x_raw = _axis_from_header(h, 1)
    cunit1 = str(h.get("CUNIT1", "")).lower()
    if cunit1.startswith("deg"):
        return x_raw.astype(float)
    if np.nanmax(np.abs(x_raw)) > 1e4:  # likely micro-deg encoding
        return (x_raw / 1e6).astype(float)
    return x_raw.astype(float)


def _lat_deg_from_header(h, force_degrees=False, force_sine=False):
    """Latitude axis in degrees. Handles CRLT (deg) vs CSLT (sine-latitude)."""
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
    """Longitude at left edge (column 0) computed from WCS header, in [0,360)."""
    crpix1, cdelt1_deg, crval1_deg = _read_lon_wcs_in_deg(h)
    lon_start = crval1_deg + (1.0 - crpix1) * cdelt1_deg  # i=1 for first column
    return float(lon_start % 360.0), cdelt1_deg


def start_lon_from_filename(fname):
    """Parse GONG-style filename to extract starting longitude (deg), else None."""
    name = fname.split('/')[-1]
    try:
        # e.g., mrzqs170404t1814c2189_268.fits.gz → 268
        return int(name[22:25])
    except Exception:
        return None

# -------------------------------------------------------------
# Diagnostics + plotting
# -------------------------------------------------------------

def plot_synoptic_aligned(fits_path, prefer_filename_for_gong=True, vmin=None, vmax=None,
                          cmap="RdBu_r", force_degrees=False, force_sine=False):
    """Plot synoptic map aligned so that the left edge is 0° Carrington.

    - Computes start longitude from header (all instruments)
    - If instrument is GONG and a filename hint exists, compares both and
      (by default) uses the filename value for alignment while reporting both.
    Returns (fig, ax, info_dict).
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
    hmi_file = "hmi.Synoptic_Mr_small.2134.fits"
    gong_file = "mrzqs170404t1814c2189_268.fits.gz"

    # --- HMI (header-only) ---
    fig, ax, info = plot_synoptic_aligned(hmi_file, vmin=-100, vmax=100, force_sine=True)
    print("HMI diagnostics:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    plt.show()

    # --- GONG (compare filename vs header; use filename for alignment) ---
    fig, ax, info = plot_synoptic_aligned(gong_file, vmin=-100, vmax=100, force_sine=True,
                                          prefer_filename_for_gong=True)
    print("\nGONG diagnostics:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    plt.show()
