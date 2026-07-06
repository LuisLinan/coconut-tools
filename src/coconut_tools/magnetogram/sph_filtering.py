"""
Read, preprocess, filter, write, and plot magnetogram boundary maps.

This module implements the spherical-harmonic preprocessing pipeline used to
turn photospheric magnetograms into COCONUT boundary-condition files. Download
and remote-candidate selection are delegated to
``coconut_tools.magnetogram.magnetogram_download``; this file handles local map
reading, latitude/longitude normalization, optional temporal interpolation,
optional Carrington-to-Stonyhurst rotation, optional flux balancing, spherical
harmonic projection/reconstruction, diagnostics, and figure generation.

Date handling follows the common magnetogram "effective time" convention:
interpolated GONG/ADAPT maps represent the requested target time, HMI small,
HMI polar-filled, and WSO use the target time by convention, and
non-interpolated GONG/ADAPT/HMI_SYNC maps use the timestamp encoded in the
selected filename.
"""
import os
from datetime import datetime
from typing import Any

import numpy as np
from astropy.io import fits
from scipy import interpolate
from scipy import special as scisp
import matplotlib.pyplot as plt
from coconut_tools.magnetogram.magnetogram_download import (
    InterpolationSelection,
    build_processing_dates,
    default_figure_path,
    generate_output_and_interpolation_map_names,
    generate_output_and_map_names,
    is_gong_map_type,
    is_gong_temporal_map_type,
    magnetogram_effective_date,
    magnetogram_display_date,
    normalize_map_type,
    parse_iso_datetime,
    resolve_figure_path,
)
from coconut_tools.tools.logger_config import setup_logger
from coconut_tools.tools.rotation_angle import (
    compute_carrington_central_meridian,
    compute_rotation_angle,
    increasing_longitude_axis,
    is_br_longitude_increasing,
)

logger = setup_logger(__name__)

_PLOT_COLOR_LIMIT_PERCENTILE = 99.0

def build_theta_phi(theta, phi):
    """Build 2D meshgrids from 1D theta and phi arrays.

    Args:
        theta (ndarray): 1D array of theta values (colatitude).
        phi (ndarray): 1D array of phi values (longitude).

    Returns:
        tuple: (Theta, Phi) meshgrids.
    """
    return np.tile(theta, (len(phi), 1)).T, np.tile(phi, (len(theta), 1))


def spherical_harmonic(m: int, l: int, Phi: np.ndarray, Theta: np.ndarray) -> np.ndarray:
    """Evaluate spherical harmonics across SciPy versions.

    Args:
        m (int): Harmonic order.
        l (int): Harmonic degree.
        Phi (ndarray): Longitude grid.
        Theta (ndarray): Colatitude grid.

    Returns:
        ndarray: Complex spherical harmonic values.
    """
    if hasattr(scisp, "sph_harm"):
        return scisp.sph_harm(m, l, Phi, Theta)
    return scisp.sph_harm_y(l, m, Theta, Phi)


def extract_gong_longitude_shift(file_path: str) -> int:
    """Extract the GONG longitude offset encoded in a filename.

    Args:
        file_path (str): Local GONG filename or path.

    Returns:
        int: Number of longitude cells used for circular shifting.
    """
    name = os.path.basename(file_path)
    try:
        return int(name.split("_")[-1].split(".")[0]) - 1
    except (IndexError, ValueError):
        logger.warning(f"Could not parse GONG longitude shift from {name}; using 0.")
        return 0


def circular_shift_longitude(Br: np.ndarray, shift: int) -> np.ndarray:
    """Apply a circular longitude shift to a magnetogram.

    Args:
        Br (np.ndarray): Magnetic field map.
        shift (int): Number of longitude cells to shift right.

    Returns:
        np.ndarray: Shifted magnetic field map.
    """
    nb_phi = Br.shape[1]
    shift = shift % nb_phi
    if shift == 0:
        return Br
    return np.hstack((Br[:, -shift:], Br[:, :-shift]))


def ensure_increasing_longitude(
    Br: np.ndarray,
    file_path: str,
    map_type: str,
) -> np.ndarray:
    """Return Br with columns ordered by increasing native longitude.

    FITS WCS keywords are used for products whose longitude direction can be
    read from the header. WSO text maps are left unchanged. HMI maps are kept as
    read because the current pipeline treats their FITS products as already
    increasing in longitude.
    """
    map_type = normalize_map_type(map_type)
    if map_type.lower() == "wso":
        return Br
    if "hmi" in map_type.lower():
        logger.info("HMI maps are assumed to have increasing longitude.")
        return Br
    if is_br_longitude_increasing(file_path):
        return Br
    logger.info("Flipping Br columns to obtain increasing longitude.")
    return np.ascontiguousarray(Br[:, ::-1])

def rotate_longitude_to_stonyhurst(
    Br: np.ndarray,
    angle_degrees: float,
    has_duplicate_endpoint: bool = False,
    zero_column: int | None = None,
) -> np.ndarray:
    """Roll an increasing-longitude map into the requested Stonyhurst frame.

    Args:
        Br (np.ndarray): Map whose longitude columns are already increasing.
        angle_degrees (float): Carrington longitude of the Stonyhurst zero
            meridian, used only to infer ``zero_column`` when it is omitted.
        has_duplicate_endpoint (bool): True when the last longitude column
            duplicates the first one, as in WSO maps with both 0 and 360 deg.
        zero_column (int | None): Column that should become the first column in
            the rotated map. If omitted, it is derived from ``angle_degrees``.

    Returns:
        np.ndarray: Longitude-rolled map, preserving any duplicate endpoint.
    """
    unique_longitudes = Br.shape[1] - 1 if has_duplicate_endpoint else Br.shape[1]
    if zero_column is None:
        zero_column = round((angle_degrees % 360.0) / 360.0 * unique_longitudes)
    shift = -zero_column
    logger.info(
        "Rotating Br to Stonyhurst by %.6f degrees (%d longitude cells).",
        angle_degrees,
        shift,
    )
    if not has_duplicate_endpoint:
        return np.roll(Br, shift=shift, axis=1)

    rotated = np.roll(Br[:, :-1], shift=shift, axis=1)
    return np.hstack((rotated, rotated[:, :1]))


def processed_longitude_axis(
    file_path: str,
    map_type: str,
    temporal: bool = False,
) -> np.ndarray:
    """Return longitude cell centers matching the processed Br column order.

    The returned array follows the same longitude ordering as maps produced by
    ``read_magnetogram`` or ``read_temporal_br_map``. For interpolated temporal
    GONG maps, the filename-encoded circular shift is applied so the longitude
    axis matches the map that was shifted before interpolation.
    """
    map_type = normalize_map_type(map_type)
    if map_type.lower() == "wso":
        return np.linspace(0.0, 360.0, 73)

    lon = increasing_longitude_axis(file_path)
    if temporal and is_gong_map_type(map_type):
        lon = np.roll(lon, extract_gong_longitude_shift(file_path))
    return lon


def closest_longitude_column(longitude: np.ndarray, target_degrees: float) -> tuple[int, float]:
    """Find the longitude column closest to a periodic target angle.

    Args:
        longitude (np.ndarray): Longitude cell centers in degrees.
        target_degrees (float): Target longitude in degrees.

    Returns:
        tuple[int, float]: Index of the closest column and signed residual in
        degrees, wrapped into [-180, 180).
    """
    residuals = (np.asarray(longitude) - target_degrees + 180.0) % 360.0 - 180.0
    index = int(np.nanargmin(np.abs(residuals)))
    return index, float(residuals[index])


def apply_configured_longitude_rotation(
    Br: np.ndarray,
    Br_linear: np.ndarray | None,
    local_file: str | list[str],
    map_type: str,
    target_date: str | datetime,
    use_interpolation: bool,
    rotate_to_stonyhurst: bool,
    effective_date: str | datetime | None = None,
) -> tuple[np.ndarray, np.ndarray | None, float | None]:
    """Apply the configured Carrington-to-Stonyhurst longitude rotation.

    The rotation date is the magnetogram effective time. For interpolated
    GONG/ADAPT products this is the requested target time because the final map
    is synthesized at that time. For non-interpolated products it is either the
    timestamp encoded in the selected filename or the product convention
    returned by ``magnetogram_effective_date``.

    Args:
        Br (np.ndarray): Processed radial field map.
        Br_linear (np.ndarray | None): Linear interpolation reference map, when
            available. It is rotated with the same shift as ``Br``.
        local_file (str | list[str]): Source magnetogram path, or the
            interpolation stencil paths.
        map_type (str): Magnetogram product type.
        target_date (str | datetime): Requested processing time.
        use_interpolation (bool): Whether temporal interpolation was requested.
        rotate_to_stonyhurst (bool): If false, maps are returned unchanged.
        effective_date (str | datetime | None): Precomputed effective time. If
            omitted, it is derived from the source file and map type.

    Returns:
        tuple[np.ndarray, np.ndarray | None, float | None]: Rotated ``Br``,
        rotated ``Br_linear`` if present, and rotation angle in degrees. The
        angle is ``None`` when rotation is disabled.
    """
    map_type = normalize_map_type(map_type)
    if not rotate_to_stonyhurst:
        return Br, Br_linear, None

    source_file = local_file[0] if isinstance(local_file, list) else local_file
    interpolated = use_interpolation and (
        is_gong_temporal_map_type(map_type) or map_type == "ADAPT"
    )
    rotation_date = (
        parse_iso_datetime(effective_date)
        if effective_date is not None
        else magnetogram_effective_date(
            source_file,
            map_type,
            target_date,
            interpolated=interpolated,
        )
    )

    if interpolated:
        # Interpolated GONG maps have already been shifted from their
        # filename-encoded origin to Carrington zero before interpolation.
        rotation_angle = compute_carrington_central_meridian(rotation_date)
    else:
        rotation_angle, rotation_date = compute_rotation_angle(
            source_file,
            date_hmi=parse_iso_datetime(target_date).isoformat(),
            map_type=map_type,
            interpolated=False,
        )

    longitude = processed_longitude_axis(
        source_file,
        map_type,
        temporal=interpolated and is_gong_temporal_map_type(map_type),
    )
    if is_gong_map_type(map_type):
        target_longitude = compute_carrington_central_meridian(rotation_date)
    else:
        target_longitude = rotation_angle
    zero_column, residual = closest_longitude_column(longitude, target_longitude)
    logger.info(
        "Stonyhurst zero uses longitude column %d with residual %.6f degrees.",
        zero_column,
        residual,
    )

    has_duplicate_endpoint = map_type.lower() == "wso"
    Br = rotate_longitude_to_stonyhurst(
        Br,
        rotation_angle,
        has_duplicate_endpoint=has_duplicate_endpoint,
        zero_column=zero_column,
    )
    if Br_linear is not None:
        Br_linear = rotate_longitude_to_stonyhurst(
            Br_linear,
            rotation_angle,
            has_duplicate_endpoint=has_duplicate_endpoint,
            zero_column=zero_column,
        )
    return Br, Br_linear, rotation_angle


def read_first_fits_image(file_path: str) -> np.ndarray:
    """Read the first FITS HDU that contains at least a 2D image array."""
    with fits.open(file_path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                return np.asarray(hdu.data)
    raise ValueError(f"No magnetogram image HDU found in FITS file: {file_path}")


def build_regular_theta_phi(Br: np.ndarray, map_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Build regular colatitude and longitude grids for a processed Br map.

    ADAPT maps are represented on a regular colatitude grid. Other supported
    FITS maps are represented on a regular sine-latitude grid converted to
    colatitude. Longitudes are endpoint-free in [0, 2*pi).

    Args:
        Br (np.ndarray): Magnetic field map.
        map_type (str): Map type.

    Returns:
        tuple[np.ndarray, np.ndarray]: One-dimensional colatitude ``theta`` and
        longitude ``phi`` vectors in radians.
    """
    map_type = normalize_map_type(map_type)
    nb_th, nb_phi = Br.shape
    if map_type == "ADAPT":
        theta = np.linspace(0.0, np.pi, nb_th)
        phi = np.linspace(0.0, 2.0 * np.pi, nb_phi, endpoint=False)
    else:
        sinlat = np.linspace(-1.0, 1.0, nb_th)
        theta = np.arcsin(sinlat) + np.pi / 2.0
        phi = np.linspace(0.0, 2.0 * np.pi, nb_phi, endpoint=False)
    return theta, phi


def read_temporal_br_map(file_path: str, map_type: str, adapt_map: int = 0) -> np.ndarray:
    """Read and normalize one FITS magnetogram used in interpolation.

    The output contains only Br, not the meshgrid. Latitude is flipped into the
    pipeline convention, longitude is normalized to increasing order, and
    temporal GONG maps are circularly shifted so column zero corresponds to the
    Carrington-zero convention before interpolation.

    Args:
        file_path (str): Local FITS file.
        map_type (str): Map type. Supported: temporal GONG variants and ADAPT.
        adapt_map (int): ADAPT realization index.

    Returns:
        np.ndarray: Normalized radial magnetic field map.
    """
    map_type = normalize_map_type(map_type)
    input_data = read_first_fits_image(file_path)
    if map_type == "ADAPT":
        Br = np.nan_to_num(input_data[adapt_map, ::-1, :])
        return ensure_increasing_longitude(Br, file_path, map_type)
    if is_gong_map_type(map_type):
        Br = np.nan_to_num(input_data[::-1, :])
        Br = ensure_increasing_longitude(Br, file_path, map_type)
        return circular_shift_longitude(Br, extract_gong_longitude_shift(file_path))
    raise ValueError(f"Temporal interpolation is not supported for {map_type}")


def interpolate_br_maps(
    Br_maps: list[np.ndarray],
    selection: InterpolationSelection,
    interpolation_order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a four-map temporal stencil onto the target time.

    ``interpolation_order=1`` returns the linear interpolation between the two
    bracketing maps. ``interpolation_order=2`` uses a cubic Hermite estimate
    based on the before-previous, before, after, and after-next maps. In both
    cases the second returned array is the linear reference map used for
    diagnostics.

    Args:
        Br_maps (list[np.ndarray]): Maps ordered as before-previous, before, after, after-next.
        selection (InterpolationSelection): Time stencil and weights.
        interpolation_order (int): 1 for linear, 2 for cubic Hermite.

    Returns:
        tuple[np.ndarray, np.ndarray]: Interpolated Br and linear Br reference.
    """
    Br_datam, Br_data, Br_data1, Br_datap = Br_maps
    Br_linear = selection.coef_before * Br_data + selection.coef_after * Br_data1
    if interpolation_order == 1:
        return np.nan_to_num(Br_linear), np.nan_to_num(Br_linear)
    if interpolation_order != 2:
        raise ValueError("interpolation_order must be 1 or 2.")

    time_norm = selection.coef_after
    h00 = 2.0 * time_norm**3.0 - 3.0 * time_norm**2.0 + 1.0
    h10 = time_norm**3.0 - 2.0 * time_norm**2.0 + time_norm
    h01 = -2.0 * time_norm**3.0 + 3.0 * time_norm**2.0
    h11 = time_norm**3.0 - time_norm**2.0

    derivative1 = 0.5 * (
        (Br_data - Br_datam) / selection.previous_interval_seconds
        + (Br_data1 - Br_data) / selection.interval_seconds
    )
    derivative2 = 0.5 * (
        (Br_datap - Br_data1) / selection.next_interval_seconds
        + (Br_data1 - Br_data) / selection.interval_seconds
    )
    Br_interpolated = (
        Br_data * h00
        + derivative1 * selection.interval_seconds * h10
        + Br_data1 * h01
        + derivative2 * selection.interval_seconds * h11
    )
    return np.nan_to_num(Br_interpolated), np.nan_to_num(Br_linear)


def read_interpolated_magnetogram(
    local_files: list[str],
    map_type: str,
    selection: InterpolationSelection,
    adapt_map: int = 0,
    interpolation_order: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read, normalize, and temporally interpolate a four-map stencil.

    The source maps are converted into the same latitude/longitude convention
    before interpolation. The returned ``Theta`` and ``Phi`` grids describe the
    interpolated map, while ``Br_linear`` is kept as a diagnostic reference for
    comparing cubic Hermite interpolation against the simpler linear result.

    Args:
        local_files (list[str]): Local files in stencil order.
        map_type (str): Map type. Supported: temporal GONG variants and ADAPT.
        selection (InterpolationSelection): Time interpolation metadata.
        adapt_map (int): ADAPT realization index.
        interpolation_order (int): 1 for linear, 2 for cubic Hermite.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Br, Theta, Phi, Br_linear.
    """
    map_type = normalize_map_type(map_type)
    logger.info("Reading interpolation stencil")
    Br_maps = [read_temporal_br_map(path, map_type, adapt_map) for path in local_files]
    shapes = {Br.shape for Br in Br_maps}
    if len(shapes) != 1:
        raise RuntimeError(f"Interpolation stencil has inconsistent shapes: {shapes}")
    Br, Br_linear = interpolate_br_maps(Br_maps, selection, interpolation_order)
    theta, phi = build_regular_theta_phi(Br, map_type)
    Theta, Phi = build_theta_phi(theta, phi)
    logger.info("End of reading interpolation stencil")
    return Br, Theta, Phi, Br_linear


def read_magnetogram(file_path, map_type, adapt_map=0):
    """Read one non-interpolated magnetogram and build its spherical grid.

    The map is converted into the pipeline convention: latitude index ordered
    from north to south after the final Br assignment, longitude columns ordered
    according to ``ensure_increasing_longitude``, ``theta`` as colatitude in
    radians, and ``phi`` as longitude in radians. WSO text maps are parsed from
    their native format and converted to Gauss.

    Args:
        file_path (str): Path to the magnetogram file.
        map_type (str): Map product type, for example ``WSO``, ``ADAPT``,
            ``GONG_mrzqs``, ``GONG_mrbqs``, ``GONG_mrbqj``, ``GONG_mrmqs``,
            ``GONG_mrnqs``, ``HMI_small``, ``HMI_polfil``, or ``HMI_SYNC``.
        adapt_map (int, optional): Index for ADAPT map. Defaults to 0.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Br map and 2D ``Theta`` and
        ``Phi`` grids in radians.
    """
    map_type = normalize_map_type(map_type)
    logger.info('Reading file')

    if map_type == 'ADAPT':
        input_data = read_first_fits_image(file_path)
        Br_map = input_data[adapt_map, ::-1, :]
        Br_map = ensure_increasing_longitude(Br_map, file_path, map_type)
        nb_th, nb_phi = Br_map.shape
        theta = np.linspace(0., np.pi, nb_th)
        phi = np.linspace(0., 2.0*np.pi, nb_phi, endpoint=False)
        Theta, Phi = build_theta_phi(theta, phi)

    elif map_type.lower() == 'wso':
        with open(file_path, 'r') as fwso:
            line = fwso.readline()
            while line.strip() == '':
                line = fwso.readline()
            header = line.split()
            lat_type = 'sinlat' if 'sine' in header else 'lat'
            nb_th = int(header[1])
            nb_phi = 73
            nb_lines = 4 * nb_phi
            nb_thplus = 4
            nb_th2 = nb_th + 2 * nb_thplus
            Br_read = np.zeros((nb_th, nb_phi))
            fwso.readline()
            idx_ph = nb_phi
            for _ in range(nb_lines):
                line = fwso.readline()
                split_line = line.split()
                if split_line[0].startswith('C'):
                    idx_th = 0
                    idx_ph -= 1
                    for val in split_line[1:]:
                        Br_read[idx_th, idx_ph] = float(val)
                        idx_th += 1
                else:
                    for val in split_line:
                        Br_read[idx_th, idx_ph] = float(val)
                        idx_th += 1

        if lat_type == 'lat':
            Br_ext = np.pad(Br_read, ((nb_thplus, nb_thplus), (0, 0)), mode='edge')
            Br_map = Br_ext * 0.01
            nb_th = Br_map.shape[0]
            theta = (np.linspace(-90., 90., nb_th) + 90.) * np.pi / 180.
        else:
            Br_data = Br_read[::-1, :] * 0.01
            sinlat = np.linspace(-14.5/15., 14.5/15., nb_th)
            theta_map = np.arcsin(sinlat) + np.pi/2.
            theta = np.linspace(0., np.pi, nb_th)
            phi = np.linspace(0., 360., nb_phi) * np.pi / 180.
            fbr = interpolate.RectBivariateSpline(theta_map, phi, Br_data)
            Br_map = fbr(theta, phi)[::-1, :]

        phi = np.linspace(0., 360., nb_phi) * np.pi / 180.
        Theta, Phi = build_theta_phi(theta, phi)

    else:
        input_data = read_first_fits_image(file_path)
        Br_data = input_data[::-1, :]
        Br_data = ensure_increasing_longitude(Br_data, file_path, map_type)
        nb_th, nb_phi = Br_data.shape
        sinlat = np.linspace(-1., 1., nb_th)
        theta = np.arcsin(sinlat) + np.pi/2.
        phi = np.linspace(0., 2.0*np.pi, nb_phi, endpoint=False)
        Theta, Phi = build_theta_phi(theta, phi)
        Br_map = np.nan_to_num(Br_data)

    logger.info("End of reading file")


    return Br_map, Theta, Phi

def project_and_reconstruct(Br, Theta, Phi, lmax, amp=1, alpha=0):
    """Project Br onto spherical harmonics and reconstruct the filtered map.

    Uses complex spherical harmonics with coefficients stored only for m >= 0.
    For a real field, the missing negative-m modes are recovered during
    reconstruction by adding 2 * real(a_lm * Y_lm) for m > 0.
    """
    logger.info("Beginning of projection")

    Br = np.asarray(Br)
    Theta = np.asarray(Theta)
    Phi = np.asarray(Phi)

    if Br.shape != Theta.shape or Br.shape != Phi.shape:
        raise ValueError("Br, Theta, and Phi must have the same 2D shape.")
    if Br.ndim != 2:
        raise ValueError("Br, Theta, and Phi must be 2D arrays.")
    if lmax < 1:
        raise ValueError("lmax must be >= 1.")
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")

    nb_th, nb_phi = Br.shape
    nb_modes_tot = int((lmax + 1) * (lmax + 2) / 2 - 1)

    theta = Theta[:, 0]
    phi = Phi[0, :]

    # Cell-width integration: more faithful than np.gradient for cell-centered grids.
    theta_edges = np.empty(theta.size + 1)
    theta_edges[0] = 0.0
    theta_edges[-1] = np.pi
    theta_edges[1:-1] = 0.5 * (theta[:-1] + theta[1:])
    dtheta_1d = np.diff(theta_edges)

    phi_unwrapped = np.unwrap(phi)
    dphi_default = 2.0 * np.pi / nb_phi
    phi_edges = np.empty(phi.size + 1)
    phi_edges[1:-1] = 0.5 * (phi_unwrapped[:-1] + phi_unwrapped[1:])
    phi_edges[0] = phi_unwrapped[0] - 0.5 * dphi_default
    phi_edges[-1] = phi_edges[0] + 2.0 * np.pi
    dphi_1d = np.diff(phi_edges)

    dtheta = np.tile(dtheta_1d, (nb_phi, 1)).T
    dphi = np.tile(dphi_1d, (nb_th, 1))

    surface_weight = np.sin(Theta) * dtheta * dphi

    coefbr = np.zeros(nb_modes_tot, dtype=complex)

    mod = 0
    for l in range(1, lmax + 1):
        logger.info(f"l = {l}")
        damping = 1.0 / (1.0 + alpha * l**2 * (l + 1) ** 2)

        for m in range(0, l + 1):
            ylm = spherical_harmonic(m, l, Phi, Theta)

            coef = np.sum(Br * np.conj(ylm) * surface_weight)
            coefbr[mod] = damping * coef

            mod += 1

    logger.info("End of projection")

    logger.info("Reconstructing Br")
    Br_mode = np.zeros_like(Br, dtype=float)

    mod = 0
    for l in range(1, lmax + 1):
        logger.info(f"l = {l}")

        for m in range(0, l + 1):
            ylm = spherical_harmonic(m, l, Phi, Theta)
            contribution = np.real(coefbr[mod] * ylm)

            if m > 0:
                contribution *= 2.0

            Br_mode += contribution
            mod += 1

    Br_mode /= 2.2
    Br_mode *= amp

    logger.info("End of reconstructing Br")
    return Br_mode, coefbr

def write_bc_file(output_name, Br_mode, theta, phi, r_st):
    """Write a COCONUT photospheric boundary-condition file.

    The file contains one spherical surface at radius ``r_st``. Grid points at
    the two poles are written once, while non-polar cells are written for every
    longitude column.

    Args:
        output_name (str): Path to output file.
        Br_mode (ndarray): Reconstructed radial field.
        theta (ndarray): 1D colatitude grid in radians.
        phi (ndarray): 1D longitude grid in radians.
        r_st (float): Spherical radius.
    """
    logger.info("Writing BC file")
    output_dir = os.path.dirname(output_name)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    nb_th, nb_phi = Br_mode.shape
    with open(output_name, 'w') as F:
        F.write('1 \n')
        F.write(f'!PHOTOSPHERE {(nb_th - 2) * nb_phi + 2} \n')
        for j in range(nb_th):
            for k in range(nb_phi):
                if ((j == 0 or j == nb_th - 1) and k != 0):
                    continue
                xcoord = r_st * np.sin(theta[j]) * np.cos(phi[k])
                ycoord = r_st * np.sin(theta[j]) * np.sin(phi[k])
                zcoord = r_st * np.cos(theta[j])
                F.write(f"{xcoord:.16e} {ycoord:.16e} {zcoord:.16e} {Br_mode[j, k]:.16e} \n")
    logger.info("End of writing BC file")


def _symmetric_color_limit(values, percentile=_PLOT_COLOR_LIMIT_PERCENTILE):
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
        return 'neither'

    extend_min = np.nanmin(finite_values) < -limit
    extend_max = np.nanmax(finite_values) > limit
    if extend_min and extend_max:
        return 'both'
    if extend_min:
        return 'min'
    if extend_max:
        return 'max'
    return 'neither'


def plot_maps(
    Br,
    Br_mode,
    theta,
    phi,
    map_type,
    visu_type,
    output_path='output_map.png',
    date=None,
):
    """Save a two-panel diagnostic figure for input and processed Br maps.

    The upper panel shows the pre-filtered/pre-projection Br map after all
    configured preprocessing. The lower panel shows the processed map that is
    written to the boundary file. The displayed date should be the magnetogram
    effective time.

    Args:
        Br (ndarray): Original radial magnetic field.
        Br_mode (ndarray): Reconstructed radial magnetic field.
        theta (ndarray): 1D colatitude grid in radians.
        phi (ndarray): 1D longitude grid in radians.
        map_type (str): Type of the input map ('WSO', 'GONG', etc.).
        visu_type (str): Visualization style ('lat' or 'sinlat').
        output_path (str): Path where the figure will be saved.
        date (str | datetime, optional): Processed date to display on the figure.
    """
    logger.info("Plotting maps")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    if date is not None:
        date_label = parse_iso_datetime(date).strftime("%Y-%m-%d %H:%M:%S")
        fig.suptitle(f"{map_type} magnetogram - {date_label} UTC", fontsize=14)


    lat = 90. - 180. * theta / np.pi
    longi = 180. * phi / np.pi
    Lat, Long = np.meshgrid(lat, longi, indexing='ij')
    Sinlat = np.sin(np.radians(Lat))
    Sinlong = Long

    vmax1 = _symmetric_color_limit(Br)
    vmax2 = _symmetric_color_limit(Br_mode)

    def stats(name, B):
        print(
            name,
            "min", np.nanmin(B),
            "max", np.nanmax(B),
            "absmax", np.nanmax(np.abs(B)),
            "p99 abs", np.nanpercentile(np.abs(B), 99),
            "mean abs", np.nanmean(np.abs(B)),
        )

    stats("original", Br)
    stats("processed", Br_mode)


    # Plot original map
    if visu_type == 'lat':
        im1 = ax1.imshow(
            Br[::-1], aspect='auto', origin='lower', cmap='seismic',
            extent=[Long.min(), Long.max(), Lat.min(), Lat.max()],
            vmin=-vmax1, vmax=vmax1
        )
        ax1.set_ylabel('Latitude', fontsize=14)
    else:
        im1 = ax1.imshow(
            Br[::-1], aspect='auto', origin='lower', cmap='seismic',
            extent=[Sinlong.min(), Sinlong.max(), Sinlat.min(), Sinlat.max()],
            vmin=-vmax1, vmax=vmax1
        )
        ax1.set_ylabel('Sine Latitude', fontsize=14)

    ax1.set_title('Original magnetogram', fontsize=16)
    ax1.set_xticks(np.arange(0., 360., 60.))
    ax1.tick_params(axis='both', which='major', labelsize=12)
    cbar1 = plt.colorbar(im1, ax=ax1, extend=_colorbar_extend(Br, vmax1))
    cbar1.set_label('Br [G]', fontsize=14)
    cbar1.ax.tick_params(labelsize=12)

    # Plot processed input map
    if visu_type == 'lat':
        extent_map = [Long.min(), Long.max(), Lat.min(), Lat.max()]
        ylabel = 'Latitude'
    else:
        extent_map = [Sinlong.min(), Sinlong.max(), Sinlat.min(), Sinlat.max()]
        ylabel = 'Sine Latitude'

    im2 = ax2.imshow(
        Br_mode[::-1], aspect='auto', origin='lower', cmap='seismic',
        extent=extent_map, vmin=-vmax2, vmax=vmax2
    )
    ax2.set_title('Processed input magnetogram', fontsize=16)
    ax2.set_ylabel(ylabel, fontsize=14)
    ax2.set_xlabel('Longitude', fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    cbar2 = plt.colorbar(im2, ax=ax2, extend=_colorbar_extend(Br_mode, vmax2))
    cbar2.set_label('Br [G/2.2]', fontsize=14)
    cbar2.ax.tick_params(labelsize=12)

    if date is not None:
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path)
    plt.close()


def _as_bool(value: Any) -> bool:
    """Convert common boolean-like configuration values to ``bool``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _regular_phi_from_br(Br: np.ndarray) -> np.ndarray:
    """Return an endpoint-free regular longitude grid matching Br columns."""
    return np.linspace(0.0, 2.0 * np.pi, Br.shape[1], endpoint=False)


def _cell_widths(values: np.ndarray, fallback: float) -> np.ndarray:
    """Estimate cell widths from ordered cell centers."""
    if values.size < 2:
        return np.full(values.shape, fallback)
    return np.concatenate([np.diff(values), [values[-1] - values[-2]]])


def _pixel_area(theta: np.ndarray, phi: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Compute spherical pixel areas from colatitude and longitude centers."""
    nb_th, nb_phi = shape
    if len(theta) != nb_th:
        raise ValueError("theta length must match the first Br dimension.")
    if len(phi) != nb_phi:
        raise ValueError("phi length must match the second Br dimension.")
    dtheta = np.tile(_cell_widths(theta, np.pi), (nb_phi, 1)).T
    dphi = np.tile(_cell_widths(phi, 2.0 * np.pi), (nb_th, 1))
    Theta, _ = build_theta_phi(theta, phi)
    return np.abs(np.sin(Theta) * dtheta * dphi)


def _surface_mean_area(
    theta: np.ndarray,
    phi: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Compute spherical areas used by the surface-mean flux correction."""
    nb_th, nb_phi = shape
    if len(theta) != nb_th:
        raise ValueError("theta length must match the first Br dimension.")
    if len(phi) != nb_phi:
        raise ValueError("phi length must match the second Br dimension.")
    theta_edges = np.empty(len(theta) + 1)
    theta_edges[0] = 0.0
    theta_edges[-1] = np.pi
    theta_edges[1:-1] = 0.5 * (theta[:-1] + theta[1:])
    lat_weights = np.abs(np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:]))
    dphi = np.abs(_cell_widths(phi, 2.0 * np.pi))
    return lat_weights[:, None] * dphi[None, :]


def _flux_summary(Br: np.ndarray, pixel_area: np.ndarray) -> tuple[float, float, float, float]:
    """Return positive flux, negative flux, net flux, and imbalance percentage."""
    positive_mask = Br > 0
    negative_mask = Br < 0
    flux_positive = float(np.sum(Br[positive_mask] * pixel_area[positive_mask]))
    flux_negative = float(np.sum(Br[negative_mask] * pixel_area[negative_mask]))
    net_flux = flux_positive + flux_negative
    denominator = flux_positive - flux_negative
    imbalance_percent = net_flux / denominator * 100 if denominator != 0 else np.nan
    return flux_positive, flux_negative, net_flux, imbalance_percent


def _log_flux_summary(
    label: str,
    Br: np.ndarray,
    pixel_area: np.ndarray,
) -> None:
    """Log flux-balance diagnostics for a Br map and pixel-area grid."""
    flux_positive, flux_negative, net_flux, imbalance_percent = _flux_summary(
        Br,
        pixel_area,
    )
    logger.info(f"{label} flux positive: {flux_positive:.6e}")
    logger.info(f"{label} flux negative: {flux_negative:.6e}")
    logger.info(f"{label} net flux: {net_flux:.6e}")
    logger.info(f"{label} flux imbalance: {imbalance_percent:.6e} %")


def correct_net_flux(
    Br: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray | None = None,
    method: str = "surface_mean",
) -> np.ndarray:
    """Reduce or remove net magnetic flux from a magnetogram.

    ``surface_mean`` subtracts the surface-weighted mean Br. This is simple and
    preserves local contrast, but it shifts every pixel by the same offset.
    ``polarity_scaling`` rescales positive and negative polarities by opposite
    factors so the integrated positive and negative fluxes balance.

    Args:
        Br (np.ndarray): Radial magnetic field map.
        theta (np.ndarray): 1D colatitude grid in radians.
        phi (np.ndarray | None): 1D longitude grid in radians. Required for
            exact pixel areas with polarity balancing. If omitted, a regular
            endpoint-free longitude grid is assumed.
        method (str): Flux correction method. ``surface_mean`` subtracts the
            surface-weighted mean Br. ``polarity_scaling`` rescales positive
            and negative polarities by opposite multiplicative factors.

    Returns:
        np.ndarray: Flux-balanced Br map.
    """
    method = method.lower()
    phi = _regular_phi_from_br(Br) if phi is None else phi
    if method in {"surface_mean", "mean", "subtract_mean"}:
        pixel_area = _surface_mean_area(theta, phi, Br.shape)
        _log_flux_summary("Before surface_mean correction", Br, pixel_area)
        Br_corrected = _correct_net_flux_surface_mean(Br, theta)
        _log_flux_summary("After surface_mean correction", Br_corrected, pixel_area)
        return Br_corrected
    if method in {"polarity_scaling", "polarity", "multiplicative"}:
        pixel_area = _pixel_area(theta, phi, Br.shape)
        _log_flux_summary("Before polarity_scaling correction", Br, pixel_area)
        Br_corrected = _correct_net_flux_polarity_scaling(Br, pixel_area)
        _log_flux_summary("After polarity_scaling correction", Br_corrected, pixel_area)
        return Br_corrected
    raise ValueError(
        "flux correction method must be 'surface_mean' or 'polarity_scaling'."
    )


def _correct_net_flux_surface_mean(Br: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Subtract the surface-weighted mean Br from all pixels."""
    theta_edges = np.empty(len(theta) + 1)
    theta_edges[0] = 0.0
    theta_edges[-1] = np.pi
    theta_edges[1:-1] = 0.5 * (theta[:-1] + theta[1:])
    lat_weights = np.abs(np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:]))
    mean_br = np.sum(Br * lat_weights[:, None]) / (Br.shape[1] * np.sum(lat_weights))
    logger.info(f"Net flux correction: subtracting surface mean Br={mean_br:.6e}")
    return Br - mean_br


def _correct_net_flux_polarity_scaling(
    Br: np.ndarray,
    pixel_area: np.ndarray,
) -> np.ndarray:
    """Balance integrated flux by rescaling positive and negative polarities."""
    flux_positive, flux_negative, _, _ = _flux_summary(Br, pixel_area)
    if flux_negative == 0 or flux_positive == 0:
        logger.warning("Flux balancing skipped because one polarity is missing.")
        return Br

    balancing_factor = np.sqrt(flux_positive / abs(flux_negative))
    logger.info(f"Flux balancing factor: {balancing_factor:.6e}")
    return np.where(Br > 0, Br / balancing_factor, Br * balancing_factor)


def process_magnetogram_date(
    config: dict[str, Any],
    target_date: str | datetime,
    method_used: str = "sph",
    output_path_fig: str | None = None,
) -> dict[str, Any]:
    """Process one target time through the spherical-harmonic pipeline.

    The function downloads or reuses the requested magnetogram, optionally
    builds a temporal interpolation, computes the product effective time, logs
    the target/effective timing, optionally rotates the map to Stonyhurst,
    optionally balances net flux, projects/reconstructs the map with spherical
    harmonics, writes the COCONUT boundary file, and optionally saves a
    diagnostic figure.

    Args:
        config (dict[str, Any]): Processing configuration. Common keys are
            ``map_type``, ``output_dir``, ``download_dir``, ``lmax``, ``amp``,
            ``r_st``, ``adapt_map``, ``write_map``, ``show_map``,
            ``visu_type``, ``alpha``, ``interpolation_order``,
            ``interpolation``, ``rotate_to_stonyhurst``, ``flux_correct``,
            ``flux_correction_method``, and ``drms_email`` or ``jsoc_email``.
        target_date (str | datetime): Requested processing time.
        method_used (str): Method label used in output filenames.
        output_path_fig (str | None): Explicit diagnostic figure path. If
            omitted, the figure name is built from the effective time.

    Returns:
        dict[str, Any]: Processing metadata, including target ``date``,
        ``effective_date``, ``magnetogram_date``, output paths, selected local
        file or interpolation stencil, optional ``Br_linear``, spherical
        harmonic coefficients, and rotation angle.
    """
    map_type = normalize_map_type(config["map_type"])
    output_dir = config.get("output_dir", "../")
    download_dir = config.get("download_dir", output_dir)
    lmax = config.get("lmax", 20)
    amp = config.get("amp", 1)
    r_st = config.get("r_st", 1.0)
    adapt_map = config.get("adapt_map", 6) #between 1 and 11
    write_map = _as_bool(config.get("write_map", True))
    show_map = _as_bool(config.get("show_map", True))
    visu_type = config.get("visu_type", "sinlat")
    alpha = config.get("alpha", 0)
    interpolation_order = config.get("interpolation_order", config.get("Interp_order", 2))
    use_interpolation = _as_bool(
        config.get("interpolation", is_gong_temporal_map_type(map_type) or map_type == "ADAPT")
    )
    rotate_to_stonyhurst = _as_bool(config.get("rotate_to_stonyhurst", True))
    flux_correction_method = config.get("flux_correction_method", "surface_mean")
    drms_email = config.get("drms_email", config.get("jsoc_email"))

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
        Br, Theta, Phi = read_magnetogram(local_file, map_type, adapt_map)
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
    )

    if _as_bool(config.get("flux_correct", False)):
        Br = correct_net_flux(
            Br,
            Theta[:, 0],
            Phi[0, :],
            method=flux_correction_method,
        )

    Br_mode, coefbr = project_and_reconstruct(Br, Theta, Phi, lmax, amp, alpha)

    if write_map:
        write_bc_file(output_name, Br_mode, Theta[:, 0], Phi[0, :], r_st)

    if show_map:
        figure_path = output_path_fig or default_figure_path(output_dir, map_type, effective_date)
        plot_maps(
            Br,
            Br_mode,
            Theta[:, 0],
            Phi[0, :],
            map_type,
            visu_type,
            output_path=figure_path,
            date=figure_date,
        )
    else:
        figure_path = None

    br_mode_area = _surface_mean_area(Theta[:, 0], Phi[0, :], Br_mode.shape)
    (
        flux_positive,
        flux_negative,
        net_flux,
        imbalance_percent,
    ) = _flux_summary(Br_mode, br_mode_area)
    logger.info(f"Br_mode flux positive: {flux_positive:.6e}")
    logger.info(f"Br_mode flux negative: {flux_negative:.6e}")
    logger.info(f"Br_mode net flux: {net_flux:.6e}")
    logger.info(f"Br_mode flux imbalance: {imbalance_percent:.6e} %")
    logger.info(f"Br_mode max: {np.max(Br_mode):.6e}")
    logger.info(f"Br_mode mean: {np.mean(Br_mode):.6e}")
    logger.info(f"Br_mode min: {np.min(Br_mode):.6e}")

    return {
        "date": parse_iso_datetime(target_date),
        "effective_date": effective_date,
        "magnetogram_date": figure_date,
        "output_name": output_name,
        "local_file": local_file,
        "figure_path": figure_path,
        "selection": selection,
        "Br_linear": Br_linear,
        "coefbr": coefbr,
        "rotation_angle": rotation_angle,
    }


def process_config(config: dict[str, Any], method_used: str = "sph") -> list[dict[str, Any]]:
    """Process all target times described by one sph_filtering configuration.

    With only ``date`` set, a single target time is processed. With
    ``cadence_hours`` and ``total_hours``, the function builds a sequence of
    target times starting at ``date`` and processes each one independently.
    When no explicit ``output_path_fig`` is provided, each figure is named from
    the effective magnetogram time.

    Config keys:
        date: Initial ISO datetime.
        cadence_hours: Cadence in hours.
        total_hours: Total duration in hours.
        interpolation: Use four-map interpolation for temporal GONG variants
            and ADAPT.
        rotate_to_stonyhurst: Rotate longitude to the Stonyhurst frame. Defaults to True.
        flux_correct: Remove net magnetic flux if True.
        flux_correction_method: ``surface_mean`` or ``polarity_scaling``.
        output_path_fig: Optional figure file or directory.

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

    #to run a steady test

    base_output_dir = r"C:\Users\luisl\Desktop\testmagnetogram"
    label = "sph"
    output_dir = os.path.join(base_output_dir, label)
    figure_output_dir = os.path.join(base_output_dir, "images")

    config = {"date": "2026-07-01T06:17:00",
        "lmax": 20,
        "amp": 1,
        "write_map": True,
        "show_map": True,
        "visu_type": "sinlat",
        "alpha": 3 * 10 ** (-6),
        "rotate_to_stonyhurst": True,
        "interpolation": False,
        "interpolation_order": 2,
        "flux_correct": False,
        "flux_correction_method": "surface_mean", #surface_mean' or 'polarity_scaling'
        "map_type": "GONG_mrbqs",
        "adapt_map": 6,
        "output_dir": output_dir,
        "download_dir": output_dir,
        "output_path_fig": os.path.join(figure_output_dir, f"{label}.png"),
        "drms_email": "luis.linan@kuleuven.be"
        }

    process_config(config, method_used="sph")

    # for time-evolving , you can use "mrzqs", "mrbqs", "mrbqj" magnetogram and you need to add  : "cadence_hours": 3, "total_hours": 72, in the config.


    #Below an example that test every map type and ADAPT realization. The output is written to the test folder and figures are saved in the images subfolder.
    r"""
    base_output_dir = r"C:\Users\luisl\Desktop\testmagnetogram\test"
    figure_output_dir = os.path.join(base_output_dir, "images")
    common_config = {
        "date": "2020-01-20T01:17:00",
        "lmax": 10,
        "amp": 1,
        "write_map": True,
        "show_map": True,
        "visu_type": "sinlat",
        "alpha": 3 * 10 ** (-6),
        "rotate_to_stonyhurst": False,
        "interpolation": False,
        "flux_correct": False,
    }

    def test_config(label: str, map_type: str, **extra: Any) -> dict[str, Any]:
        output_dir = os.path.join(base_output_dir, label)
        return {
            **common_config,
            "map_type": map_type,
            "output_dir": output_dir,
            "download_dir": output_dir,
            "output_path_fig": os.path.join(figure_output_dir, f"{label}.png"),
            **extra,
        }

    configs = [
        test_config("GONG_mrzqs", "GONG_mrzqs"),
        test_config("GONG_mrbqs", "GONG_mrbqs"),
        test_config("GONG_mrbqj", "GONG_mrbqj"),
        test_config("GONG_mrmqs", "GONG_mrmqs"),
        test_config("GONG_mrnqs", "GONG_mrnqs"),
    ]
    configs.extend(
        test_config(f"ADAPT_{adapt_map:02d}", "ADAPT", adapt_map=adapt_map)
        for adapt_map in range(1, 12)
    )
    configs.extend(
        [
            test_config("HMI_polfil", "HMI_polfil"),
            test_config(
                "HMI_SYNC",
                "HMI_SYNC",
                drms_email="luis.linan@kuleuven.be",
            ),
            test_config("HMI_small", "HMI_small"),
        ]
    )


    for config in configs:
        try:
            process_config(config, method_used="sph")
        except Exception as exc:
            logger.warning(
                f'Failed to process {config["date"]} and {config["map_type"]}: {exc}'
            )
    """
