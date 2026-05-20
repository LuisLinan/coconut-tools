"""
Utilities for generating and downloading magnetogram input files for COCONUT simulations.

This module supports the generation of COCONUT-compatible magnetic boundary files
based on real solar data from various sources (e.g., WSO, GONG, ADAPT, HMI).

Main capabilities:
- Construct output filenames and download magnetogram data using Sunpy or custom rules,
- Build single-date or cadence-based time series configurations,
- Download four-map stencils and temporally interpolate GONG/ADAPT magnetograms,
- Handle spherical harmonic resolution (`lmax`),
- Provide helper functions to parse timestamps and build mesh grids.

Used to prepare magnetic input for running COCONUT coronal MHD simulations.
"""
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

import sunpy.coordinates.sun
import sunpy.util.net
import numpy as np
import requests
from bs4 import BeautifulSoup
from astropy.io import fits
from scipy import interpolate
from scipy import special as scisp
import matplotlib.pyplot as plt
from coconut_tools.logger_config import setup_logger

logger = setup_logger(__name__)


@dataclass(frozen=True)
class MagnetogramCandidate:
    """Remote magnetogram candidate with an observation timestamp."""

    name: str
    date: datetime
    remote_url: str


@dataclass(frozen=True)
class InterpolationSelection:
    """Four-map temporal interpolation stencil and its time weights."""

    before_previous: MagnetogramCandidate
    before: MagnetogramCandidate
    after: MagnetogramCandidate
    after_next: MagnetogramCandidate
    coef_before: float
    coef_after: float
    interval_seconds: float
    previous_interval_seconds: float
    next_interval_seconds: float
    target_date: datetime


def parse_iso_datetime(date: str | datetime) -> datetime:
    """Parse a date value into a datetime.

    Args:
        date (str | datetime): ISO datetime string or datetime object.

    Returns:
        datetime: Parsed datetime.
    """
    if isinstance(date, datetime):
        return date
    return datetime.fromisoformat(date)


def format_timestamp(date: str | datetime) -> str:
    """Format a datetime as YYYYMMDDHHMMSS for output filenames.

    Args:
        date (str | datetime): Date to format.

    Returns:
        str: Compact timestamp.
    """
    return parse_iso_datetime(date).strftime("%Y%m%d%H%M%S")


def build_output_name(
    date: str | datetime,
    map_type: str,
    output_dir: str,
    lmax: int | None,
    method_used: str = "sph",
) -> str:
    """Build the COCONUT boundary filename for a target date.

    Args:
        date (str | datetime): Target date.
        map_type (str): Map type.
        output_dir (str): Directory for the boundary file.
        lmax (int | None): Maximum spherical harmonic degree.
        method_used (str): Method label included in the output stem.

    Returns:
        str: Output filename ending with _YYYYMMDDHHMMSS.dat.
    """
    prefixes = {
        "WSO": "map_wso",
        "GONG": "map_gong",
        "ADAPT": "map_adapt",
        "HMI_polfil": "map_hmi_polfil",
        "HMI_small": "map_hmi_small",
    }
    prefix = prefixes.get(map_type)
    if prefix is None:
        raise ValueError(f"Unsupported map_type: {map_type}")

    filename = f"{prefix}_lmax{lmax}_{method_used}_{format_timestamp(date)}.dat"
    return os.path.join(output_dir, filename)


def _hrefs_containing(soup: BeautifulSoup, token: str) -> list[str]:
    """Extract hrefs containing a token from a parsed HTML page."""
    hrefs = []
    for node in soup.find_all("a"):
        href = node.get("href")
        if href and token in href:
            hrefs.append(href)
    return hrefs


@lru_cache(maxsize=128)
def fetch_remote_names(remote_dir: str, file_id: str) -> list[str]:
    """List remote filenames from a GONG-style directory page.

    Args:
        remote_dir (str): Remote directory URL.
        file_id (str): Filename token to keep.

    Returns:
        list[str]: Matching filenames.
    """
    try:
        response = requests.get(remote_dir)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning(f"Could not list remote directory {remote_dir}: {exc}")
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    return _hrefs_containing(soup, file_id)


def build_gong_remote_dir(date: datetime, file_id: str = "mrzqs") -> str:
    """Build a GONG QR magnetogram directory URL for one day.

    Args:
        date (datetime): Day to list.
        file_id (str): GONG file identifier.

    Returns:
        str: Remote directory URL.
    """
    return (
        f"https://gong.nso.edu/data/magmap/QR/{file_id[2:]}/"
        f"{date.year}{date.month:02d}/"
        f"{file_id}{str(date.year)[2:]}{date.month:02d}{date.day:02d}/"
    )


def list_gong_candidates(date: str | datetime) -> list[MagnetogramCandidate]:
    """List GONG candidates around a target date.

    The previous, current, and next UTC days are queried so a four-map stencil
    can cross midnight.

    Args:
        date (str | datetime): Target date.

    Returns:
        list[MagnetogramCandidate]: Sorted unique candidates.
    """
    date_datetime = parse_iso_datetime(date)
    file_id = "mrzqs"
    candidates = []
    for day_offset in (-1, 0, 1):
        current_day = date_datetime + timedelta(days=day_offset)
        remote_dir = build_gong_remote_dir(current_day, file_id)
        for name in fetch_remote_names(remote_dir, file_id):
            parsed_date = parse_date_from_filename(name, file_id + "%y%m%dt%H%M", "c")
            if parsed_date is not None:
                candidates.append(MagnetogramCandidate(name, parsed_date, remote_dir + name))
    return unique_sorted_candidates(candidates)


def list_adapt_candidates(date: str | datetime) -> list[MagnetogramCandidate]:
    """List ADAPT candidates around a target date.

    Args:
        date (str | datetime): Target date.

    Returns:
        list[MagnetogramCandidate]: Sorted unique candidates.
    """
    date_datetime = parse_iso_datetime(date)
    file_id = "adapt40311"
    years = {
        (date_datetime - timedelta(days=1)).year,
        date_datetime.year,
        (date_datetime + timedelta(days=1)).year,
    }
    candidates = []
    for year in sorted(years):
        remote_dir = f"https://gong.nso.edu/adapt/maps/gong/{year}/"
        for name in fetch_remote_names(remote_dir, file_id):
            try:
                parsed_date = datetime.strptime(name.split("_")[2], "%Y%m%d%H%M")
            except Exception as exc:
                logger.warning(f"Skipping file {name} due to parsing error: {exc}")
                continue
            candidates.append(MagnetogramCandidate(name, parsed_date, remote_dir + name))
    return unique_sorted_candidates(candidates)


def unique_sorted_candidates(
    candidates: list[MagnetogramCandidate],
) -> list[MagnetogramCandidate]:
    """Sort candidates and keep one entry per timestamp."""
    unique_by_date = {}
    for candidate in sorted(candidates, key=lambda item: item.date):
        unique_by_date.setdefault(candidate.date, candidate)
    return list(unique_by_date.values())


def list_remote_candidates(
    date: str | datetime,
    map_type: str,
) -> list[MagnetogramCandidate]:
    """List remote temporal candidates for a map type.

    Args:
        date (str | datetime): Target date.
        map_type (str): Map type.

    Returns:
        list[MagnetogramCandidate]: Sorted candidates.
    """
    if map_type == "GONG":
        return list_gong_candidates(date)
    if map_type == "ADAPT":
        return list_adapt_candidates(date)
    raise ValueError(f"Temporal candidate listing is not supported for {map_type}")


def select_nearest_candidate(
    candidates: list[MagnetogramCandidate],
    target_date: str | datetime,
) -> MagnetogramCandidate:
    """Select the candidate closest to a target date."""
    date_datetime = parse_iso_datetime(target_date)
    if not candidates:
        raise RuntimeError("No valid magnetogram file found on the remote server.")
    return min(candidates, key=lambda item: abs((item.date - date_datetime).total_seconds()))


def select_interpolation_stencil(
    candidates: list[MagnetogramCandidate],
    target_date: str | datetime,
) -> InterpolationSelection:
    """Select four magnetograms around a target date for Hermite interpolation.

    Args:
        candidates (list[MagnetogramCandidate]): Sorted remote candidates.
        target_date (str | datetime): Target interpolation date.

    Returns:
        InterpolationSelection: Four-map stencil and coefficients.
    """
    date_datetime = parse_iso_datetime(target_date)
    candidates = unique_sorted_candidates(candidates)
    for index in range(1, len(candidates) - 2):
        before = candidates[index]
        after = candidates[index + 1]
        if before.date <= date_datetime <= after.date:
            seconds_before = (date_datetime - before.date).total_seconds()
            seconds_after = (after.date - date_datetime).total_seconds()
            interval = seconds_before + seconds_after
            previous_interval = (before.date - candidates[index - 1].date).total_seconds()
            next_interval = (candidates[index + 2].date - after.date).total_seconds()
            if interval <= 0 or previous_interval <= 0 or next_interval <= 0:
                raise RuntimeError("Invalid temporal interpolation stencil.")
            return InterpolationSelection(
                before_previous=candidates[index - 1],
                before=before,
                after=after,
                after_next=candidates[index + 2],
                coef_before=seconds_after / interval,
                coef_after=seconds_before / interval,
                interval_seconds=interval,
                previous_interval_seconds=previous_interval,
                next_interval_seconds=next_interval,
                target_date=date_datetime,
            )
    raise RuntimeError(
        f"Could not find four magnetograms around {date_datetime.isoformat()}."
    )


def download_candidate(candidate: MagnetogramCandidate, output_dir: str) -> str:
    """Download one candidate if it is missing locally.

    Args:
        candidate (MagnetogramCandidate): Remote candidate.
        output_dir (str): Local download directory.

    Returns:
        str: Local file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    local_file = os.path.join(output_dir, candidate.name)
    if not os.path.exists(local_file):
        local_file = sunpy.util.net.download_file(
            candidate.remote_url,
            directory=output_dir,
            overwrite=True,
        )
        logger.info(f"Downloaded map: {local_file}")
    else:
        logger.info(f"Map already exists locally: {local_file}")
    return local_file


def download_interpolation_magnetograms(
    date: str | datetime,
    map_type: str,
    output_dir: str,
) -> tuple[list[str], InterpolationSelection]:
    """Download the four magnetograms needed for temporal interpolation.

    Args:
        date (str | datetime): Target interpolation date.
        map_type (str): Map type. Supported: GONG, ADAPT.
        output_dir (str): Local download directory.

    Returns:
        tuple[list[str], InterpolationSelection]: Local files in stencil order and selection.
    """
    candidates = list_remote_candidates(date, map_type)
    selection = select_interpolation_stencil(candidates, date)
    stencil = [
        selection.before_previous,
        selection.before,
        selection.after,
        selection.after_next,
    ]
    local_files = [download_candidate(candidate, output_dir) for candidate in stencil]
    return local_files, selection


def parse_date_from_filename(name, fmt, split_token):
    """Parse date from filename using a given format and delimiter.

    Args:
        name (str): Filename.
        fmt (str): Format string to parse the date.
        split_token (str): Token to split the filename before parsing.

    Returns:
        datetime: Parsed datetime object.
    """
    try:
        return datetime.strptime(name.split(split_token)[0], fmt)
    except Exception as e:
        logger.warning(f"Skipping file {name} due to parsing error: {e}")
        return None

def generate_output_and_map_names(
    date,
    map_type,
    output_dir,
    lmax,
    method_used="sph",
):
    """Generate output filename and download magnetogram file based on map type.

    Args:
        date (str): Date string in ISO format (e.g., '2020-12-07T15:00:00').
        map_type (str): Map type ('WSO', 'GONG', 'ADAPT', 'HMI').
        output_dir (str): Directory to store output file.
        lmax (int): Spherical harmonic maximum degree.
        method_used (str) : method used for filtering ( e.g NLD )

    Returns:
        tuple: (output_name (str), local_file (str))
    """
    date_datetime = parse_iso_datetime(date)

    cr_number = int(sunpy.coordinates.sun.carrington_rotation_number(date_datetime))
    logger.info(f"Carrington rotation number: {cr_number}")

    output_name = build_output_name(
        date_datetime,
        map_type,
        output_dir,
        lmax,
        method_used=method_used,
    )

    if map_type == 'WSO':
        map_name = f'WSO.{cr_number}.txt'
        remote_file = f"http://wso.stanford.edu/synoptic/{map_name}"

    elif map_type in ("GONG", "ADAPT"):
        candidate = select_nearest_candidate(
            list_remote_candidates(date_datetime, map_type),
            date_datetime,
        )
        map_name = candidate.name
        remote_file = candidate.remote_url

    elif map_type == 'HMI_small':
        map_name = f"hmi.Synoptic_Mr_small.{cr_number}.fits"
        remote_file = f"http://jsoc.stanford.edu/data/hmi/synoptic/{map_name}"
    elif map_type == 'HMI_polfil':
        map_name = f'hmi.Synoptic_Mr_polfil.{cr_number}.fits'
        remote_file = f"http://jsoc.stanford.edu/data/hmi/synoptic/{map_name}"


    else:
        raise ValueError(f"Unsupported map_type: {map_type}")

    local_file = os.path.join(output_dir, map_name)

    if not os.path.exists(local_file):
        os.makedirs(output_dir, exist_ok=True)
        local_file = sunpy.util.net.download_file(
            remote_file,
            directory=output_dir,
            overwrite=True,
        )
        logger.info(f"Downloaded map: {local_file}")
    else:
        logger.info(f"Map already exists locally: {local_file}")

    logger.info(f"Output file: {output_name}")
    logger.info(f"Downloaded map: {local_file}")

    return output_name, local_file

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


def build_regular_theta_phi(Br: np.ndarray, map_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Build the 1D theta/phi grids used by sph_filtering.

    Args:
        Br (np.ndarray): Magnetic field map.
        map_type (str): Map type.

    Returns:
        tuple[np.ndarray, np.ndarray]: Theta and phi vectors.
    """
    nb_th, nb_phi = Br.shape
    if map_type == "ADAPT":
        theta = np.linspace(0.0, np.pi, nb_th)
        phi = np.linspace(0.0, 2.0 * np.pi, nb_phi)
    else:
        sinlat = np.linspace(-1.0, 1.0, nb_th)
        theta = np.arcsin(sinlat) + np.pi / 2.0
        phi = np.linspace(0.0, 2.0 * np.pi, nb_phi)
    return theta, phi


def read_temporal_br_map(file_path: str, map_type: str, adapt_map: int = 0) -> np.ndarray:
    """Read one FITS magnetogram for temporal interpolation.

    Args:
        file_path (str): Local FITS file.
        map_type (str): Map type. Supported: GONG, ADAPT.
        adapt_map (int): ADAPT realization index.

    Returns:
        np.ndarray: Radial magnetic field map.
    """
    input_data = fits.getdata(file_path, ext=0)
    if map_type == "ADAPT":
        return np.nan_to_num(input_data[adapt_map, ::-1, :])
    if map_type == "GONG":
        Br = np.nan_to_num(input_data[::-1, :])
        return circular_shift_longitude(Br, extract_gong_longitude_shift(file_path))
    raise ValueError(f"Temporal interpolation is not supported for {map_type}")


def interpolate_br_maps(
    Br_maps: list[np.ndarray],
    selection: InterpolationSelection,
    interpolation_order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate four Br maps in time.

    Args:
        Br_maps (list[np.ndarray]): Maps ordered as before-previous, before, after, after-next.
        selection (InterpolationSelection): Time stencil and weights.
        interpolation_order (int): 1 for linear, 2 for cubic Hermite.

    Returns:
        tuple[np.ndarray, np.ndarray]: Interpolated Br and linear interpolation.
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
    """Read and temporally interpolate a four-map magnetogram stencil.

    Args:
        local_files (list[str]): Local files in stencil order.
        map_type (str): Map type. Supported: GONG, ADAPT.
        selection (InterpolationSelection): Time interpolation metadata.
        adapt_map (int): ADAPT realization index.
        interpolation_order (int): 1 for linear, 2 for cubic Hermite.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Br, Theta, Phi, Br_linear.
    """
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
    """Read magnetic field map file and extract Br, Theta, Phi grids.

    Args:
        file_path (str): Path to the magnetogram file.
        map_type (str): Type of the map ('WSO', 'GONG', 'ADAPT', 'HMI_small', 'HMI_pofil').
        adapt_map (int, optional): Index for ADAPT map. Defaults to 0.

    Returns:
        tuple: (Br_map, Theta, Phi) arrays.
    """
    logger.info('Reading file')

    if map_type == 'ADAPT':
        input_data = fits.getdata(file_path, ext=0)
        Br_map = input_data[adapt_map, ::-1, :]
        nb_th, nb_phi = Br_map.shape
        theta = np.linspace(0., np.pi, nb_th)
        phi = np.linspace(0., 2.0*np.pi, nb_phi)
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
        input_data = fits.getdata(file_path)
        Br_data = input_data[::-1, :]
        nb_th, nb_phi = Br_data.shape
        sinlat = np.linspace(-1., 1., nb_th)
        theta = np.arcsin(sinlat) + np.pi/2.
        phi = np.linspace(0., 2.0*np.pi, nb_phi)
        Theta, Phi = build_theta_phi(theta, phi)
        Br_map = np.nan_to_num(Br_data)

    logger.info("End of reading file")


    return Br_map, Theta, Phi

def project_and_reconstruct(Br, Theta, Phi, lmax, amp=1, alpha=0):
    """Project Br on spherical harmonics and reconstruct Br_mode.

    Args:
        Br (ndarray): Original radial field.
        Theta (ndarray): Colatitude grid.
        Phi (ndarray): Longitude grid.
        lmax (int): Maximum spherical harmonic degree.
        amp (float): Amplitude factor for reconstruction.
        alpha (float): Enhances the filtering for higher frequencies.

    Returns:
        tuple: (Br_mode, coefbr)
    """
    logger.info('Beginning of projection')
    nb_th, nb_phi = Br.shape
    nb_modes_tot = int((lmax + 1) * (lmax + 2) / 2 - 1)

    dtheta = np.tile(np.gradient(Theta[:, 0]), (nb_phi, 1)).T
    dphi = np.tile(np.gradient(Phi[0, :]), (nb_th, 1))

    coefbr = np.zeros(nb_modes_tot, dtype=complex)
    mod = 0
    for l in range(1, lmax + 1):
        logger.info(f"l = {l}")
        for m in range(0, l + 1):
            ylm = spherical_harmonic(m, l, Phi, Theta) / (1+alpha*l**2*(l+1)**2)
            integrand = Br * np.conj(ylm) * np.sin(Theta) * dtheta * dphi
            coefbr[mod] = np.sum(integrand)
            mod += 1
    logger.info('End of projection')

    logger.info('Reconstructing Br')
    Br_mode = np.zeros_like(Br)
    mod = 0
    for l in range(1, lmax + 1):
        logger.info(f"l = {l}")
        for m in range(0, l + 1):
            ylm = spherical_harmonic(m, l, Phi, Theta)
            Br_mode += np.real(coefbr[mod] * ylm)
            mod += 1

    Br_mode /= 2.2
    Br_mode *= amp
    logger.info('End of reconstructing Br')
    return Br_mode, coefbr

def write_bc_file(output_name, Br_mode, theta, phi, r_st):
    """Write boundary condition file.

    Args:
        output_name (str): Path to output file.
        Br_mode (ndarray): Reconstructed radial field.
        theta (ndarray): 1D theta grid.
        phi (ndarray): 1D phi grid.
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
    """Plot original and reconstructed Br maps.

    Args:
        Br (ndarray): Original radial magnetic field.
        Br_mode (ndarray): Reconstructed radial magnetic field.
        theta (ndarray): 1D theta grid.
        phi (ndarray): 1D phi grid.
        map_type (str): Type of the input map ('WSO', 'GONG', etc.).
        visu_type (str): Visualization style ('lat' or 'sinlat').
        lat_type (str, optional): Latitude type for WSO ('lat' or 'sinlat').
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

    if map_type == 'ADAPT':
        visu_type = 'lat'

    lat = 90. - 180. * theta / np.pi
    longi = 180. * phi / np.pi
    Lat, Long = np.meshgrid(lat, longi, indexing='ij')
    Sinlat = np.sin(np.radians(Lat))
    Sinlong = Long

    vmax1 = np.max(np.abs(Br))
    vmax1= 100
    vmax2 = np.max(np.abs(Br_mode))
    vmax2 = 30
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
    cbar1 = plt.colorbar(im1, ax=ax1)
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
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Br [G/2.2]', fontsize=14)
    cbar2.ax.tick_params(labelsize=12)

    if date is not None:
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path)
    plt.close()


def _as_bool(value: Any) -> bool:
    """Convert common config values to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def correct_net_flux(Br: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Remove the surface-weighted mean Br from a magnetogram.

    Args:
        Br (np.ndarray): Radial magnetic field map.
        theta (np.ndarray): 1D colatitude grid in radians.

    Returns:
        np.ndarray: Flux-balanced Br map.
    """
    theta_edges = np.empty(len(theta) + 1)
    theta_edges[0] = 0.0
    theta_edges[-1] = np.pi
    theta_edges[1:-1] = 0.5 * (theta[:-1] + theta[1:])
    lat_weights = np.abs(np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:]))
    mean_br = np.sum(Br * lat_weights[:, None]) / (Br.shape[1] * np.sum(lat_weights))
    logger.info(f"Net flux correction: subtracting surface mean Br={mean_br:.6e}")
    return Br - mean_br


def build_processing_dates(
    start_date: str | datetime,
    cadence_hours: float | None = None,
    total_hours: float | None = None,
) -> list[datetime]:
    """Build target dates from a start date, cadence, and total duration.

    Args:
        start_date (str | datetime): First target date.
        cadence_hours (float | None): Cadence in hours.
        total_hours (float | None): Total duration in hours. If None, one date is returned.

    Returns:
        list[datetime]: Target dates to process.
    """
    start = parse_iso_datetime(start_date)
    if total_hours is None or total_hours <= 0:
        return [start]
    if cadence_hours is None or cadence_hours <= 0:
        raise ValueError("cadence_hours must be positive when total_hours is set.")

    dates = []
    current = start
    end = start + timedelta(hours=float(total_hours))
    while current < end:
        dates.append(current)
        current += timedelta(hours=float(cadence_hours))
    return dates


def append_timestamp_to_path(path: str, date: str | datetime) -> str:
    """Append a compact timestamp before a path extension."""
    root, ext = os.path.splitext(path)
    return f"{root}_{format_timestamp(date)}{ext or '.png'}"


def default_figure_path(output_dir: str, map_type: str, date: str | datetime) -> str:
    """Build a default diagnostic figure path for one target date."""
    return os.path.join(output_dir, f"{map_type.lower()}_{format_timestamp(date)}.png")


def generate_output_and_interpolation_map_names(
    date,
    map_type,
    output_dir,
    lmax,
    method_used="sph",
    download_dir=None,
):
    """Generate output name and download four maps for temporal interpolation.

    Args:
        date (str | datetime): Target date.
        map_type (str): Map type. Supported: GONG, ADAPT.
        output_dir (str): Directory for the boundary file.
        lmax (int | None): Maximum spherical harmonic degree.
        method_used (str): Method label kept for compatibility.
        download_dir (str | None): Directory for downloaded FITS files.

    Returns:
        tuple[str, list[str], InterpolationSelection]: Output name, files, selection.
    """
    output_name = build_output_name(
        date,
        map_type,
        output_dir,
        lmax,
        method_used=method_used,
    )
    local_files, selection = download_interpolation_magnetograms(
        date,
        map_type,
        download_dir or output_dir,
    )
    logger.info(f"Output file: {output_name}")
    return output_name, local_files, selection


def process_magnetogram_date(
    config: dict[str, Any],
    target_date: str | datetime,
    method_used: str = "sph",
    output_path_fig: str | None = None,
) -> dict[str, Any]:
    """Process one target date from a sph_filtering configuration.

    Args:
        config (dict[str, Any]): Processing configuration.
        target_date (str | datetime): Date to process.
        method_used (str): Method label kept for compatibility.
        output_path_fig (str | None): Diagnostic figure path.

    Returns:
        dict[str, Any]: Paths and processing metadata.
    """
    map_type = config["map_type"]
    output_dir = config.get("output_dir", "../")
    download_dir = config.get("download_dir", output_dir)
    lmax = config.get("lmax", 20)
    amp = config.get("amp", 1)
    r_st = config.get("r_st", 1.0)
    adapt_map = config.get("adapt_map", 6)
    write_map = _as_bool(config.get("write_map", True))
    show_map = _as_bool(config.get("show_map", True))
    visu_type = config.get("visu_type", "sinlat")
    alpha = config.get("alpha", 0)
    interpolation_order = config.get("interpolation_order", config.get("Interp_order", 2))
    use_interpolation = _as_bool(config.get("interpolation", map_type in {"GONG", "ADAPT"}))

    if use_interpolation and map_type in {"GONG", "ADAPT"}:
        output_name, local_files, selection = generate_output_and_interpolation_map_names(
            target_date,
            map_type,
            output_dir,
            lmax,
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
            lmax,
            method_used,
        )
        Br, Theta, Phi = read_magnetogram(local_file, map_type, adapt_map)
        Br_linear = None
        selection = None

    if _as_bool(config.get("flux_correct", False)):
        Br = correct_net_flux(Br, Theta[:, 0])

    Br_mode, coefbr = project_and_reconstruct(Br, Theta, Phi, lmax, amp, alpha)

    if write_map:
        write_bc_file(output_name, Br_mode, Theta[:, 0], Phi[0, :], r_st)

    if show_map:
        figure_path = output_path_fig or default_figure_path(output_dir, map_type, target_date)
        plot_maps(
            Br,
            Br_mode,
            Theta[:, 0],
            Phi[0, :],
            map_type,
            visu_type,
            output_path=figure_path,
            date=target_date,
        )
    else:
        figure_path = None

    return {
        "date": parse_iso_datetime(target_date),
        "output_name": output_name,
        "local_file": local_file,
        "figure_path": figure_path,
        "selection": selection,
        "Br_linear": Br_linear,
        "coefbr": coefbr,
    }


def process_config(config: dict[str, Any], method_used: str = "sph") -> list[dict[str, Any]]:
    """Process one sph_filtering configuration.

    The configuration keeps the existing keys and adds optional multi-date
    processing.

    Config keys:
        date: Initial ISO datetime.
        cadence_hours: Cadence in hours.
        total_hours: Total duration in hours.
        interpolation: Use four-map interpolation for GONG/ADAPT.
        flux_correct: Remove net magnetic flux if True.

    Args:
        config (dict[str, Any]): Processing configuration.
        method_used (str): Method label kept for compatibility.

    Returns:
        list[dict[str, Any]]: Per-date processing results.
    """
    target_dates = build_processing_dates(
        config["date"],
        cadence_hours=config.get("cadence_hours", config.get("candence")),
        total_hours=config.get("total_hours"),
    )
    output_path_fig = config.get("output_path_fig")
    use_unique_figures = len(target_dates) > 1 and output_path_fig is not None
    results = []
    for target_date in target_dates:
        figure_path = (
            append_timestamp_to_path(output_path_fig, target_date)
            if use_unique_figures
            else output_path_fig
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
    # --- Example runs ---
    # Multi-date example:
    # {
    #     "date": "2025-10-09T18:00:00",
    #     "map_type": "GONG",
    #     "cadence_hours": 3,
    #     "total_hours": 72,
    #     "interpolation": True,
    #     "interpolation_order": 2,
    #     "flux_correct": True,
    #     "lmax": 20,
    #     "output_dir": "../",
    #     "download_dir": "../raw/",
    # }
    """
    configs = [
        {
            "date": '2020-12-07T15:00:00', "map_type": 'GONG',
            "lmax": 15, "amp": 1, "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "../", "output_path_fig": "../hmi_20201207.png",
            "alpha" : 0
        },
        {
            "date": '2022-03-11T12:00:00', "map_type": 'GONG',
            "lmax": 15, "amp": 0.8, "write_map": False, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "../", "output_path_fig": "../gong_20220311.png",
            "alpha": 0
        },
        {
            "date": '2023-08-15T00:00:00', "map_type": 'ADAPT', "adapt_map": 4,
            "lmax": 15, "amp": 1.2, "write_map": True, "show_map": False,
            "visu_type": "sinlat",
            "output_dir": "../", "output_path_fig": "../adapt_20230815.png",
            "alpha": 0
        },
        {
            "date": '2024-09-12T06:00:00', "map_type": 'WSO',
            "lmax": 15, "amp": 1.0, "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "../", "output_path_fig": "../wso_20240912.png",
            "alpha": 0
        }
    ]
    """
    """    
    configs = [
        {
            "date": '2013-03-13T12:00:00', "map_type": 'GONG',
            "lmax": 20, "amp": 1, "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/euhforia/magnetogram/alpha/", "output_path_fig": "E:/euhforia/magnetogram/alpha/GONG_20130313T120000.png",
            "alpha": 3*10**(-6)
        }]
    """

    configs = [
    {
        "date": '2017-09-04T18:00:00', "map_type": 'HMI_polfil',
        "lmax": 20, "amp": 1, "write_map": True, "show_map": True,
        "visu_type": "sinlat",
        "output_dir": "E:/euhforia/magnetogram/2017/old/", "output_path_fig": "E:/euhforia/magnetogram/2017/old/hmi_lmax20.png",
    },
        {
            "date": '2017-09-04T18:00:00', "map_type": 'HMI_polfil',
            "lmax": 10, "amp": 1, "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/euhforia/magnetogram/2017/old/",
            "output_path_fig": "E:/euhforia/magnetogram/2017/old/hmi_lmax10.png",
        },
        {
            "date": '2017-09-04T18:00:00', "map_type": 'HMI_polfil',
            "lmax": 15, "amp": 1, "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/euhforia/magnetogram/2017/old/",
            "output_path_fig": "E:/euhforia/magnetogram/2017/old/hmi_lmax15.png",
        },
        {
            "date": '2017-09-04T18:00:00', "map_type": 'HMI_polfil',
            "lmax": 50, "amp": 1, "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/euhforia/magnetogram/2017/alpha/",
            "output_path_fig": "E:/euhforia/magnetogram/2017/alpha/hmi_lmax50.png",
            "alpha": 3 * 10 ** (-6)
        },
    {
        "date": '2017-09-04T18:00:00', "map_type": 'GONG',
        "lmax": 20, "amp": 1, "write_map": True, "show_map": True,
        "visu_type": "sinlat",
        "output_dir": "E:/euhforia/magnetogram/2017/old/", "output_path_fig": "E:/euhforia/magnetogram/2017/old/gong_lmax20.png",
    },
        {
            "date": '2017-09-04T18:00:00', "map_type": 'GONG',
            "lmax": 10, "amp": 1, "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/euhforia/magnetogram/2017/old/",
            "output_path_fig": "E:/euhforia/magnetogram/2017/old/gong_lmax10.png",
        },
        {
            "date": '2017-09-04T18:00:00', "map_type": 'GONG',
            "lmax": 15, "amp": 1, "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/euhforia/magnetogram/2017/old/",
            "output_path_fig": "E:/euhforia/magnetogram/2017/old/gong_lmax15.png",
        },
        {
            "date": '2017-09-04T18:00:00', "map_type": 'GONG',
            "lmax": 50, "amp": 1, "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/euhforia/magnetogram/2017/alpha/",
            "output_path_fig": "E:/euhforia/magnetogram/2017/alpha/gong_lmax50.png",
            "alpha": 3 * 10 ** (-6)
        },
    ]

    config = {
        "date": "2012-07-13T00:00:00",
        "map_type": "GONG",
        "cadence_hours": 3,
        "total_hours": 240,
        "interpolation": True,
        "interpolation_order": 2,
        "flux_correct": False,
        "lmax": 20,
        "amp": 1,
        "write_map": True,
        "show_map": True,
        "visu_type": "sinlat",
        "output_dir": "E:/COCONUT/2012/alpha/",
        "output_path_fig": "E:/COCONUT/2012/alpha/image/",
        "download_dir": "E:/COCONUT/2012/raw/",
        "alpha": 3 * 10 ** (-6)
    }

    process_config(config, method_used="sph")


    for config in configs:
        try:
            process_config(config, method_used="sph")
        except Exception as exc:
            logger.warning(
                f'Failed to process {config["date"]} and {config["map_type"]}: {exc}'
            )
