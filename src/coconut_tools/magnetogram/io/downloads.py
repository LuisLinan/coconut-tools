"""Discover, select, and download supported magnetogram products."""
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import requests
import sunpy.coordinates.sun
import sunpy.util.net
from bs4 import BeautifulSoup

from coconut_tools.tools.logger_config import setup_logger

logger = setup_logger(__name__)

GONG_DEFAULT_FILE_ID = "mrzqs"
GONG_SYNCHRONIC_FILE_IDS = {"mrzqs", "mrbqs", "mrbqj"}
GONG_DIACHRONIC_FILE_IDS = {"mrmqs", "mrnqs"}
GONG_FILE_IDS = GONG_SYNCHRONIC_FILE_IDS | GONG_DIACHRONIC_FILE_IDS
HMI_SYNC_SERIES = "hmi.Mrdailysynframe_720s_nrt"
HMI_HOURLY_SERIES = "hmi.mrdailysynframe_polfil_720s_nrt"
HMI_HOURLY_SEGMENT = "Mr_polfil"
HMI_FDT_REMOTE_DIR = "https://gong.nso.edu/adapt/maps/hmi-fdtl/"
HMI_FDT_CARRINGTON_FILE_ID = "adapt40i11"


MAP_TYPE_ALIASES = {
    "custom": "custom",
    "wso": "WSO",
    "adapt": "ADAPT",
    "hmi_polfil": "HMI_polfil",
    "hmi_small": "HMI_small",
    "hmi_sync": "HMI_SYNC",
    "hmi_hourly": "HMI_hourly",
    "hmi_fdt": "HMI_fdt",
}


def normalize_map_type(map_type: str) -> str:
    """Return the canonical map type, accepting case-insensitive input."""
    if not isinstance(map_type, str):
        raise TypeError("map_type must be a string.")

    normalized_key = map_type.strip().casefold()
    if normalized_key == "gong":
        return "GONG"
    if normalized_key.startswith("gong_"):
        file_id = normalized_key.split("_", 1)[1]
        if file_id in GONG_FILE_IDS:
            return f"GONG_{file_id}"
        supported = ", ".join(f"GONG_{item}" for item in sorted(GONG_FILE_IDS))
        raise ValueError(f"Unsupported GONG map_type: {map_type}. Use one of {supported}.")

    canonical = MAP_TYPE_ALIASES.get(normalized_key)
    if canonical is not None:
        return canonical

    supported = ", ".join(
        [
            "WSO",
            "ADAPT",
            "HMI_polfil",
            "HMI_small",
            "HMI_SYNC",
            "HMI_hourly",
            "HMI_fdt",
            "custom",
        ]
        + ["GONG"]
        + [f"GONG_{item}" for item in sorted(GONG_FILE_IDS)]
    )
    raise ValueError(f"Unsupported map_type: {map_type}. Use one of {supported}.")


@dataclass(frozen=True)
class MagnetogramCandidate:
    """Remote magnetogram candidate with an observation timestamp."""

    name: str
    date: datetime
    remote_url: str | None


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
    """Parse a date value into a datetime."""
    if isinstance(date, datetime):
        return date
    return datetime.fromisoformat(date)


def format_timestamp(date: str | datetime) -> str:
    """Format a datetime as YYYYMMDDHHMMSS for output filenames."""
    return parse_iso_datetime(date).strftime("%Y%m%d%H%M%S")


def build_output_name(
    map_type: str,
    output_dir: str,
    method_used: str = "sph",
) -> str:
    """Build the COCONUT boundary filename for a target date."""
    map_type = normalize_map_type(map_type)
    if is_gong_map_type(map_type):
        file_id = gong_file_id_from_map_type(map_type)
        prefix = "map_gong" if map_type == "GONG" else f"map_gong_{file_id}"
        filename = f"{prefix}_{method_used}.dat"
        return os.path.join(output_dir, filename)

    prefixes = {
        "custom": "map_custom",
        "WSO": "map_wso",
        "ADAPT": "map_adapt",
        "HMI_polfil": "map_hmi_polfil",
        "HMI_small": "map_hmi_small",
        "HMI_SYNC": "map_hmi_sync",
        "HMI_hourly": "map_hmi_hourly",
        "HMI_fdt": "map_hmi_fdt",
    }
    prefix = prefixes.get(map_type)
    if prefix is None:
        raise ValueError(f"Unsupported map_type: {map_type}")

    filename = f"{prefix}_{method_used}.dat"
    return os.path.join(output_dir, filename)


def is_gong_map_type(map_type: str) -> bool:
    """Return True for legacy and file-id-specific GONG map types."""
    if not isinstance(map_type, str):
        return False
    normalized_key = map_type.strip().casefold()
    return normalized_key == "gong" or normalized_key.startswith("gong_")


def is_gong_diachronic_map_type(map_type: str) -> bool:
    """Return True for GONG diachronic map types."""
    if not is_gong_map_type(map_type):
        return False
    return gong_file_id_from_map_type(map_type) in GONG_DIACHRONIC_FILE_IDS


def is_gong_temporal_map_type(map_type: str) -> bool:
    """Return True for GONG map types supported by four-map interpolation."""
    return is_gong_map_type(map_type) and not is_gong_diachronic_map_type(map_type)


def gong_file_id_from_map_type(map_type: str) -> str:
    """Resolve the GONG file identifier encoded by a map type."""
    map_type = normalize_map_type(map_type)
    if map_type == "GONG":
        return GONG_DEFAULT_FILE_ID
    if not map_type.startswith("GONG_"):
        raise ValueError(f"Not a GONG map_type: {map_type}")

    file_id = map_type.split("_", 1)[1]
    if file_id not in GONG_FILE_IDS:
        supported = ", ".join(f"GONG_{item}" for item in sorted(GONG_FILE_IDS))
        raise ValueError(f"Unsupported GONG map_type: {map_type}. Use one of {supported}.")
    return file_id


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
    """List remote filenames from a GONG-style directory page."""
    try:
        response = requests.get(remote_dir)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning(f"Could not list remote directory {remote_dir}: {exc}")
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    return _hrefs_containing(soup, file_id)


def build_gong_remote_dir(date: datetime, file_id: str = "mrzqs") -> str:
    """Build a GONG QR magnetogram directory URL for one day."""
    return (
        f"https://gong.nso.edu/data/magmap/QR/{file_id[2:]}/"
        f"{date.year}{date.month:02d}/"
        f"{file_id}{str(date.year)[2:]}{date.month:02d}{date.day:02d}/"
    )


def list_gong_candidates(
    date: str | datetime,
    file_id: str = GONG_DEFAULT_FILE_ID,
) -> list[MagnetogramCandidate]:
    """List GONG candidates around a target date."""
    date_datetime = parse_iso_datetime(date)
    candidates = []
    for day_offset in (-1, 0, 1):
        current_day = date_datetime + timedelta(days=day_offset)
        remote_dir = build_gong_remote_dir(current_day, file_id)
        for name in fetch_remote_names(remote_dir, file_id):
            parsed_date = parse_date_from_filename(name, file_id + "%y%m%dt%H%M", "c")
            if parsed_date is not None:
                candidates.append(MagnetogramCandidate(name, parsed_date, remote_dir + name))
    return unique_sorted_candidates(candidates)


def _parse_gong_file_date(name: str, file_id: str) -> datetime | None:
    """Parse the observation date from a GONG filename."""
    try:
        return datetime.strptime(name.split("c")[0], file_id + "%y%m%dt%H%M")
    except ValueError as exc:
        logger.warning(f"Skipping file {name} due to parsing error: {exc}")
        return None


def list_gong_diachronic_candidates(
    date: str | datetime,
    file_id: str,
) -> list[MagnetogramCandidate]:
    """List GONG diachronic files around the requested month.

    Integral GONG maps are published once per Carrington rotation.  Looking
    only in the target calendar month can therefore select the *next* map even
    when the previous rotation is closer to the requested date.  Search the
    previous, current, and next month and let the common nearest-candidate
    selector compare the physical dates encoded in the filenames.
    """
    date_datetime = parse_iso_datetime(date)
    candidates = []
    month_index = date_datetime.year * 12 + date_datetime.month - 1
    for month_offset in (-1, 0, 1):
        current_index = month_index + month_offset
        year, zero_based_month = divmod(current_index, 12)
        remote_base = (
            f"https://gong.nso.edu/data/magmap/QR/{file_id[2:]}/"
            f"{year}{zero_based_month + 1:02d}/"
        )
        page_text = requests.get(remote_base).text
        soup = BeautifulSoup(page_text, "html.parser")
        folder_names = [
            node.get("href").strip("/")
            for node in soup.find_all("a")
            if node.get("href") is not None
            and node.get("href").startswith(file_id)
        ]

        for map_folder in folder_names:
            remote_dir = remote_base + map_folder + "/"
            page_text = requests.get(remote_dir).text
            soup = BeautifulSoup(page_text, "html.parser")
            file_names = [
                node.get("href")
                for node in soup.find_all("a")
                if node.get("href") is not None and file_id in node.get("href")
            ]
            for file_name in file_names:
                file_date = _parse_gong_file_date(file_name, file_id)
                if file_date is not None:
                    candidates.append(
                        MagnetogramCandidate(
                            file_name,
                            file_date,
                            remote_dir + file_name,
                        )
                    )
    return unique_sorted_candidates(candidates)


def list_adapt_candidates(date: str | datetime) -> list[MagnetogramCandidate]:
    """List ADAPT candidates around a target date."""
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


def parse_hmi_fdt_filename_date(name: str) -> datetime | None:
    """Parse an HMI-FDT Carrington-map timestamp from its ADAPT filename."""
    basename = os.path.basename(name)
    match = re.fullmatch(
        rf"{HMI_FDT_CARRINGTON_FILE_ID}_[^_]+_(?P<date>\d{{12}})_"
        r"[^_]+\.(?:fts|fits)(?:\.gz)?",
        basename,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None

    try:
        return datetime.strptime(match.group("date"), "%Y%m%d%H%M")
    except ValueError as exc:
        logger.warning(
            "Could not parse HMI_fdt observation date from %s: %s",
            basename,
            exc,
        )
        return None


def list_hmi_fdt_candidates(date: str | datetime) -> list[MagnetogramCandidate]:
    """List HMI-FDT maps on the fixed Carrington grid.

    The HMI-FDTL directory publishes both ``adapt40i11`` Carrington-fixed maps
    and ``adapt41i11`` central-meridian-centered maps for each timestamp.  Only
    the former are candidates so every temporal stencil shares one longitude
    frame before interpolation.
    """
    parse_iso_datetime(date)
    candidates = []
    for name in fetch_remote_names(
        HMI_FDT_REMOTE_DIR,
        HMI_FDT_CARRINGTON_FILE_ID,
    ):
        parsed_date = parse_hmi_fdt_filename_date(name)
        if parsed_date is not None:
            candidates.append(
                MagnetogramCandidate(
                    name,
                    parsed_date,
                    HMI_FDT_REMOTE_DIR + name,
                )
            )
    return unique_sorted_candidates(candidates)


def parse_hmi_hourly_filename_date(name: str) -> datetime | None:
    """Parse the observation timestamp from an HMI hourly filename."""
    basename = os.path.basename(name)
    match = re.search(
        r"hmi\.synoptic_hourly_(?P<date>\d{8})_(?P<time>\d{6})\.fits(?:\.gz)?$",
        basename,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None

    try:
        return datetime.strptime(
            match.group("date") + match.group("time"),
            "%Y%m%d%H%M%S",
        )
    except ValueError as exc:
        logger.warning(
            "Could not parse HMI_hourly observation date from %s: %s",
            basename,
            exc,
        )
        return None


def list_hmi_candidates(date: str | datetime) -> list[MagnetogramCandidate]:
    """List HMI hourly candidates within one day of a target date."""
    import drms

    date_datetime = parse_iso_datetime(date)
    start = (date_datetime - timedelta(days=1)).strftime("%Y.%m.%d_%H:%M:%S_TAI")
    end = (date_datetime + timedelta(days=1)).strftime("%Y.%m.%d_%H:%M:%S_TAI")
    recordset = f"{HMI_HOURLY_SERIES}[{start}-{end}]"

    logger.info("Searching available HMI_hourly records: %s", recordset)
    keys, segments = drms.Client().query(
        recordset,
        key="T_REC",
        seg=HMI_HOURLY_SEGMENT,
    )

    candidates = []
    for index in range(len(keys)):
        t_rec = str(keys["T_REC"].iloc[index])
        try:
            record_date = parse_jsoc_trec(t_rec)
        except ValueError as exc:
            logger.warning("Skipping HMI_hourly record %s: %s", t_rec, exc)
            continue

        name = f"hmi.synoptic_hourly_{record_date.strftime('%Y%m%d_%H%M%S')}.fits"
        candidate_date = parse_hmi_hourly_filename_date(name)
        if candidate_date is None:
            continue

        segment_path = str(segments[HMI_HOURLY_SEGMENT].iloc[index]).strip()
        if is_jsoc_segment_offline(segment_path):
            # Keep the timestamp as a valid candidate.  The download stage can
            # ask JSOC/SUMS to stage the archived segment through DRMS.
            remote_url = None
        elif segment_path.startswith(("http://", "https://")):
            remote_url = segment_path
        else:
            remote_url = f"http://jsoc.stanford.edu{segment_path}"
        candidates.append(MagnetogramCandidate(name, candidate_date, remote_url))

    return unique_sorted_candidates(candidates)


def is_jsoc_segment_offline(segment_path: str | None) -> bool:
    """Return whether JSOC reported metadata without an online segment path."""
    if segment_path is None:
        return True
    normalized = str(segment_path).strip().casefold()
    return normalized in {
        "",
        "none",
        "nan",
        "nodatafile",
        "nodatadirectory",
    }


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
    """List remote temporal candidates for a map type."""
    map_type = normalize_map_type(map_type)
    if is_gong_map_type(map_type):
        file_id = gong_file_id_from_map_type(map_type)
        if file_id in GONG_DIACHRONIC_FILE_IDS:
            return list_gong_diachronic_candidates(date, file_id)
        return list_gong_candidates(date, file_id)
    if map_type == "ADAPT":
        return list_adapt_candidates(date)
    if map_type == "HMI_hourly":
        return list_hmi_candidates(date)
    if map_type == "HMI_fdt":
        return list_hmi_fdt_candidates(date)
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
    """Select four magnetograms around a target date for Hermite interpolation."""
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
    """Download one candidate if it is missing locally."""
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


def download_hmi_hourly_magnetogram(
    candidate: MagnetogramCandidate,
    output_dir: str,
    drms_email: str | None = None,
) -> tuple[str, str]:
    """Download an HMI hourly map and attach its complete JSOC metadata.

    JSOC returns ``NoDataDirectory`` instead of a URL when an archived segment
    is not currently staged on disk.  In that case, use a DRMS export request
    to stage the exact segment rather than treating the sentinel as a hostname.
    """
    import urllib.error
    import urllib.request

    import drms
    import numpy as np
    from astropy.io import fits

    os.makedirs(output_dir, exist_ok=True)
    local_file = os.path.join(output_dir, candidate.name)
    if os.path.exists(local_file):
        logger.info("Map already exists locally: %s", local_file)
        return local_file, candidate.name

    target_time = parse_hmi_hourly_filename_date(candidate.name)
    if target_time is None:
        raise ValueError(
            f"Cannot determine the HMI_hourly target time from filename: {candidate.name}"
        )

    t_rec = target_time.strftime("%Y.%m.%d_%H:%M:%S_TAI")
    recordset = f"{HMI_HOURLY_SERIES}[{t_rec}]"
    client = drms.Client(email=drms_email) if drms_email else drms.Client()

    data = None
    direct_error = None
    if candidate.remote_url is not None:
        logger.info("Direct JSOC download for %s", candidate.name)
        logger.debug("HMI_hourly segment URL: %s", candidate.remote_url)
        temp_file = local_file + ".temp"
        try:
            urllib.request.urlretrieve(candidate.remote_url, temp_file)
            with fits.open(temp_file) as hdul:
                image_hdu = next(
                    (hdu for hdu in hdul if hdu.data is not None),
                    None,
                )
                if image_hdu is None:
                    raise RuntimeError(
                        f"No image data found in downloaded FITS file: {temp_file}"
                    )
                data = np.nan_to_num(image_hdu.data)
        except (OSError, urllib.error.URLError) as exc:
            direct_error = exc
            logger.warning(
                "Direct download failed for %s (%s); trying a JSOC DRMS export.",
                candidate.remote_url,
                exc,
            )
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    if data is None:
        if not drms_email:
            reason = (
                "JSOC reports NoDataDirectory for this archived segment"
                if candidate.remote_url is None
                else f"the direct download failed: {direct_error}"
            )
            raise ValueError(
                f"Cannot download HMI_hourly record {t_rec}: {reason}. "
                "Set config['drms_email'] to a registered JSOC email so DRMS "
                "can stage the archived FITS segment."
            )

        export_recordset = f"{recordset}{{{HMI_HOURLY_SEGMENT}}}"
        logger.info("Staging archived JSOC segment: %s", export_recordset)
        try:
            export = client.export(export_recordset, protocol="fits")
            with tempfile.TemporaryDirectory(prefix="coconut_hmi_hourly_") as temp_dir:
                downloaded_files = export.download(temp_dir)
                downloads = downloaded_files.download
                exported_file = (
                    downloads.iloc[0]
                    if hasattr(downloads, "iloc")
                    else downloads[0]
                )
                exported_file = str(exported_file)
                if not exported_file.lower().endswith((".fits", ".fits.gz")):
                    raise RuntimeError(
                        f"JSOC did not return a FITS file: {exported_file}"
                    )
                with fits.open(exported_file) as hdul:
                    image_hdu = next(
                        (hdu for hdu in hdul if hdu.data is not None),
                        None,
                    )
                    if image_hdu is None:
                        raise RuntimeError(
                            "No image data found in staged HMI_hourly FITS: "
                            f"{exported_file}"
                        )
                    data = np.nan_to_num(image_hdu.data)
        except Exception as exc:
            raise RuntimeError(
                f"JSOC found HMI_hourly record {t_rec}, but its FITS segment "
                "could not be staged or downloaded. The NRT file may no longer "
                "be recoverable from the online archive; JSOC recommends "
                "contacting jsoc@sun.stanford.edu for unavailable files."
            ) from exc

    keys = client.query(recordset, key=drms.JsocInfoConstants.all)
    if keys.empty:
        raise RuntimeError(f"No JSOC metadata found for HMI_hourly record {t_rec}.")

    map_header_dict = dict(keys.iloc[0])
    if "T_OBS" in map_header_dict:
        try:
            map_header_dict["DATE-OBS"] = parse_jsoc_trec(
                str(map_header_dict["T_OBS"])
            ).strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError as exc:
            logger.warning("Could not convert HMI_hourly T_OBS: %s", exc)

    map_header_dict.update(
        {
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "CDELT2": 0.07957747154594767,
            "BUNIT": "gauss",
            "CRDER1": 0,
            "CRDER2": 0,
        }
    )
    for old_key, new_key in (
        ("T_REC_epoch", "TRECEPOC"),
        ("T_REC_step", "TRECSTEP"),
        ("T_REC_unit", "TRECUNIT"),
    ):
        if old_key in map_header_dict:
            map_header_dict[new_key] = map_header_dict.pop(old_key)
    for key in ("CSYSER1", "CSYSER2", "HGLN_OBS", "BLD_VERS"):
        map_header_dict.pop(key, None)

    header = fits.Header()
    for key, value in map_header_dict.items():
        try:
            header[key] = value
        except (TypeError, ValueError) as exc:
            logger.debug("Skipping incompatible FITS header card %s: %s", key, exc)

    new_hdu = fits.CompImageHDU(data=data, header=header)
    new_hdu.writeto(local_file, overwrite=True)
    logger.info("Downloaded map: %s", local_file)
    return local_file, candidate.name


def download_interpolation_magnetograms(
    date: str | datetime,
    map_type: str,
    output_dir: str,
    drms_email: str | None = None,
) -> tuple[list[str], InterpolationSelection]:
    """Download the four magnetograms needed for temporal interpolation."""
    map_type = normalize_map_type(map_type)
    if is_gong_diachronic_map_type(map_type):
        raise ValueError("Temporal interpolation is not supported for GONG diachronic maps.")
    candidates = list_remote_candidates(date, map_type)
    selection = select_interpolation_stencil(candidates, date)
    stencil = [
        selection.before_previous,
        selection.before,
        selection.after,
        selection.after_next,
    ]
    if map_type == "HMI_hourly":
        if drms_email:
            local_files = [
                download_hmi_hourly_magnetogram(
                    candidate,
                    output_dir,
                    drms_email=drms_email,
                )[0]
                for candidate in stencil
            ]
        else:
            local_files = [
                download_hmi_hourly_magnetogram(candidate, output_dir)[0]
                for candidate in stencil
            ]
    else:
        local_files = [
            download_candidate(candidate, output_dir)
            for candidate in stencil
        ]
    return local_files, selection


def parse_date_from_filename(name, fmt, split_token):
    """Parse date from filename using a given format and delimiter."""
    try:
        return datetime.strptime(name.split(split_token)[0], fmt)
    except Exception as exc:
        logger.warning(f"Skipping file {name} due to parsing error: {exc}")
        return None


def parse_hmi_sync_filename_date(name: str) -> datetime | None:
    """Parse the timestamp from an HMI_SYNC JSOC FITS filename."""
    basename = os.path.basename(name)
    match = re.search(r"(?P<date>\d{8})_(?P<time>\d{6})(?:_TAI)?", basename)
    if match is None:
        return None

    try:
        return datetime.strptime(
            match.group("date") + match.group("time"),
            "%Y%m%d%H%M%S",
        )
    except ValueError as exc:
        logger.warning("Could not parse HMI_SYNC observation date from %s: %s", basename, exc)
        return None


def magnetogram_display_date(
    file_path: str,
    map_type: str,
    target_date: str | datetime,
    interpolated: bool = False,
) -> datetime:
    """Return the date that should be displayed for a processed magnetogram."""
    return magnetogram_effective_date(
        file_path,
        map_type,
        target_date,
        interpolated=interpolated,
    )


def magnetogram_effective_date(
    file_path: str,
    map_type: str,
    target_date: str | datetime,
    interpolated: bool = False,
) -> datetime:
    """Return the timestamp represented by a processed magnetogram.

    Interpolated maps represent the requested target time. Non-interpolated
    GONG, ADAPT, HMI_SYNC, HMI_hourly, and HMI_fdt maps represent the
    observation time encoded in their filenames. HMI small/polar-filled and
    WSO products use the requested target time by convention.
    """
    map_type = normalize_map_type(map_type)
    target = parse_iso_datetime(target_date)
    if interpolated:
        return target
    if map_type == "custom":
        logger.info(
            "Custom magnetogram display and rotation time comes from the "
            "configured date: %s.",
            target.isoformat(),
        )
        return target
    if map_type == "HMI_SYNC":
        parsed_date = parse_hmi_sync_filename_date(file_path)
        if parsed_date is not None:
            return parsed_date
    if map_type == "HMI_hourly":
        parsed_date = parse_hmi_hourly_filename_date(file_path)
        if parsed_date is not None:
            return parsed_date
    if map_type == "HMI_fdt":
        parsed_date = parse_hmi_fdt_filename_date(file_path)
        if parsed_date is not None:
            return parsed_date
    if map_type in {"HMI_small", "HMI_polfil"}:
        return target

    name = os.path.basename(file_path)
    if is_gong_map_type(map_type):
        file_id = gong_file_id_from_map_type(map_type)
        try:
            return datetime.strptime(name.split("c")[0], file_id + "%y%m%dt%H%M")
        except ValueError as exc:
            logger.warning("Could not parse GONG observation date from %s: %s", name, exc)
    elif map_type == "ADAPT":
        try:
            return datetime.strptime(name.split("_")[2], "%Y%m%d%H%M")
        except (IndexError, ValueError) as exc:
            logger.warning("Could not parse ADAPT observation date from %s: %s", name, exc)
    elif map_type.lower() == "wso":
        logger.info(
            "WSO filenames contain a Carrington rotation but no precise observation "
            "time; using the target date for the figure."
        )
        return target

    logger.warning(
        "Could not determine the observation date from %s; using target date %s.",
        name,
        target.isoformat(),
    )
    return target

def parse_jsoc_trec(t_rec: str) -> datetime:
    """Parse a JSOC T_REC string like 2026.07.01_00:24:00_TAI."""
    value = str(t_rec).replace("_TAI", "")

    for fmt in ("%Y.%m.%d_%H:%M:%S.%f", "%Y.%m.%d_%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass

    raise ValueError(f"Cannot parse JSOC T_REC: {t_rec}")

def find_nearest_hmi_sync_record(
    client,
    date_datetime: datetime,
    half_window_hours: int = 12,
) -> tuple[str, datetime]:
    """Find the nearest available HMI_SYNC record around a target datetime."""
    start_datetime = date_datetime - timedelta(hours=half_window_hours)
    duration_hours = 2 * half_window_hours

    start_jsoc = start_datetime.strftime("%Y.%m.%d_%H:%M:%S_TAI")

    recordset = f"{HMI_SYNC_SERIES}[{start_jsoc}/{duration_hours}h]"

    logger.info(f"Searching available HMI_SYNC records: {recordset}")

    records = client.query(recordset, key="T_REC")

    if records.empty:
        raise RuntimeError(
            f"No HMI_SYNC records found around {date_datetime} "
            f"within +/- {half_window_hours} hours."
        )

    candidates = []

    for _, row in records.iterrows():
        t_rec = str(row["T_REC"])
        candidate_datetime = parse_jsoc_trec(t_rec)
        candidates.append((t_rec, candidate_datetime))

    nearest_t_rec, nearest_datetime = min(
        candidates,
        key=lambda item: abs(item[1] - date_datetime),
    )

    logger.info(f"Requested date: {date_datetime}")
    logger.info(f"Nearest available HMI_SYNC record: {nearest_t_rec}")

    return nearest_t_rec, nearest_datetime

def parse_jsoc_trec(t_rec: str) -> datetime:
    """Parse a JSOC T_REC string like 2025.11.22_08:24:00_TAI."""
    value = str(t_rec).replace("_TAI", "")

    for fmt in ("%Y.%m.%d_%H:%M:%S.%f", "%Y.%m.%d_%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    raise ValueError(f"Cannot parse JSOC T_REC: {t_rec}")


def find_hmi_sync_candidates(
    client,
    date_datetime: datetime,
    half_window_hours: int = 12,
) -> list[tuple[str, datetime]]:
    """Return available HMI_SYNC metadata records sorted by distance to the target date."""
    start_datetime = date_datetime - timedelta(hours=half_window_hours)
    duration_hours = 2 * half_window_hours

    start_jsoc = start_datetime.strftime("%Y.%m.%d_%H:%M:%S_TAI")
    recordset = f"{HMI_SYNC_SERIES}[{start_jsoc}/{duration_hours}h]"

    logger.info(f"Searching available HMI_SYNC records: {recordset}")

    records = client.query(recordset, key="T_REC")

    if records.empty:
        return []

    candidates = []

    for _, row in records.iterrows():
        t_rec = str(row["T_REC"])
        candidate_datetime = parse_jsoc_trec(t_rec)
        candidates.append((t_rec, candidate_datetime))

    candidates.sort(key=lambda item: abs(item[1] - date_datetime))

    return candidates


def is_jsoc_unavailable_error(exc: Exception) -> bool:
    """Return True if JSOC found the record but the data file is unavailable."""
    msg = str(exc).lower()

    unavailable_patterns = [
        "status=25",
        "offline",
        "no fits files were exported",
        "requested fits files no longer exist",
        "nodatafile",
        "nodatadirectory",
    ]

    return any(pattern in msg for pattern in unavailable_patterns)


def ensure_hmi_sync_wcs(
    client,
    t_rec: str,
    local_file: str,
) -> str:
    """Attach authoritative JSOC longitude WCS metadata to an HMI_SYNC FITS."""
    from astropy.io import fits

    series = f"{HMI_SYNC_SERIES}[{t_rec}]"
    wcs_keys = ("CRVAL1", "CRPIX1", "CDELT1", "CTYPE1", "CUNIT1")
    metadata = client.query(series, key=",".join(wcs_keys))
    if metadata.empty:
        raise RuntimeError(f"No JSOC WCS metadata found for HMI_SYNC record {t_rec}.")

    values = metadata.iloc[0]
    for key in ("CRVAL1", "CRPIX1", "CDELT1"):
        try:
            value = float(values[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid JSOC {key} metadata for HMI_SYNC record {t_rec}."
            ) from exc
        if not math.isfinite(value) or (key == "CDELT1" and value == 0.0):
            raise RuntimeError(
                f"Invalid JSOC {key} metadata for HMI_SYNC record {t_rec}."
            )

    try:
        with fits.open(local_file, mode="update") as hdul:
            image_hdu = next(
                (
                    hdu
                    for hdu in hdul
                    if hdu.data is not None and hdu.data.ndim >= 2
                ),
                None,
            )
            if image_hdu is None:
                raise RuntimeError(
                    f"No image data found in downloaded HMI_SYNC FITS: {local_file}"
                )
            for key in wcs_keys:
                value = values.get(key)
                if value is not None:
                    image_hdu.header[key] = value
            hdul.flush()
    except OSError as exc:
        raise RuntimeError(
            f"Could not update HMI_SYNC FITS metadata: {local_file}"
        ) from exc

    logger.info(
        "Attached HMI_SYNC longitude WCS for %s: CRVAL1=%s, CRPIX1=%s, CDELT1=%s",
        t_rec,
        values["CRVAL1"],
        values["CRPIX1"],
        values["CDELT1"],
    )
    return local_file


def try_download_hmi_sync_record(
    client,
    t_rec: str,
    output_dir: str,
) -> tuple[str, str] | None:
    """Try to download one HMI_SYNC record. Return None if the data file is unavailable."""
    series = f"{HMI_SYNC_SERIES}[{t_rec}]"

    logger.info(f"Requesting JSOC series: {series}")

    try:
        export = client.export(series, protocol="fits")
        downloaded_files = export.download(output_dir)

        downloads = downloaded_files.download
        local_file = downloads.iloc[0] if hasattr(downloads, "iloc") else downloads[0]
        local_file = str(local_file)

        if not local_file.lower().endswith((".fits", ".fits.gz")):
            raise RuntimeError(f"JSOC did not return a FITS file: {local_file}")

        local_file = ensure_hmi_sync_wcs(client, t_rec, local_file)
        map_name = Path(local_file).name

        logger.info(f"Downloaded map: {local_file}")

        return local_file, map_name

    except Exception as exc:
        if is_jsoc_unavailable_error(exc):
            logger.warning(f"HMI_SYNC record found but data file unavailable: {t_rec}")
            return None

        raise


def download_hmi_sync_magnetogram(
    date_datetime: datetime,
    output_dir: str,
    drms_email: str | None,
) -> tuple[str, str]:
    """Download the nearest downloadable HMI_SYNC magnetogram through JSOC DRMS."""
    if not drms_email:
        raise ValueError("HMI_SYNC requires a DRMS email in config['drms_email'].")

    import drms

    os.makedirs(output_dir, exist_ok=True)

    client = drms.Client(email=drms_email)

    tried_records = set()

    for half_window_hours in (12, 24, 48, 72):
        candidates = find_hmi_sync_candidates(
            client=client,
            date_datetime=date_datetime,
            half_window_hours=half_window_hours,
        )

        if not candidates:
            logger.warning(
                f"No HMI_SYNC metadata records found within +/- {half_window_hours} h "
                f"around {date_datetime}."
            )
            continue

        logger.info(f"Requested date: {date_datetime}")
        logger.info(
            f"Nearest HMI_SYNC metadata record: {candidates[0][0]} "
            f"at {candidates[0][1]}"
        )

        for t_rec, candidate_datetime in candidates:
            if t_rec in tried_records:
                continue

            tried_records.add(t_rec)

            delta_minutes = abs(candidate_datetime - date_datetime).total_seconds() / 60.0

            logger.info(
                f"Trying HMI_SYNC record {t_rec} "
                f"with time difference {delta_minutes:.1f} min"
            )

            result = try_download_hmi_sync_record(
                client=client,
                t_rec=t_rec,
                output_dir=output_dir,
            )

            if result is not None:
                return result

    raise RuntimeError(
        f"No downloadable HMI_SYNC magnetogram found around {date_datetime}. "
        f"Metadata records exist, but the nearest data files may be offline on JSOC. "
        f"JSOC suggests contacting jsoc@sun.stanford.edu for offline files."
    )


def generate_output_and_map_names(
    date,
    map_type,
    output_dir,
    method_used="sph",
    drms_email: str | None = None,
):
    """Generate output filename and download magnetogram file based on map type."""

    map_type = normalize_map_type(map_type)
    date_datetime = parse_iso_datetime(date)

    cr_number = int(sunpy.coordinates.sun.carrington_rotation_number(date_datetime))
    logger.info(f"Carrington rotation number: {cr_number}")

    output_name = build_output_name(
        map_type,
        output_dir,
        method_used=method_used,
    )

    if map_type == "WSO":
        map_name = f"WSO.{cr_number}.txt"
        remote_file = f"http://wso.stanford.edu/synoptic/{map_name}"
    elif is_gong_map_type(map_type) or map_type in {"ADAPT", "HMI_fdt"}:
        candidate = select_nearest_candidate(
            list_remote_candidates(date_datetime, map_type),
            date_datetime,
        )
        map_name = candidate.name
        remote_file = candidate.remote_url
    elif map_type == "HMI_small":
        map_name = f"hmi.Synoptic_Mr_small.{cr_number}.fits"
        remote_file = f"http://jsoc.stanford.edu/data/hmi/synoptic/{map_name}"
    elif map_type == "HMI_polfil":
        map_name = f"hmi.Synoptic_Mr_polfil.{cr_number}.fits"
        remote_file = f"http://jsoc.stanford.edu/data/hmi/synoptic/{map_name}"
    elif map_type == "HMI_SYNC":
        local_file, map_name = download_hmi_sync_magnetogram(
            date_datetime,
            output_dir,
            drms_email,
        )
        logger.info(f"Output file: {output_name}")
        logger.info(f"Downloaded map: {local_file}")
        return output_name, local_file
    elif map_type == "HMI_hourly":
        candidate = select_nearest_candidate(
            list_remote_candidates(date_datetime, map_type),
            date_datetime,
        )
        local_file, map_name = download_hmi_hourly_magnetogram(
            candidate,
            output_dir,
            drms_email=drms_email,
        )
        logger.info(f"Output file: {output_name}")
        logger.info(f"Downloaded map: {local_file}")
        return output_name, local_file
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


def build_processing_dates(
    start_date: str | datetime,
    cadence_hours: float | None = None,
    total_hours: float | None = None,
) -> list[datetime]:
    """Build target dates from a start date, cadence, and total duration."""
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
    map_type = normalize_map_type(map_type)
    return os.path.join(output_dir, f"{map_type.lower()}_{format_timestamp(date)}.png")


def resolve_figure_path(
    output_path_fig: str | None,
    output_dir: str,
    map_type: str,
    date: str | datetime,
    use_unique_name: bool = False,
) -> str:
    """Build a figure filename from an optional file or directory path."""
    if not output_path_fig:
        return default_figure_path(output_dir, map_type, date)

    filename = os.path.basename(output_path_fig)
    if filename.lower() == ".png":
        return default_figure_path(os.path.dirname(output_path_fig), map_type, date)

    if (
        output_path_fig.endswith(("/", "\\"))
        or os.path.isdir(output_path_fig)
        or not os.path.splitext(output_path_fig)[1]
    ):
        return default_figure_path(output_path_fig, map_type, date)

    if use_unique_name:
        return append_timestamp_to_path(output_path_fig, date)
    return output_path_fig


def generate_output_and_interpolation_map_names(
    date,
    map_type,
    output_dir,
    method_used="sph",
    download_dir=None,
    drms_email: str | None = None,
):
    """Generate output name and download four maps for temporal interpolation."""

    map_type = normalize_map_type(map_type)
    output_name = build_output_name(
        map_type,
        output_dir,
        method_used=method_used,
    )
    local_files, selection = download_interpolation_magnetograms(
        date,
        map_type,
        download_dir or output_dir,
        drms_email=drms_email,
    )
    logger.info(f"Output file: {output_name}")
    return output_name, local_files, selection
