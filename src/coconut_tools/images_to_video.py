"""
images_to_video.py

Build a chronological MP4 and GIF from a time-ordered series of images.

The input filenames are expected to follow the pattern:
    <prefix>_YYYY-MM-DDTHH-MM-SS.<ext>

Example:
    bclt_2019-07-02T12-03-58.png
    n_2019-07-02T12-03-58.pdf
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Iterable, List

import numpy as np
import imageio

from coconut_tools.logger_config import setup_logger

logger = setup_logger(__name__)


TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%S"


@dataclass(frozen=True)
class MediaConfig:
    """Configuration for building media outputs.

    Args:
        input_dir (Path): Directory containing the images.
        output_dir (Path | None): Directory where outputs are written. If None,
            a subfolder named "output_media" is created in input_dir.
        prefix (str): Filename prefix before the timestamp.
        extension (str): Image extension without dot (e.g., "png" or "pdf").
        fps (float): Frames per second for outputs.
        gif_loop (int): GIF loop count. 0 means infinite.
    """

    input_dir: Path
    output_dir: Path | None
    prefix: str
    extension: str = "png"
    fps: float = 3.0
    gif_loop: int = 0


def _parse_timestamp(filename: str, prefix: str) -> datetime | None:
    """Parse the timestamp from a filename.

    Args:
        filename (str): Filename without extension.
        prefix (str): Expected prefix before the underscore.

    Returns:
        datetime | None: Parsed datetime if successful, otherwise None.
    """
    pattern = rf"^{re.escape(prefix)}_(\d{{4}}-\d{{2}}-\d{{2}}T\d{{2}}-\d{{2}}-\d{{2}})$"
    match = re.match(pattern, filename)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), TIMESTAMP_FORMAT)
    except ValueError:
        return None


def _parse_unix_timestamp(filename: str, prefix: str, postfix: str | None = None) -> datetime | None:
    """Parse a Unix timestamp from a filename with an optional postfix.

    Expected format:
        <prefix>_<unix_timestamp>-<postfix>

    Args:
        filename (str): Filename without extension.
        prefix (str): Expected prefix before the underscore.
        postfix (str | None): Optional postfix after the dash. If provided,
            it must match exactly.

    Returns:
        datetime | None: Parsed datetime in UTC if successful, otherwise None.
    """
    postfix_pattern = r".+" if postfix is None else re.escape(postfix)
    pattern = rf"^{re.escape(prefix)}_(\d+)-({postfix_pattern})$"
    match = re.match(pattern, filename)
    if not match:
        return None
    try:
        return datetime.utcfromtimestamp(int(match.group(1)))
    except (ValueError, OSError, OverflowError):
        return None


def _find_sorted_images(input_dir: Path, prefix: str, extension: str) -> List[Path]:
    """Find and sort image files chronologically based on filename timestamps.

    Args:
        input_dir (Path): Directory containing images.
        prefix (str): Filename prefix.
        extension (str): File extension without dot.

    Returns:
        list[Path]: Sorted list of image file paths.
    """
    logger.info("Scanning for files in %s", input_dir)
    pattern = f"{prefix}_*.{extension}"
    candidates = sorted(input_dir.glob(pattern))
    if not candidates:
        logger.warning("No files found with pattern %s", pattern)
        return []

    with_timestamps = []
    for path in candidates:
        timestamp = _parse_timestamp(path.stem, prefix)
        if timestamp is None:
            logger.warning("Skipping file with unexpected name: %s", path.name)
            continue
        with_timestamps.append((timestamp, path))

    if not with_timestamps:
        logger.error("No files matched the expected timestamp format for prefix %s", prefix)
        return []

    with_timestamps.sort(key=lambda item: item[0])
    sorted_paths = [path for _, path in with_timestamps]
    logger.info("Found %d image(s) to process", len(sorted_paths))
    return sorted_paths


def _find_sorted_images_unix_timestamp(
    input_dir: Path, prefix: str, extension: str, postfix: str | None = None
) -> List[Path]:
    """Find and sort image files based on Unix timestamps in filenames.

    Args:
        input_dir (Path): Directory containing images.
        prefix (str): Filename prefix.
        extension (str): File extension without dot.
        postfix (str | None): Optional postfix after the timestamp.

    Returns:
        list[Path]: Sorted list of image file paths.
    """
    logger.info("Scanning for files in %s", input_dir)
    pattern = f"{prefix}_*.{extension}"
    candidates = sorted(input_dir.glob(pattern))
    if not candidates:
        logger.warning("No files found with pattern %s", pattern)
        return []

    with_timestamps = []
    for path in candidates:
        timestamp = _parse_unix_timestamp(path.stem, prefix, postfix=postfix)
        if timestamp is None:
            logger.warning("Skipping file with unexpected name: %s", path.name)
            continue
        with_timestamps.append((timestamp, path))

    if not with_timestamps:
        logger.error("No files matched the expected Unix timestamp format for prefix %s", prefix)
        return []

    with_timestamps.sort(key=lambda item: item[0])
    sorted_paths = [path for _, path in with_timestamps]
    logger.info("Found %d image(s) to process", len(sorted_paths))
    return sorted_paths


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize frame data to a 3-channel uint8 array.

    Args:
        frame (np.ndarray): Input image array.

    Returns:
        np.ndarray: Normalized 3-channel uint8 image.
    """
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame


def _resolve_output_dir(input_dir: Path, output_dir: Path | None) -> Path:
    """Resolve and create the output directory if needed.

    Args:
        input_dir (Path): Base input directory.
        output_dir (Path | None): Desired output directory.

    Returns:
        Path: Existing or newly created output directory.
    """
    resolved = output_dir or (input_dir / "output_media")
    resolved.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", resolved)
    return resolved


def build_gif_and_video(config: MediaConfig, postfix: str | None = None) -> None:
    """Create GIF and MP4 outputs from a series of images.

    Args:
        config (MediaConfig): Configuration for building media outputs.

    Returns:
        None
    """
    input_dir = config.input_dir
    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return

    output_dir = _resolve_output_dir(input_dir, config.output_dir)
    files = _find_sorted_images(input_dir, config.prefix, config.extension)
    if not files:
        return

    fps = max(float(config.fps), 0.1)
    postfix_tag = f"_{postfix}" if postfix else ""
    mp4_path = output_dir / f"{config.prefix}{postfix_tag}_timeline.mp4"
    gif_path = output_dir / f"{config.prefix}{postfix_tag}_timeline.gif"

    logger.info("Writing MP4 to %s (fps=%.2f)", mp4_path, fps)
    logger.info("Writing GIF to %s (fps=%.2f)", gif_path, fps)

    try:
        with imageio.get_writer(mp4_path, format="ffmpeg", fps=fps) as mp4_writer, imageio.get_writer(
            gif_path, format="gif", mode="I", duration=1.0 / fps, loop=config.gif_loop
        ) as gif_writer:
            for idx, path in enumerate(files, start=1):
                logger.info("Reading frame %d/%d: %s", idx, len(files), path.name)
                frame = imageio.v3.imread(path)
                frame = _normalize_frame(frame)
                mp4_writer.append_data(frame)
                gif_writer.append_data(frame)
    except Exception as exc:
        logger.exception("Failed to write media outputs: %s", exc)
        logger.error("If MP4 fails, ensure ffmpeg is installed and on PATH.")
        return

    logger.info("Done: %s and %s", mp4_path.name, gif_path.name)


def build_gif_and_video_unix_timestamp(config: MediaConfig, postfix: str | None = None) -> None:
    """Create GIF and MP4 outputs using Unix timestamps in filenames.

    Args:
        config (MediaConfig): Configuration for building media outputs.
        postfix (str | None): Optional postfix after the timestamp (e.g., "expansion_factor").

    Returns:
        None
    """
    input_dir = config.input_dir
    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return

    output_dir = _resolve_output_dir(input_dir, config.output_dir)
    files = _find_sorted_images_unix_timestamp(input_dir, config.prefix, config.extension, postfix=postfix)
    if not files:
        return

    fps = max(float(config.fps), 0.1)
    postfix_tag = f"_{postfix}" if postfix else ""
    mp4_path = output_dir / f"{config.prefix}{postfix_tag}_timeline.mp4"
    gif_path = output_dir / f"{config.prefix}{postfix_tag}_timeline.gif"

    logger.info("Writing MP4 to %s (fps=%.2f)", mp4_path, fps)
    logger.info("Writing GIF to %s (fps=%.2f)", gif_path, fps)

    try:
        with imageio.get_writer(mp4_path, format="ffmpeg", fps=fps) as mp4_writer, imageio.get_writer(
            gif_path, format="gif", mode="I", duration=1.0 / fps, loop=config.gif_loop
        ) as gif_writer:
            for idx, path in enumerate(files, start=1):
                logger.info("Reading frame %d/%d: %s", idx, len(files), path.name)
                frame = imageio.v3.imread(path)
                frame = _normalize_frame(frame)
                mp4_writer.append_data(frame)
                gif_writer.append_data(frame)
    except Exception as exc:
        logger.exception("Failed to write media outputs: %s", exc)
        logger.error("If MP4 fails, ensure ffmpeg is installed and on PATH.")
        return

    logger.info("Done: %s and %s", mp4_path.name, gif_path.name)


if __name__ == "__main__":
    """Entry point with inline configuration (no argparse)."""
    
    prefix =  ["bclt","blon","br","vr","vlon","nscaled","vclt"]
    
    for pre in prefix:
        config = MediaConfig(
            input_dir=Path(r"E:/euhforia/dat8/image"),
            output_dir=None,
            prefix=pre,
            extension="png",
            fps=3.0,
            gif_loop=0,
        )
        build_gif_and_video(config)

    for pre in prefix:
        config = MediaConfig(
            input_dir=Path(r"E:/euhforia/dat4/image"),
            output_dir=None,
            prefix=pre,
            extension="png",
            fps=3.0,
            gif_loop=0,
        )
        build_gif_and_video(config)

    """
    postfix = ["expansion_factor", "magnetogram", "number_density", "radial_field", "field_lower_boundary",
               "radial_velocity", "temperature","open_and_closed_field_regions"]
    for post in postfix:
        config = MediaConfig(
            input_dir=Path(r"E:/euhforia/image/wsa"),
            output_dir=None,
            prefix="solar_wind_boundary",
            extension="png",
            fps=12.0,
            gif_loop=0,
        )
        build_gif_and_video_unix_timestamp(config, postfix=post)


    
    for pre in prefix:
        config = MediaConfig(
            input_dir=Path(r"E:/euhforia/wsa/result/image"),
            output_dir=None,
            prefix=pre,
            extension="png",
            fps=3.0,
            gif_loop=0,
        )
        build_gif_and_video(config)

    for pre in prefix:
        config = MediaConfig(
            input_dir=Path(r"E:/euhforia/dat8/result/image"),
            output_dir=None,
            prefix=pre,
            extension="png",
            fps=3.0,
            gif_loop=0,
        )
        build_gif_and_video(config)
    

    postfix = ["expansion_factor", "magnetogram", "number_density", "radial_field", "field_lower_boundary",
               "radial_velocity", "temperature","open_and_closed_field_regions"]
    for post in postfix:
        config = MediaConfig(
            input_dir=Path(r"E:/euhforia/image/wsa"),
            output_dir=None,
            prefix="solar_wind_boundary",
            extension="png",
            fps=12.0,
            gif_loop=0,
        )
        build_gif_and_video_unix_timestamp(config, postfix=post)

    postfix = ["input"]
    for post in postfix:
        config = MediaConfig(
            input_dir=Path(r"E:/euhforia/image/inner_boundary/"),
            output_dir=None,
            prefix="solar_wind_boundary",
            extension="png",
            fps=12.0,
            gif_loop=0,
        )
        build_gif_and_video_unix_timestamp(config, postfix=post)
    """