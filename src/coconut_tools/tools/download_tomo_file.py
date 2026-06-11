"""
Download STEREO tomography files (.dat and .jpg) from solarphysics.abersytwyth.ac.uk archive.

This module fetches files of the format:
    - tomo_sta_cor2a_<yyyymmdd>_<altitude>.dat
    - tomo_sta_cor2a_<yyyymmdd>_<altitude>.jpg
from:
    https://solarphysics.aber.ac.uk/Archives/tomography/cor2a/<year>/distance_<altitude>/

Functions:
    - download_tomography_file(date, altitude, output_dir): Downloads .dat and .jpg.
    - list_available_altitudes(year): List available altitudes for a given year.
"""

import os
import requests
from coconut_tools.tools.logger_config import setup_logger
from bs4 import BeautifulSoup
from pathlib import Path

BASE_URL = "https://solarphysics.aber.ac.uk/Archives/tomography/cor2a"


logger = setup_logger(__name__)

def download_tomography_file(date: str, altitude: str, output_dir: str) -> None:
    """
    Download the .dat and .jpg tomography file for a given date and altitude.

    Args:
        date (str): The date in 'YYYYMMDD' format.
        altitude (str): The altitude string, e.g., '5-0', '4-4'.
        output_dir (str): The directory where to save the files.

    Returns:
        None
    """
    year = date[:4]
    url = f"{BASE_URL}/{year}/distance_{altitude}/"

    dat_filename = f"tomo_sta_cor2a_{date}_{altitude}.dat"
    jpg_filename = f"tomo_sta_cor2a_{date}_{altitude}.jpg"

    os.makedirs(output_dir, exist_ok=True)

    try:
        for filename in [dat_filename, jpg_filename]:
            file_url = url + filename
            response = requests.get(file_url)
            if response.status_code == 200:
                with open(Path(output_dir) / filename, "wb") as f:
                    f.write(response.content)
                logger.info(f"Downloaded: {filename}")
            else:
                raise FileNotFoundError
    except FileNotFoundError:
        logger.error(f"File not found for altitude '{altitude}' on {date}. Checking available altitudes...")
        logger.warning("Available altitudes:")
        for alt in list_available_altitudes(year):
            logger.warning(f" - {alt}")

def list_available_altitudes(year: str) -> list:
    """
    List all available altitude folders for a given year.

    Args:
        year (str): Year as a string (e.g., '2019').

    Returns:
        list: List of altitude folder names (e.g., ['5-0', '6-5']).
    """
    url = f"{BASE_URL}/{year}/"
    response = requests.get(url)
    if response.status_code != 200:
        logger.error(f"Could not access year directory for {year}.")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    folders = [a['href'].replace('distance_', '').replace('/', '')
               for a in soup.find_all('a', href=True) if a['href'].startswith('distance_')]
    return folders

if __name__ == '__main__':
    # Example usage
    download_tomography_file("20190704", "30-0", "./")
