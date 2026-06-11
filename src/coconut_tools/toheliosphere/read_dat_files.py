"""Reads physical simulation data from a formatted text file."""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def read_data(
    filename: str,
    reduced: bool = False,
    extended: bool = False,
) -> Tuple[
    str,
    Optional[float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    np.ndarray,
]:
    """Read physical simulation data from a formatted text file.

    The file contains grids (colatitude, longitude), then blocks for the fields.
    When reduced is True, only reads vr, number_density, temperature, Br.

    Args:
        filename (str): Path to the file to be read.
        reduced (bool, optional): If True, only read vr, number_density, temperature, Br.
            Default is False.
        extended (bool, optional): If True, also read the radius value. Default is False.

    Returns:
        tuple:
            date (str),
            r (Optional[float]),
            clt (np.ndarray),
            lon (np.ndarray),
            vr (np.ndarray) in km/s,
            vlon (Optional[np.ndarray]) in km/s,
            vclt (Optional[np.ndarray]) in km/s,
            density (np.ndarray) in m^-3,
            br (np.ndarray) in T (or whatever unit the file provides),
            bclt (Optional[np.ndarray]),
            blon (Optional[np.ndarray]),
            temp (np.ndarray) in K
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    date = lines[1].strip()

    headers = {
        "clt": "Colatitude grid points:\n",
        "lon": "Longitude grid points:\n",
        "vr": "vr\n",
        "vp": "vp\n",
        "vt": "vt\n",
        "density": "number_density\n",
        "temp": "temperature\n",
        "br": "Br\n",
        "bp": "Bp\n",
        "bt": "Bt\n",
    }


    if reduced:
        expected = ["clt", "lon", "vr", "density", "temp", "br"]
    else:
        expected = ["clt", "lon", "vr", "vp", "vt", "density", "temp", "br", "bp", "bt"]

    if extended:
        headers["r"] = "Radius of sphere:\n"
        expected.append('r')

    # Find indices of expected headers in the file
    indices = {}
    for key in expected:
        h = headers[key]
        if h not in lines:
            raise ValueError(f"Header {key} not found in file: {h.strip()}")
        indices[key] = lines.index(h)

    r: Optional[float] = None
    if extended:
        r = float(lines[indices["r"] + 1])

    idx_clt = indices["clt"] + 1
    idx_lon = indices["lon"] + 1

    nb_clt = int(lines[idx_clt - 2])
    nb_lon = int(lines[idx_lon - 2])

    clt = np.array([float(lines[idx_clt + i]) for i in range(nb_clt)])
    lon = np.array([float(lines[idx_lon + i]) for i in range(nb_lon)])

    def read_array(start: int, end: int) -> np.ndarray:
        arr = np.array([[float(v) for v in line.split()] for line in lines[start:end]])
        return arr.reshape(nb_lon, nb_clt)

    if reduced:
        vr = read_array(indices["vr"] + 1, indices["density"]) / 1000.0
        density = read_array(indices["density"] + 1, indices["temp"])
        temp = read_array(indices["temp"] + 1, indices["br"])
        br = read_array(indices["br"] + 1, len(lines))

        vlon = None
        vclt = None
        blon = None
        bclt = None
    else:
        vr = read_array(indices["vr"] + 1, indices["vp"]) / 1000.0
        vlon = read_array(indices["vp"] + 1, indices["vt"]) / 1000.0
        vclt = read_array(indices["vt"] + 1, indices["density"]) / 1000.0
        density = read_array(indices["density"] + 1, indices["temp"])
        temp = read_array(indices["temp"] + 1, indices["br"])
        br = read_array(indices["br"] + 1, indices["bp"])
        blon = read_array(indices["bp"] + 1, indices["bt"])
        bclt = read_array(indices["bt"] + 1, len(lines))

    return date, r, clt, lon, vr, vlon, vclt, density, br, bclt, blon, temp

