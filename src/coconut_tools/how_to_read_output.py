"""
This module provides functions to read and process simulation outputs from the COCONUT code.
It includes conversion of CFmesh files and VTU files into structured data and spherical coordinates
suitable for heliospheric modeling and visualization.
"""

import os
import errno
import numpy as np
from datetime import datetime
import re
from typing import List, Tuple
import pandas as pd
import pyvista as pyv

from coconut_tools.logger_config import setup_logger

logger = setup_logger(__name__)


def readstruct(lines: List[str]) -> Tuple[int, int, int, int, int, int, list]:
    """Parse structure of CFmesh file and extract section indices and metadata.

    Args:
        lines (List[str]): List of lines from the CFmesh file.

    Returns:
        Tuple containing:
            - idx0 (int): Index where element connectivity list starts.
            - idx1 (int): Index where node list starts.
            - idx2 (int): Index where state list starts.
            - idx3 (int): Index where Outlet section starts.
            - nbelements (int): Number of elements.
            - nend (int): Number of "!END" markers.
            - comment (List[List[int, str]]): List of all comment lines with their indices.
    """
    exp = re.compile(r'!NB_ELEM (-?\d+)')

    nbelements, idx0, idx1, idx2, idx3, nend = 0, 0, 0, 0, 0, 0
    comment = []

    for i, line in enumerate(lines):
        if line.startswith("!"):
            comment.append([i, line])
            if line.startswith("!LIST_NODE"):
                idx1 = i + 1
            elif line.startswith("!LIST_STATE 1"):
                idx2 = i + 1
            elif line.startswith("!LIST_ELEM"):
                idx0 = i + 1
            elif line.startswith("!NB_ELEM "):
                nbelements = int(exp.search(line).group(1))
            elif line.startswith("!TRS_NAME Outlet"):
                idx3 = i
            elif line.startswith("!END"):
                nend += 1

    return idx0, idx1, idx2, idx3, nbelements, nend, comment


def read_cfmesh_file(input_cfmesh: str) -> dict:
    """Read and parse a CFmesh output file from COCONUT simulations.

    Args:
        input_cfmesh (str): Path to the CFmesh input file.

    Returns:
        dict: Dictionary containing parsed mesh and physical quantities:
            - 'connectivity': element connectivity array
            - 'coordinates': node coordinates array
            - 'cell_centers': computed center positions for each element
            - 'B': magnetic field components (Bx, By, Bz)
            - 'V': velocity components (Vx, Vy, Vz)
            - 'rho': density (cm^-3)
            - 'P': pressure (Pa)
            - 'T': temperature (K)
            - 'r': radial distances of cell centers
    """

    if not os.path.isfile(input_cfmesh):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), input_cfmesh)

    with open(input_cfmesh, "r") as f:
        lines = f.readlines()

    # Logging start time
    logger.info(f"Script started at {datetime.now().strftime('%H:%M:%S')}")

    # Structure parsing
    idx0, idx1, idx2, idx3, nbelements, nend, comment = readstruct(lines)

    connectivity = np.loadtxt(lines[idx0:idx0 + nbelements], dtype=int)
    coordinates = np.loadtxt(lines[idx1:idx2 - 1], dtype=float)
    other = np.loadtxt(lines[idx0 + nbelements + 6:idx3], dtype=int)  # placeholder name
    other2 = np.loadtxt(lines[idx3 + 5:idx1 - 2], dtype=int)  # placeholder name

    # Cell centers computation
    nodes = [connectivity[:, i] for i in range(6)]
    cell_centers = np.mean([coordinates[n] for n in nodes], axis=0)

    # Extract initial data (physical variables)
    bd = comment[-nend - 1][0] + 1
    bf = comment[-nend][0]
    Initialdata = np.loadtxt(lines[bd:bf], dtype=np.float64)

    # Physical quantities
    r = np.linalg.norm(cell_centers, axis=1)
    rho = Initialdata[:, 0] * 1.67e-13 / 1.67e-27 # n'=rho/mp=mu*n
    V = Initialdata[:, 1:4] * 480248.0 #m/s
    B = Initialdata[:, 4:7] * 2.2e-4 #T
    P = Initialdata[:, 7] * 0.03851 # Pa
    T = P / rho / 2 / (1.38e-23) 

    return {
        'connectivity': connectivity,
        'coordinates': coordinates,
        'cell_centers': cell_centers,
        'r': r,
        'rho': rho,
        'V': V,
        'B': B,
        'P': P,
        'T': T
    }

def read_vtu_to_spherical_vectors(input_vtu: str) -> dict:
    """Convert a VTU file from COCONUT into spherical coordinates and physical quantities.

    Args:
        input_vtu (str): Path to the VTU file to read.

    Returns:
        dict: Dictionary of spherical components and coordinates including:
            - 'r': radial distances
            - 'theta': polar angle (rad)
            - 'phi': azimuthal angle (rad)
            - 'phi2pi': azimuth in [0, 2pi]
            - 'vr': radial velocity
            - 'vclt': colatitudinal velocity
            - 'vlon': longitudinal velocity
            - 'br': radial magnetic field
            - 'bclt': colatitudinal magnetic field
            - 'blon': longitudinal magnetic field
            - 'rho': density
            - 'p': pressure

    Notes:
        Normalization constants (if needed):
            vrkms = vr * 480248.0         # from code units to km/s
            vpkms = vlon * 480248.0       # from code units to km/s
            vtkms = vclt * 480248.0       # from code units to km/s

            brt   = br * 2.2e-4            # from code units to Tesla
            blont = blon * 2.2e-4          # from code units to Tesla
            bclt  = bclt * 2.2e-4          # from code units to Tesla

            tk    = p / rho * 1.7756e7     # from code units to Kelvin
            nsi   = rho * 1.0e14           # from code units to m3
    """

    logger.info(f"Loading file {input_vtu}")

    mesh = pyv.read(input_vtu)
    points = mesh.points
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Quantities
    rho = mesh['rho']
    p = mesh['p']
    bx = mesh['Bx']
    by = mesh['By']
    bz = mesh['Bz']
    vx = mesh['v'][:, 0]
    vy = mesh['v'][:, 1]
    vz = mesh['v'][:, 2]

    # Remove duplicates
    df = pd.DataFrame(points)
    df_drop = df.drop_duplicates()
    idx = df_drop.index.to_numpy()
    points = df_drop.to_numpy()

    nb_dup = len(points) - len(idx)
    x, y, z = x[idx], y[idx], z[idx]
    rho, p = rho[idx], p[idx]
    bx, by, bz = bx[idx], by[idx], bz[idx]
    vx, vy, vz = vx[idx], vy[idx], vz[idx]

    logger.info(f"Removed {nb_dup} duplicates.")
    logger.info("Converting from cartesian to spherical coordinates...")

    EPSILON = 1e-8
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r_bis = np.sqrt(x ** 2 + y ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    phi2pi = phi % (2 * np.pi)

    # Compute spherical velocity components
    vr = (x * vx + y * vy + z * vz) / r
    vlon = (-y * vx + x * vy) / (r_bis + EPSILON)
    vclt = (x * z * vx + y * z * vy - (r_bis + EPSILON) * vz) / ((r_bis + EPSILON) * r)

    # Compute spherical magnetic field components
    br = (x * bx + y * by + z * bz) / r
    blon = (-y * bx + x * by) / (r_bis + EPSILON)
    bclt = (x * z * bx + y * z * by - (r_bis + EPSILON) * bz) / ((r_bis + EPSILON) * r)

    logger.info("Conversion complete.")

    return {
        'r': r,
        'theta': theta,
        'phi': phi,
        'phi2pi': phi2pi,
        'vr': vr,
        'vclt': vclt,
        'vlon': vlon,
        'br': br,
        'bclt': bclt,
        'blon': blon,
        'rho': rho,
        'p': p
    }


if __name__ == "__main__":

    # Example VTU file path for test
    input_path = "./example_coconut_output.vtu"  # Replace with real file path
    result = read_vtu_to_spherical_vectors(input_path)

    # Example VTU file path for test
    input_path = "./example_coconut_output.CFmesh"  # Replace with real file path
    result = read_cfmesh_file(input_path)
    