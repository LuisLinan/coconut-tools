"""
Module for handling CFmesh files and injecting TDm/RBSL structures into the COCONUT model.

This module provides functions to:
- read and parse `corona.CFmesh` files used in MHD simulations with COCONUT,
- interpolate Cartesian magnetic fields (from VTK/VTU files) to the cell centers of a CFmesh grid,
- generate a new CFmesh file with the injected magnetic structure,
- convert Cartesian magnetic fields to spherical components (Br, Bθ, Bφ),
- aggregate multiple `.vtu` files from parallel COCONUT simulations.

Main functions:
- `read_CFmesh`: reads a CFmesh file and returns the magnetic field components in spherical coordinates.
- `read_vtu`: merges `.vtu` files from a parallel run of COCONUT.
- `readstruct`: extracts the internal structure and block indices from a CFmesh file.

This module is used to enable the injection of TDm or RBSL flux ropes as initial conditions
in COCONUT-based simulations.
"""

import numpy as np
import pandas as pd
import pyvista as pyv
from tqdm import tqdm
from vtk.util.numpy_support import vtk_to_numpy
import errno
import os.path
from typing import List, Tuple
import re
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def readstruct(lines: List[str]) -> Tuple[int, int, int, int, int, int, list]:
    """ Find the structure of the CFmesh

    Args:
        lines (list(str)): contains the CFmesh lines

    Returns:
        idx0 (int): where the node start
        idx1 (int): where the state start
        idx2 (int): where the elem start
        idx3 (int): where I don't know what start
        nend (int): nbr of !end
        nbelements (int): nb of elements
        comment list(list(int,str)): all the non number lines and where there are
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
                nbelements = int(exp.findall(line)[0])
            elif line.startswith("!TRS_NAME Outlet"):
                idx3 = i
            elif line.startswith("!END"):
                nend += 1
            else:
                continue
        else:
            continue
    return idx0, idx1, idx2, idx3, nbelements, nend, comment

def read_CFmesh(path_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read and extract magnetic field data from a COCONUT CFmesh file.

    This function computes the magnetic field components in spherical coordinates
    (radial, co-latitudinal, and longitudinal) at the center of each mesh cell.

    Args:
        path_file (str): Path to the directory containing the 'corona.CFmesh' file.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - Unique cell centers (N x 3)
            - Radial magnetic field component `br`
            - Co-latitudinal magnetic field component `bclt`
            - Longitudinal magnetic field component `blon`

    Raises:
        FileNotFoundError: If 'corona.CFmesh' does not exist in the given path.
    """

    inputfile = os.path.join(path_file, 'corona.CFmesh')
    if not os.path.isfile(inputfile):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), inputfile)

    with open(inputfile, "r") as MHDinfile:
        lines = MHDinfile.readlines()

    # Get structure and metadata
    idx0, idx1, idx2, idx3, nbelements, nend, comment = readstruct(lines)

    connectivity = np.loadtxt(lines[idx0:idx0 + nbelements], dtype=int)
    other = np.loadtxt(lines[idx0 + nbelements + 6:idx3], dtype=int)
    other2 = np.loadtxt(lines[idx3 + 5:idx1 - 2], dtype=int)
    coordinates = np.loadtxt(lines[idx1:idx2 - 1], dtype=float)

    # Compute cell centers
    nodes = [connectivity[:, i] for i in range(6)]
    cell_center_x = sum(coordinates[n, 0] for n in nodes) / 6.0
    cell_center_y = sum(coordinates[n, 1] for n in nodes) / 6.0
    cell_center_z = sum(coordinates[n, 2] for n in nodes) / 6.0

    cell_centers = np.stack((cell_center_x, cell_center_y, cell_center_z), axis=1)

    # Load initial data
    bd = comment[-nend - 1][0] + 1
    bf = comment[-nend][0]
    Initialdata = np.loadtxt(lines[bd:bf], dtype=np.float64)

    x, y, z = cell_centers[:, 0], cell_centers[:, 1], cell_centers[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r_bis = np.sqrt(x ** 2 + y ** 2)

    Bx, By, Bz = Initialdata[:, 4], Initialdata[:, 5], Initialdata[:, 6]

    EPSILON = 0  # Can be changed to small positive value if needed to avoid division by zero

    br = (x * Bx + y * By + z * Bz) / r
    blon = (-y * Bx + x * By) / (r_bis + EPSILON)
    bclt = (x * z * Bx + y * z * By - (r_bis + EPSILON) ** 2 * Bz) / ((r_bis + EPSILON) * r)

    uni_grid_after, index_after = np.unique(cell_centers, axis=0, return_index=True)

    return uni_grid_after, br[index_after], bclt[index_after], blon[index_after]

def read_vtu(path_file: str, nb_proc: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge and clean multiple VTU files generated by COCONUT and compute magnetic field components in spherical coordinates.

    Args:
        path_file (str): Directory where the VTU files are located.
        nb_proc (int): Number of processors (and thus number of VTU files to read).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - Unique 3D coordinates of the points (shape: N x 3)
            - Radial component of B field (br)
            - Co-latitudinal component (bt)
            - Azimuthal component (bp)
    """

    logger.info('Loading VTU files...')
    for i in tqdm(range(nb_proc), desc='Reading VTU files'):

        fileloc = os.path.join(path_file, f'corona-flow0-P{i}.vtu')
        mesh = pyv.read(fileloc)

        # Grid
        points_loc = vtk_to_numpy(mesh.GetPoints().GetData())
        nb_pts_loc = points_loc.shape[0]

        x_loc, y_loc, z_loc = points_loc[:, 0], points_loc[:, 1], points_loc[:, 2]

        # Magnetic field
        bx_loc = mesh.get_array('Bx')
        by_loc = mesh.get_array('By')
        bz_loc = mesh.get_array('Bz')

        if i == 0:
            points, x, y, z = points_loc, x_loc, y_loc, z_loc
            bx, by, bz = bx_loc, by_loc, bz_loc
        else:
            points = np.concatenate((points, points_loc))
            x = np.concatenate((x, x_loc))
            y = np.concatenate((y, y_loc))
            z = np.concatenate((z, z_loc))
            bx = np.concatenate((bx, bx_loc))
            by = np.concatenate((by, by_loc))
            bz = np.concatenate((bz, bz_loc))

    # Remove duplicates
    df = pd.DataFrame(points)
    df_unique = df.drop_duplicates()
    idx = df_unique.index.to_numpy()
    points = df_unique.to_numpy()
    nb_dup = len(x) - len(points)

    x, y, z = x[idx], y[idx], z[idx]
    bx, by, bz = bx[idx], by[idx], bz[idx]

    logger.info(f'Removed {nb_dup} duplicates.')

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    rxy = np.sqrt(x ** 2 + y ** 2)

    br = (x * bx + y * by + z * bz) / r
    bt = (x * z * bx + y * z * by - (x ** 2 + y ** 2) * bz) / (r * rxy)
    bp = (-y * bx + x * by) / rxy

    return points, br, bt, bp