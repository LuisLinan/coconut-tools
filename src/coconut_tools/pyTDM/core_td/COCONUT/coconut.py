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
from datetime import datetime
from scipy.interpolate import griddata, RBFInterpolator, RegularGridInterpolator, LinearNDInterpolator
import os


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
                nbelements = int(exp.search(line).group(1))
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

def restart_coolfluid(path: str, vtsfile: str, path_save: str, alpha_0 :float, theta_0 :float, phi_0 :float, d :float,
                      a :float, R :float,name, method: str = 'rbf', save: bool = False) -> None:
    """
    Create the CFmesh with the Tdm within

    Args:
        path (str):      Where is the CFmesh file
        vtsfile (str):   Where is the vtk file
        path_save (str): Where save the newCFmesh
        method (str):    Which method we want to use for the interpolation
        save (bool):     If true we save a vtk file with the TDm in the solar wind

    Returns:
        None

    Raise:
        FileNotFoundError: If the CFmesh doesn't exist
        NameError:         If the interpolation method doesn't exist
    """
    inputfile = os.path.join(path, "corona.CFmesh")

    if not os.path.isfile(inputfile):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), inputfile)
    else:
        MHDinfile = open(inputfile, "r")

    lines = MHDinfile.readlines()

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    logger.info(f'Scripts starts at {current_time}')

    # define the structure of the CFmesh
    idx0, idx1, idx2, idx3, nbelements, nend, comment = readstruct(lines)

    # read the connectivity
    connectivity = np.loadtxt(lines[idx0:idx0 + nbelements], dtype=int)

    # read I don't what
    other = np.loadtxt(lines[idx0 + nbelements + 6:idx3], dtype=int)

    other2 = np.loadtxt(lines[idx3 + 5:idx1 - 2], dtype=int)

    # read the coordinates
    coordinates = np.loadtxt(lines[idx1:idx2 - 1], dtype=float)


    # cell centers
    n1 = connectivity[:, 0]
    n2 = connectivity[:, 1]
    n3 = connectivity[:, 2]
    n4 = connectivity[:, 3]
    n5 = connectivity[:, 4]
    n6 = connectivity[:, 5]

    cell_center_x = 1. / 6. * (
            coordinates[n1, 0] + coordinates[n2, 0] + coordinates[n3, 0] + coordinates[n4, 0] + coordinates[n5, 0] +
            coordinates[n6, 0])
    cell_center_y = 1. / 6. * (
            coordinates[n1, 1] + coordinates[n2, 1] + coordinates[n3, 1] + coordinates[n4, 1] + coordinates[n5, 1] +
            coordinates[n6, 1])
    cell_center_z = 1. / 6. * (
            coordinates[n1, 2] + coordinates[n2, 2] + coordinates[n3, 2] + coordinates[n4, 2] + coordinates[n5, 2] +
            coordinates[n6, 2])

    cell_centers = np.zeros((len(cell_center_x), 3))
    cell_centers[:, 0] = cell_center_x
    cell_centers[:, 1] = cell_center_y
    cell_centers[:, 2] = cell_center_z

    # Initial data
    bd = comment[-nend - 1][0] + 1
    bf = comment[-nend][0]
    Initialdata = np.loadtxt(lines[bd:bf], dtype=float)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    logger.info(f'Ok Cfmesh is read at {current_time}')

    # Time to interpolate
    DATA = createdata(vtsfile)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    logger.info(f'Ok data created and save at {current_time}')

    X_tointerp = DATA[:, 0]
    Y_tointerp = DATA[:, 1]
    Z_tointerp = DATA[:, 2]

    bx_tointerp = DATA[:, 3]
    by_tointerp = DATA[:, 4]
    bz_tointerp = DATA[:, 5]

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    logger.info(f'Time to interpolate at {current_time}')

    if re.match('nearest', method):

        Initialdata[:, 4] = griddata((X_tointerp, Y_tointerp, Z_tointerp), bx_tointerp, cell_centers, method=method)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        logger.info(f'1/3  {current_time}')

        Initialdata[:, 5] = griddata((X_tointerp, Y_tointerp, Z_tointerp), by_tointerp, cell_centers, method=method)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        logger.info(f'2/3  {current_time}')

        Initialdata[:, 6] = griddata((X_tointerp, Y_tointerp, Z_tointerp), bz_tointerp, cell_centers, method=method)

    elif re.match('rbf', method):
        grid_interp = np.stack((X_tointerp, Y_tointerp, Z_tointerp), axis=1)

        ri = RBFInterpolator(grid_interp, bx_tointerp, kernel='linear', neighbors=50)
        Initialdata[:, 4] = ri(cell_centers)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        logger.info(f'1/3  {current_time}')

        ri = RBFInterpolator(grid_interp, by_tointerp, kernel='linear', neighbors=50)
        Initialdata[:, 5] = ri(cell_centers)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        logger.info(f'2/3  {current_time}')

        ri = RBFInterpolator(grid_interp, bz_tointerp, kernel='linear', neighbors=50)
        Initialdata[:, 6] = ri(cell_centers)

    elif re.match('linear', method):
        grid_interp = np.stack((X_tointerp, Y_tointerp, Z_tointerp), axis=1)
        ri = LinearNDInterpolator(grid_interp, bx_tointerp)
        Initialdata[:, 4] = ri(cell_centers)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        logger.info(f'1/3  {current_time}')

        ri = LinearNDInterpolator(grid_interp, by_tointerp)
        Initialdata[:, 5] = ri(cell_centers)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        logger.info(f'2/3  {current_time}')

        ri = LinearNDInterpolator(grid_interp, bz_tointerp)
        Initialdata[:, 6] = ri(cell_centers)

    else:
        raise NameError('Need a correct interpolation method')


    """
    logger.info(f'we add vr')
    xc = (1.0-d) * np.sin(theta_0) * np.cos(phi_0)
    yc = (1.0-d) * np.sin(theta_0) * np.sin(phi_0)
    zc = (1.0-d) * np.cos(theta_0)

    xnew=cell_centers[:, 0]-xc
    ynew=cell_centers[:, 1]-yc
    znew=cell_centers[:, 2]-zc

    xpr, ypr, zpr = rotation(xnew, ynew, znew, theta_0, phi_0, alpha_0, reverse=False)

    cltmeshnew = np.arccos(zpr / np.sqrt(xpr ** 2 + ypr ** 2 + zpr ** 2))

    rhonew = np.sqrt(xpr ** 2 + ypr ** 2 + zpr ** 2)

    rho = np.sqrt(rhonew ** 2 + R ** 2 - 2 * rhonew * R * np.sin(cltmeshnew))


    magnitude = np.sqrt(cell_centers[rho<=a, 0] ** 2 + cell_centers[rho<=a, 1] ** 2 + cell_centers[rho<=a, 2] ** 2)

    Initialdata[rho<=a, 1] = (500.0/480.24838) * cell_centers[rho<=a, 0] / magnitude
    Initialdata[rho<=a, 2] = (500.0/480.24838) * cell_centers[rho<=a, 1] / magnitude
    Initialdata[rho<=a, 3] = (500.0/480.24838) * cell_centers[rho<=a, 2] / magnitude
    """
    path_file = path_save + '/' + name + '/'
    os.makedirs(path_file, exist_ok=True)

    # save the vtk file
    if save:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        logger.info(f'Save the new vtk file at {current_time}')
        n_cells, n_elbycell = np.shape(connectivity)
        n_elbycell = n_elbycell - 1
        cells = np.zeros((n_cells, n_elbycell + 1))
        cells[:, 1:] = connectivity[:, :-1].astype('int32')
        cells[:, 0] = n_elbycell
        celltypes = np.empty(n_cells, dtype=np.uint32)
        celltypes[:] = 13
        cells = cells.ravel().astype(int)
        grid = pyv.UnstructuredGrid(cells, celltypes, coordinates)
        grid.cell_data['Bx'] = Initialdata[:, 4]
        grid.cell_data['By'] = Initialdata[:, 5]
        grid.cell_data['Bz'] = Initialdata[:, 6]
        grid.cell_data['Vx'] = Initialdata[:, 1]
        grid.cell_data['Vy'] = Initialdata[:, 2]
        grid.cell_data['Vz'] = Initialdata[:, 3]
        grid.save(f'{path_file}TDM_after.vtk')

    # Now save the new file
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    logger.info(f'Save the file at {current_time}')

    MHDinfile.close()

    outputfile = path_file + 'corona_restart.CFmesh'

    with open(outputfile, "w", newline="\n") as MHDoutfile:

        # 1. Header comments
        for i in range(0, 15):
            MHDoutfile.write(comment[i][1])

        # 2. Connectivity
        for row in connectivity.astype(str):
            MHDoutfile.write(" ".join(row) + "\n")

        for i in range(15, 21):
            MHDoutfile.write(comment[i][1])

        # 3. Other
        for row in other.astype(str):
            MHDoutfile.write(" ".join(row) + "\n")

        for i in range(21, 26):
            MHDoutfile.write(comment[i][1])

        # 4. Other2
        for row in other2.astype(str):
            MHDoutfile.write(" ".join(row) + "\n")

        for i in range(26, 28):
            MHDoutfile.write(comment[i][1])

        # 5. Coordinates
        for row in coordinates.astype(str):
            MHDoutfile.write(" ".join(row) + "\n")

        # 6. Comment before Initialdata
        MHDoutfile.write(comment[28][1])

        # 7. Initial data
        for row in Initialdata.astype(str):
            MHDoutfile.write(" ".join(row) + "\n")

        # 8. Footer comment
        MHDoutfile.write(comment[-1][1])

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    logger.info(f'End at {current_time}')

    return

def createdata(vtkfile: str) -> List[List]:
    """ Create the data file containing the magnetic field

    Args:
       vtkfile (str):  Magnetic field file

    Returns:
       List (list(float,float,float,float,float,float) : coordinates and magnetic field components

    Raises:
       FileNotFoundError: if the Tdm file doesn't exist

    """

    vtkfile=vtkfile+'.vts'
    if not os.path.isfile(vtkfile):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), vtkfile)
    else:
        # noinspection PyUnusedLocal
        mesh = pyv.read(vtkfile)

    B_cart = mesh.get_array('B_cart')
    points_loc = vtk_to_numpy(mesh.GetPoints().GetData())

    # Flat the value
    X_tointerp = np.ndarray.flatten(points_loc[:, 0])
    Y_tointerp = np.ndarray.flatten(points_loc[:, 1])
    Z_tointerp = np.ndarray.flatten(points_loc[:, 2])
    bx_tointerp = np.ndarray.flatten(B_cart[:, 0])
    by_tointerp = np.ndarray.flatten(B_cart[:, 1])
    bz_tointerp = np.ndarray.flatten(B_cart[:, 2])

    # remove the nan values
    mask = np.isfinite(bx_tointerp)

    X_tointerp = X_tointerp[mask]
    Y_tointerp = Y_tointerp[mask]
    Z_tointerp = Z_tointerp[mask]
    bx_tointerp = bx_tointerp[mask]
    by_tointerp = by_tointerp[mask]
    bz_tointerp = bz_tointerp[mask]

    DATA = np.array(
        [[X_tointerp[i], Y_tointerp[i], Z_tointerp[i], bx_tointerp[i], by_tointerp[i], bz_tointerp[i]] for i in
         range(len(X_tointerp))])

    return DATA