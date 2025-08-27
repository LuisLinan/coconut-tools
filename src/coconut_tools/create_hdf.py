"""
Module for extracting and converting plasma simulation data into HDF5 format.

This module provides utility functions to process different types of simulation outputs:
    - CFmesh format (custom structured mesh files)
    - VTU files (e.g., from Paraview-compatible simulations like COCONUT)
    - Formatted .dat files

The main functionalities include:
    - Sampling physical quantities (velocity, magnetic field, density, temperature, etc.)
      at satellite locations or specific grid coordinates
    - Interpolating or extracting the closest grid point
    - Saving extracted time series into standardized HDF5 format

Available functions:
    - `create_hdf_from_cfmesh`: For CFmesh files
    - `create_hdf_from_vtu`: For VTU files
    - `create_hdf_from_dat`: For formatted .dat files with spherical coordinates

"""

import os
import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union
from scipy.interpolate import RBFInterpolator
import h5py
import numpy as np
import natsort
import pyvista as pv
from vtk.util.numpy_support import vtk_to_numpy
from coconut_tools.logger_config import setup_logger
import errno
from datetime import datetime
from coconut_tools.read_dat_files import read_data as read_dat


logger = setup_logger(__name__)

def find_nearest_index_vtu(points: np.ndarray, target: List[float]) -> int:
    """Find the index of the point closest to the target coordinates from vtu file.

    Args:
        points (np.ndarray): Array of shape (N, 3) containing point coordinates.
        target (List[float]): Target coordinates [x, y, z].

    Returns:
        int: Index of the closest point.
    """
    distances = np.linalg.norm(points - np.array(target), axis=1)
    idx = np.argmin(distances)
    logger.info(f"Closest point found at x={points[idx, 0]}, y={points[idx, 1]}, z={points[idx, 2]}")
    return idx


def create_hdf_from_vtu(
    satellite_positions: Dict[str, List[float]],
    folder_patterns: Dict[str, str],
    output_dir: Path,
    delta_t: float,
    time_step: float
) -> None:
    """Extracts plasma quantities from VTU files at satellite positions and stores them in HDF5.

    Args:
        satellite_positions (Dict[str, List[float]]): Mapping of satellite names to 3D positions.
        folder_patterns (Dict[str, str]): Mapping of case names to glob patterns of VTU files.
        output_dir (Path): Directory to store output HDF5 files.
        delta_t (float): Time step duration (in code units).
        time_step (float): Output time step multiplier.

    Returns:
        None
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for sat_name, sat_pos in satellite_positions.items():
        logger.info(f"Processing satellite: {sat_name}")

        # Identify first VTU file to determine mesh and index
        first_case = next(iter(folder_patterns))
        first_file = natsort.natsorted(glob.glob(folder_patterns[first_case]))[0]
        mesh = pv.read(first_file)
        points = vtk_to_numpy(mesh.GetPoints().GetData())
        idx = find_nearest_index_vtu(points, sat_pos)

        for case_name, pattern in folder_patterns.items():
            logger.info(f"\tProcessing case: {case_name}")
            vtu_files = natsort.natsorted(glob.glob(pattern))

            B, V, Vx, Vy, Vz, Bx, By, Bz, T, rho, time = ([] for _ in range(11))

            for i, file_path in enumerate(vtu_files):
                mesh = pv.read(file_path)
                try:
                    bx = mesh.get_array('Bx')[idx] * 2.2e5
                    by = mesh.get_array('By')[idx] * 2.2e5
                    bz = mesh.get_array('Bz')[idx] * 2.2e5
                    vx = mesh.get_array('v')[idx, 0] * 480.24838
                    vy = mesh.get_array('v')[idx, 1] * 480.24838
                    vz = mesh.get_array('v')[idx, 2] * 480.24838
                    density = mesh.get_array('rho')[idx] * 1.67e-16
                    pressure = mesh.get_array('p')[idx]
                    temp = pressure * 1.27 * 1.67e-24 / (2.0 * density * 1.3807e-16)

                    Bx.append(bx)
                    By.append(by)
                    Bz.append(bz)
                    Vx.append(vx)
                    Vy.append(vy)
                    Vz.append(vz)
                    rho.append(density)
                    T.append(temp)
                    B.append(np.sqrt(bx**2 + by**2 + bz**2))
                    V.append(np.sqrt(vx**2 + vy**2 + vz**2))
                    time.append(i * delta_t * time_step * 0.402)

                except Exception as e:
                    logger.warning(f"\t\tSkipping file {file_path} due to error: {e}")
                    continue

            # Save HDF5
            output_file = output_dir / f"Data_{case_name}.hdf5"
            with h5py.File(output_file, "w") as f:
                f.create_dataset('B', data=np.array(B))
                f.create_dataset('V', data=np.array(V))
                f.create_dataset('Vx', data=np.array(Vx))
                f.create_dataset('Vy', data=np.array(Vy))
                f.create_dataset('Vz', data=np.array(Vz))
                f.create_dataset('Bx', data=np.array(Bx))
                f.create_dataset('By', data=np.array(By))
                f.create_dataset('Bz', data=np.array(Bz))
                f.create_dataset('T', data=np.array(T))
                f.create_dataset('rho', data=np.array(rho))
                f.create_dataset('time', data=np.array(time))

            logger.info(f"\tSaved data to {output_file}")

def readstruct(lines: List[str]) -> Tuple[int, int, int, int, int, int, List[Tuple[int, str]]]:
    """
    Parses the structure of a CFmesh file by locating key section headers.

    Args:
        lines (List[str]): List of lines from the CFmesh file.

    Returns:
        Tuple containing:
            idx0 (int): Index where element connectivity starts.
            idx1 (int): Index where nodal coordinates start.
            idx2 (int): Index where state vector starts.
            idx3 (int): Index marking start of outlet-related section.
            nbelements (int): Number of elements in the mesh.
            nend (int): Number of !END markers.
            comment (List[Tuple[int, str]]): Positions and content of all comment lines.
    """
    exp = re.compile(r'!NB_ELEM (-?\d+)')
    idx0 = idx1 = idx2 = idx3 = nbelements = nend = 0
    comment = []

    for i, line in enumerate(lines):
        if line.startswith("!"):
            comment.append((i, line))
            if line.startswith("!LIST_ELEM"):
                idx0 = i + 1
            elif line.startswith("!LIST_NODE"):
                idx1 = i + 1
            elif line.startswith("!LIST_STATE 1"):
                idx2 = i + 1
            elif line.startswith("!NB_ELEM "):
                match = exp.search(line)
                if match:
                    nbelements = int(match.group(1))
            elif line.startswith("!TRS_NAME Outlet"):
                idx3 = i
            elif line.startswith("!END"):
                nend += 1

    return idx0, idx1, idx2, idx3, nbelements, nend, comment

def read_cfmesh(input_cfmesh: Union[str, os.PathLike]) -> Tuple[np.ndarray, ...]:
    """
    Reads and parses a CFmesh formatted MHD input file to extract geometry and plasma variables.

    Args:
        input_cfmesh (str or Path): Path to the CFmesh file.

    Returns:
        Tuple of NumPy arrays containing:
            cell_center_x, cell_center_y, cell_center_z,
            Bx, By, Bz, rho, Vx, Vy, Vz, Pressure, Temperature
    """
    if not os.path.isfile(input_cfmesh):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), input_cfmesh)

    with open(input_cfmesh, "r") as f:
        lines = f.readlines()

    idx0, idx1, idx2, idx3, nbelements, nend, comment = readstruct(lines)

    connectivity = np.loadtxt(lines[idx0:idx0 + nbelements], dtype=int)
    coordinates = np.loadtxt(lines[idx1:idx2 - 1], dtype=float)

    # Compute cell centers from 6-node hexahedra
    node_indices = [connectivity[:, i] for i in range(6)]
    cell_center_x = sum(coordinates[n, 0] for n in node_indices) / 6
    cell_center_y = sum(coordinates[n, 1] for n in node_indices) / 6
    cell_center_z = sum(coordinates[n, 2] for n in node_indices) / 6

    # Initial state data
    bd = comment[-nend - 1][0] + 1
    bf = comment[-nend][0]
    initial_data = np.loadtxt(lines[bd:bf], dtype=float)

    # Convert and scale physical quantities
    Bx = initial_data[:, 4] * 2.2e-4
    By = initial_data[:, 5] * 2.2e-4
    Bz = initial_data[:, 6] * 2.2e-4
    rho = initial_data[:, 0] * 1.67e-13 / 1.67e-27
    Vx = initial_data[:, 1] * 480248.0
    Vy = initial_data[:, 2] * 480248.0
    Vz = initial_data[:, 3] * 480248.0
    Pressure = initial_data[:, 7] * 0.03851
    Temperature = Pressure / (rho * 2 * 1.38e-23)

    return (
        cell_center_x, cell_center_y, cell_center_z,
        Bx, By, Bz, rho, Vx, Vy, Vz, Pressure, Temperature
    )

def find_closest_index_cf(x: np.ndarray, y: np.ndarray, z: np.ndarray, target_coord: Tuple[float, float, float]) -> int:
    """
    Find the index of the point closest to a given target in 3D Cartesian coordinates.

    Args:
        x (np.ndarray): X coordinates of the mesh points.
        y (np.ndarray): Y coordinates of the mesh points.
        z (np.ndarray): Z coordinates of the mesh points.
        target_coord (Tuple[float, float, float]): Target point in Cartesian coordinates (x, y, z).

    Returns:
        int: Index of the closest point in the mesh.
    """
    target = np.array(target_coord)
    points = np.stack((x, y, z), axis=1)
    distances = np.linalg.norm(points - target, axis=1)
    return np.argmin(distances)

def _spherical_unit_vectors_from_cartesian(pos: Tuple[float, float, float]):
    """Return spherical basis (r̂, θ̂, φ̂) at a Cartesian position.

    Notes
    -----
    - θ (colatitude) is measured from +Z (0 at north pole, π/2 at equator).
    - φ (longitude) is measured in the XY-plane from +X toward +Y via atan2(y, x).
    - Only *direction* matters for the basis; pos can be in solar radii (R☉).
    """
    x, y, z = pos
    r = np.sqrt(x*x + y*y + z*z)
    if r == 0:
        # Degenerate; return canonical basis to avoid NaNs
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])

    # Angles
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))  # colatitude
    phi = np.arctan2(y, x)                        # longitude

    # Unit vectors in Cartesian components
    r_hat = np.array([np.sin(theta) * np.cos(phi),
                      np.sin(theta) * np.sin(phi),
                      np.cos(theta)])

    theta_hat = np.array([np.cos(theta) * np.cos(phi),
                          np.cos(theta) * np.sin(phi),
                          -np.sin(theta)])

    phi_hat = np.array([-np.sin(phi),
                        np.cos(phi),
                        0.0])

    return r_hat, theta_hat, phi_hat


def create_hdf_from_cfmesh(
    satellite_positions: Dict[str, Tuple[float, float, float]],
    folder_patterns: Dict[str, str],
    output_dir: Path,
    delta_t: float,
    time_step: float,
    method: str = "nearest",
) -> None:
    """Extract plasma quantities from CFmesh files and save time series (incl. spherical components).

    Parameters
    ----------
    satellite_positions : Dict[str, Tuple[float, float, float]]
        Satellite positions in Cartesian coordinates (x, y, z) **in solar radii (R☉)**.
    folder_patterns : Dict[str, str]
        Mapping of case names to glob patterns of CFmesh files.
    output_dir : Path
        Directory to store output HDF5 files.
    delta_t : float
        Base time step duration (code units).
    time_step : float
        Output time step multiplier.
    method : str, optional
        'nearest' to extract at closest mesh node, or 'interpolate' to RBF-interpolate
        the fields at the satellite position.

    Produces
    --------
    For each case, an HDF5 file with datasets:
        Bx, By, Bz, Vx, Vy, Vz, rho, T, P, time,
        vr, vclt, vlon, br, bclt, blon

    Notes
    -----
    - Spherical components follow (r, θ, φ) with θ the **colatitude**.
    - The time array uses the user's original scaling ``i * delta_t * time_step * 0.402``.
    """

    output_dir.mkdir(exist_ok=True, parents=True)

    for sat_name, sat_coord in satellite_positions.items():
        logger.info(f"Processing satellite: {sat_name}")

        # Use first file of the first case to get reference mesh (for 'nearest')
        first_case = next(iter(folder_patterns))
        first_files = natsort.natsorted(glob.glob(folder_patterns[first_case]))
        if not first_files:
            raise FileNotFoundError(f"No files match pattern for first case '{first_case}': {folder_patterns[first_case]}")
        first_file = first_files[0]

        x, y, z, *_ = read_cfmesh(first_file)
        closest_index = find_closest_index_cf(x, y, z, sat_coord) if method == "nearest" else None
        if method == "nearest":
            logger.info(
                f"Closest mesh point (R☉): x={x[closest_index]:.6g} y={y[closest_index]:.6g} z={z[closest_index]:.6g}"
            )

        for case_name, pattern in folder_patterns.items():
            logger.info(f"\tProcessing case: {case_name}")

            Bx_list, By_list, Bz_list = [], [], []
            Vx_list, Vy_list, Vz_list = [], [], []
            rho_list, T_list, P_list, time_list = [], [], [], []

            # New: spherical components
            vr_list, vclt_list, vlon_list = [], [], []
            br_list, bclt_list, blon_list = [], [], []

            files = natsort.natsorted(glob.glob(pattern))
            if not files:
                logger.warning(f"\tNo files for case '{case_name}' with pattern: {pattern}")
                continue

            for i, file in enumerate(files):
                logger.info(f"\t\tReading {file}")
                x, y, z, Bx, By, Bz, rho, Vx, Vy, Vz, P, T = read_cfmesh(file)

                # Selection / interpolation
                if method == "nearest":
                    idx = closest_index
                    bx, by, bz = float(Bx[idx]), float(By[idx]), float(Bz[idx])
                    vx, vy, vz = float(Vx[idx]), float(Vy[idx]), float(Vz[idx])
                    rho_val, t_val, p_val = float(rho[idx]), float(T[idx]), float(P[idx])
                    pos = (float(x[idx]), float(y[idx]), float(z[idx]))

                elif method == "interpolate":
                    grid = np.column_stack((x, y, z))
                    sat = np.asarray(sat_coord)[None, :]
                    bx = float(RBFInterpolator(grid, Bx, kernel='linear', neighbors=20)(sat)[0])
                    by = float(RBFInterpolator(grid, By, kernel='linear', neighbors=20)(sat)[0])
                    bz = float(RBFInterpolator(grid, Bz, kernel='linear', neighbors=20)(sat)[0])
                    vx = float(RBFInterpolator(grid, Vx, kernel='linear', neighbors=20)(sat)[0])
                    vy = float(RBFInterpolator(grid, Vy, kernel='linear', neighbors=20)(sat)[0])
                    vz = float(RBFInterpolator(grid, Vz, kernel='linear', neighbors=20)(sat)[0])
                    rho_val = float(RBFInterpolator(grid, rho, kernel='linear', neighbors=20)(sat)[0])
                    t_val = float(RBFInterpolator(grid, T, kernel='linear', neighbors=20)(sat)[0])
                    p_val = float(RBFInterpolator(grid, P, kernel='linear', neighbors=20)(sat)[0])
                    pos = tuple(map(float, sat_coord))
                else:
                    raise ValueError("method must be 'nearest' or 'interpolate'")

                # Append Cartesian & scalar quantities
                Bx_list.append(bx); By_list.append(by); Bz_list.append(bz)
                Vx_list.append(vx); Vy_list.append(vy); Vz_list.append(vz)
                rho_list.append(rho_val); T_list.append(t_val); P_list.append(p_val)
                time_list.append(i * delta_t * time_step * 0.402)

                # Compute spherical components at 'pos' (R☉), project V and B
                r_hat, theta_hat, phi_hat = _spherical_unit_vectors_from_cartesian(pos)
                v_vec = np.array([vx, vy, vz], dtype=float)
                b_vec = np.array([bx, by, bz], dtype=float)

                vr = float(v_vec @ r_hat)
                v_theta = float(v_vec @ theta_hat)
                v_phi = float(v_vec @ phi_hat)
                br = float(b_vec @ r_hat)
                b_theta = float(b_vec @ theta_hat)
                b_phi = float(b_vec @ phi_hat)

                # Store using requested names: vr, vclt(θ), vlon(φ); br, bclt(θ), blon(φ)
                vr_list.append(vr)
                vclt_list.append(v_theta)
                vlon_list.append(v_phi)
                br_list.append(br)
                bclt_list.append(b_theta)
                blon_list.append(b_phi)

            # Write HDF5 for this case & satellite
            if files:
                # Include satellite name in file to avoid overwriting between satellites
                output_file = output_dir / f"CFData_{case_name}.hdf5"
                with h5py.File(output_file, 'w') as hdf:
                    hdf.create_dataset('Bx', data=np.asarray(Bx_list))
                    hdf.create_dataset('By', data=np.asarray(By_list))
                    hdf.create_dataset('Bz', data=np.asarray(Bz_list))
                    hdf.create_dataset('Vx', data=np.asarray(Vx_list))
                    hdf.create_dataset('Vy', data=np.asarray(Vy_list))
                    hdf.create_dataset('Vz', data=np.asarray(Vz_list))
                    hdf.create_dataset('rho', data=np.asarray(rho_list))
                    hdf.create_dataset('T', data=np.asarray(T_list))
                    hdf.create_dataset('P', data=np.asarray(P_list))
                    hdf.create_dataset('time', data=np.asarray(time_list))

                    # New spherical components
                    hdf.create_dataset('vr', data=np.asarray(vr_list))
                    hdf.create_dataset('vclt', data=np.asarray(vclt_list))
                    hdf.create_dataset('vlon', data=np.asarray(vlon_list))
                    hdf.create_dataset('br', data=np.asarray(br_list))
                    hdf.create_dataset('bclt', data=np.asarray(bclt_list))
                    hdf.create_dataset('blon', data=np.asarray(blon_list))

                logger.info(f"\tSaved data to {output_file}")


def find_closest_indices(clt: np.ndarray, lon: np.ndarray, target_clt: float, target_lon: float) -> Tuple[int, int]:
    """Finds the indices in colatitude and longitude grids closest to the desired values (in degrees).

    Args:
        clt (np.ndarray): Array of colatitudes (degrees).
        lon (np.ndarray): Array of longitudes (degrees).
        target_clt (float): Target colatitude in degrees.
        target_lon (float): Target longitude in degrees.

    Returns:
        Tuple[int, int]: Indices for colatitude and longitude.
    """
    clt_idx = np.argmin(np.abs(clt - target_clt))
    lon_idx = np.argmin(np.abs(lon - target_lon))
    return clt_idx, lon_idx


def create_hdf_from_dat(
    folder_patterns: Dict[str, str],
    output_dir: Path,
    target_clt: float,
    target_lon: float
) -> None:
    """Creates HDF5 files from .dat files containing spherical grid data.

    Args:
        folder_patterns (Dict[str, str]): Dictionary of simulation name -> glob pattern for dat files.
        output_dir (Path): Output directory for HDF5 files.
        target_clt (float): Desired colatitude in degrees.
        target_lon (float): Desired longitude in degrees.

    Returns:
        None
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine closest indices from first file
    first_case = next(iter(folder_patterns))
    first_file = natsort.natsorted(glob.glob(folder_patterns[first_case]))[0]
    _, clt_ref, lon_ref, *_ = read_dat(first_file)
    clt_idx, lon_idx = find_closest_indices(clt_ref, lon_ref, target_clt, target_lon)
    logger.info(f"Closest indices - colat: {clt_idx}, lon: {lon_idx}")

    for case_name, pattern in folder_patterns.items():
        logger.info(f"Processing case: {case_name}")
        files = natsort.natsorted(glob.glob(pattern))

        date_l, clt_l, lon_l = [], [], []
        vr_l, vlon_l, vclt_l = [], [], []
        density_l, br_l, bclt_l, blon_l, temp_l = [], [], [], [], []

        for i, file in enumerate(files):
            logger.info(f"\tReading file: {file}")
            date, clt, lon, vr, vlon, vclt, density, br, bclt, blon, temp = read_dat(file)
            date_l.append(datetime.fromisoformat(date).timestamp())
            clt_l.append(clt[clt_idx])
            lon_l.append(lon[lon_idx])
            vr_l.append(vr[lon_idx][clt_idx])
            vlon_l.append(vlon[lon_idx][clt_idx])
            vclt_l.append(vclt[lon_idx][clt_idx])
            density_l.append(density[lon_idx][clt_idx])
            br_l.append(br[lon_idx][clt_idx])
            bclt_l.append(bclt[lon_idx][clt_idx])
            blon_l.append(blon[lon_idx][clt_idx])
            temp_l.append(temp[lon_idx][clt_idx])

        output_file = output_dir / f"DatData_{case_name}.hdf5"
        with h5py.File(output_file, 'w') as hdf:
            hdf.create_dataset('date', data=np.array(date_l))
            hdf.create_dataset('clt', data=np.array(clt_l))
            hdf.create_dataset('lon', data=np.array(lon_l))
            hdf.create_dataset('vr', data=np.array(vr_l))
            hdf.create_dataset('vclt', data=np.array(vclt_l))
            hdf.create_dataset('vlon', data=np.array(vlon_l))
            hdf.create_dataset('br', data=np.array(br_l))
            hdf.create_dataset('bclt', data=np.array(bclt_l))
            hdf.create_dataset('blon', data=np.array(blon_l))
            hdf.create_dataset('density', data=np.array(density_l))
            hdf.create_dataset('temperature', data=np.array(temp_l))

        logger.info(f"\tSaved data to {output_file}")

if __name__ == "__main__":
    # Example test satellite in Cartesian coordinates or spherical for .dat
    satellite_cartesian = {"STA": [-10.0, 0.0, 0.0]}  # For CFMesh or VTU
    satellite_spherical = {"STA": (90.0, 0.0)}        # For .dat (colatitude=90°, longitude=0°)

    delta_t = 0.005
    time_step = 20
    output_dir = Path("test_hdf_output")

    # --- Test 1: VTU ---
    vtu_folders = {
        "COCONUT_A": "test_data/vtu/caseA/*.vtu",
        "COCONUT_B": "test_data/vtu/caseB/*.vtu",
    }
    create_hdf_from_vtu(satellite_cartesian, vtu_folders, output_dir / "vtu", delta_t, time_step)

    # --- Test 2: CFMesh ---
    cfmesh_cases = {
        "CFMESH_A": "test_data/cfmesh/caseA/*.CFmesh",
        "CFMESH_B": "test_data/cfmesh/caseB/*.CFmesh"
    }
    create_hdf_from_cfmesh(satellite_cartesian, cfmesh_cases, output_dir / "cfmesh", delta_t, time_step)

    # --- Test 3: DAT ---
    dat_cases = {
        "TOMO_A": "test_data/dat/caseA/*.dat",
        "TOMO_B": "test_data/dat/caseB/*.dat",
    }
    create_hdf_from_dat(satellite_spherical, dat_cases, output_dir / "dat")