"""
Module for generating EUHFORIA-compatible solar wind boundary files from COCONUT outputs.

This module processes CFmesh plasma simulation output, interpolates it onto a heliospheric boundary grid,
and formats the result into `.dat` files usable as input for heliospheric simulations like EUHFORIA.

Main tasks:
- Load and interpret CFmesh geometry and field data,
- Interpolate plasma and magnetic field variables onto a spherical grid,
- Optionally rotate the grid to match Carrington/Stonyhurst frames,
- Save results in the required EUHFORIA `.dat` format.

Typically used as a post-processing step after a COCONUT coronal simulation.
"""

import re
import glob
import os
import errno
import math
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
from scipy.interpolate import RBFInterpolator
from coconut_tools.rotation_angle import compute_rotation_angle
from coconut_tools.logger_config import setup_logger

logger = setup_logger(__name__)


def readstruct(lines: List[str]) -> Tuple[int, int, int, int, int, int, List[Tuple[int, str]]]:
    """Reads the structure of a CFmesh file.

    Args:
        lines (List[str]): Lines of the CFmesh file.

    Returns:
        Tuple[int, int, int, int, int, int, List[Tuple[int, str]]]: Indices of node, state, element, outlet, number of elements, number of !END markers, and list of comments.
    """
    exp = re.compile(r'!NB_ELEM (-?\d+)')
    nbelements, idx0, idx1, idx2, idx3, nend = 0, 0, 0, 0, 0, 0
    comment = []

    for i, line in enumerate(lines):
        if line.startswith("!"):
            comment.append((i, line))
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

def extract_number(file_name: str) -> int:
    """Extracts the numerical index from a CFmesh filename.

    Args:
        file_name (str): The filename to extract the number from.

    Returns:
        int: Extracted number or 0 if not found.
    """
    match = re.search(r'corona-iter_(\d+)\.CFmesh$', file_name)
    return int(match.group(1)) if match else 0
def create_boundary_fromcfmesh(inputfile: str, time: str, rad_out: float, nb_th: int, nb_phi: int, eps: float, output_dat: str, full_output: bool = True) -> None:
    """Creates a boundary file by interpolating CFmesh volume data onto a spherical grid.

    Args:
        inputfile (str): Path to the input CFmesh file.
        time (str): Timestamp string in ISO format.
        rad_out (float): Target radius for the boundary.
        nb_th (int): Number of colatitude grid points.
        nb_phi (int): Number of longitude grid points.
        eps (float): Small angle offset to avoid poles.
        output_dat (str): Output file path.
        full_output (bool): Whether to include all variables or a reduced subset.
    """
    if not os.path.isfile(inputfile):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), inputfile)

    with open(inputfile, "r") as f:
        lines = f.readlines()

    idx0, idx1, idx2, idx3, nbelements, nend, comment = readstruct(lines)

    connectivity = np.loadtxt(lines[idx0:idx0 + nbelements], dtype=int)
    coordinates = np.loadtxt(lines[idx1:idx2 - 1], dtype=float)

    nodes = connectivity[:, :6]
    centers = coordinates[nodes].mean(axis=1)
    r = np.linalg.norm(centers, axis=1)

    r_mask = (r > rad_out * 0.95) & (r < rad_out * 1.05)
    mask = np.where(r_mask)[0]

    LAT = np.linspace(eps, np.pi - eps, nb_th)
    LON = np.linspace(eps, 2.0 * np.pi - eps, nb_phi)
    lat, lon = np.meshgrid(LAT, LON, indexing='ij')

    r_def = rad_out
    x_sph = r_def * np.sin(lat) * np.cos(lon)
    y_sph = r_def * np.sin(lat) * np.sin(lon)
    z_sph = r_def * np.cos(lat)

    grid = np.column_stack((x_sph.flatten(), y_sph.flatten(), z_sph.flatten()))

    bd = comment[-nend - 1][0] + 1
    bf = comment[-nend][0]
    Initialdata = np.loadtxt(lines[bd:bf], dtype=np.float64)

    rho0 = Initialdata[mask, 0] * 1.67e-13 / 1.67e-27
    Vx0 = Initialdata[mask, 1] * 480248.0
    Vy0 = Initialdata[mask, 2] * 480248.0
    Vz0 = Initialdata[mask, 3] * 480248.0
    Bx = Initialdata[mask, 4] * 2.2e-4
    By = Initialdata[mask, 5] * 2.2e-4
    Bz = Initialdata[mask, 6] * 2.2e-4
    Pressure = Initialdata[mask, 7] * 0.03851
    temp = Pressure / rho0 / 2 / 1.38e-23

    x, y, z = centers[mask].T
    r_bis = np.hypot(x, y)
    EPSILON = 1e-8

    vr = (x * Vx0 + y * Vy0 + z * Vz0) / r[mask]
    vlon = (-y * Vx0 + x * Vy0) / (r_bis + EPSILON)
    vclt = (x * z * Vx0 + y * z * Vy0 - (r_bis + EPSILON) * Vz0) / (r[mask] * (r_bis + EPSILON))
    br = (x * Bx + y * By + z * Bz) / r[mask]
    blon = (-y * Bx + x * By) / (r_bis + EPSILON)
    bclt = (x * z * Bx + y * z * By - (r_bis + EPSILON) * Bz) / (r[mask] * (r_bis + EPSILON))

    interp_fields = {
        'vr': vr,
        'vp': vlon,
        'vt': vclt,
        'number_density': rho0,
        'temperature': temp,
        'Br': br,
        'Bp': blon,
        'Bt': bclt
    }

    selected_keys = ['vr', 'vp', 'vt', 'number_density', 'temperature', 'Br', 'Bp', 'Bt'] if full_output else ['vr', 'number_density', 'temperature', 'Br']

    coords = np.column_stack((x, y, z))
    coords_unique, indices_unique = np.unique(coords, axis=0, return_index=True)

    interpolated = {}
    for key in selected_keys:
        interpolator = RBFInterpolator(coords_unique, interp_fields[key][indices_unique], kernel='linear', neighbors=50)
        interp_vals = interpolator(grid).reshape(nb_th, nb_phi)
        interpolated[key] = interp_vals

    with open(output_dat, 'w') as f:
        f.write(f'Time:\n{time}\nRadius of sphere:\n14959787070.0\n')
        f.write(f'Number of colatitude grid points:\n{nb_th}\nColatitude grid points:\n')
        f.write('\n'.join(f'{v:.19e}' for v in LAT) + '\n')
        f.write(f'Number of longitude grid points:\n{nb_phi}\nLongitude grid points:\n')
        f.write('\n'.join(f'{v:.19e}' for v in LON) + '\n')

        for key in selected_keys:
            f.write(f'{key}\n')
            f.write('\n'.join(
                '\n'.join(f'{interpolated[key][j, k]:.19e}' for j in range(nb_th))
                for k in range(nb_phi)) + '\n')

def rotation(input_dat: str, output_dat: str, angle_degrees: float, full_output: bool = True) -> None:
    """Rotates the longitude data in a boundary file and adjusts selected fields accordingly.

    Args:
        input_dat (str): Path to the input data file.
        output_dat (str): Path to the output data file.
        angle_degrees (float): Rotation angle in degrees.
        full_output (bool): Whether to include all variables (True) or only vr, number_density, temperature, Br (False).
    """
    with open(input_dat) as f:
        lines = f.read().splitlines()

    date = lines[1]
    clt_start = lines.index('Colatitude grid points:')
    lon_start = lines.index('Longitude grid points:')

    full_keys = ['vr', 'vp', 'vt', 'number_density', 'temperature', 'Br', 'Bp', 'Bt']
    reduced_keys = ['vr', 'number_density', 'temperature', 'Br']
    keys = full_keys if full_output else reduced_keys

    indices = {key: lines.index(key) for key in full_keys}

    clt = [float(x) for x in lines[clt_start + 1:lon_start - 2]]
    lon_before = [float(x) + np.radians(angle_degrees) for x in lines[lon_start + 1:indices['vr']]]
    lon = [math.fmod(angle, 2 * math.pi) for angle in lon_before]
    min_index = np.argmin(lon)
    lon = np.roll(lon, -min_index)

    def shift_data(key):
        all_keys = full_keys
        next_key_index = indices[all_keys[all_keys.index(key)+1]] if key != 'Bt' else None
        data = [float(x) for x in lines[indices[key] + 1:next_key_index]] if next_key_index else [float(x) for x in lines[indices[key] + 1:]]
        return np.roll(data, -min_index * len(clt))

    data_arrays = {key: shift_data(key) for key in keys}

    with open(output_dat, 'w') as fp:
        fp.write(f"Time:\n{date}\nRadius of sphere:\n14959787070.0\n")
        fp.write(f"Number of colatitude grid points:\n{len(clt)}\nColatitude grid points:\n")
        np.savetxt(fp, clt)
        fp.write(f"Number of longitude grid points:\n{len(lon)}\nLongitude grid points:\n")
        np.savetxt(fp, lon)
        for name in keys:
            fp.write(f"{name}\n")
            np.savetxt(fp, data_arrays[name])

if __name__ == "__main__":

    """ Example usage of the module to process a series of CFmesh files and generate boundary .dat files.
    """
    

    magnetogram_path = "E:/GU V2/magnetogram/mrzqs170404t1814c2189_268.fits"
    date_hmi = None
    auto_compute_rotation = True
    manual_rotation_angle = 180.0  # fallback if auto_compute_rotation is False

    files = glob.glob('E:/coconut_cme/coriolis/fromfullmhd/CFmesh/*.CFmesh')
    files = sorted(files, key=extract_number)

    angle, date_dt = compute_rotation_angle(magnetogram_path, date_hmi)
    print(angle, date_dt)

    first_time = date_dt.strftime("%Y-%m-%dT%H:%M:%S")
    rad_out = 21.5
    nb_th = 180
    nb_phi = 360
    eps = 0.01
    full_output = True  # <--- change this to True for full variable set

    """
    for i, file in enumerate(files):
        logger.info(f"Processing file {i}: {file}")
        date_dt = datetime.strptime(first_time, '%Y-%m-%dT%H:%M:%S')
        #date_i = date_dt + timedelta(hours=20 * i * 0.005 * 0.402)
        date_i = date_dt + timedelta(seconds=i*145)
        timestamp_i = str(int(date_i.timestamp()))

        output_dat_temp = f'E:/coconut_cme/coriolis/fromfullmhd/dat/solar_wind_boundary_{timestamp_i}_temps.dat'
        output_dat = f'E:/coconut_cme/coriolis/fromfullmhd/dat/solar_wind_boundary_{timestamp_i}.dat'

        time = date_i.strftime("%Y-%m-%dT%H:%M:%S")

        if os.path.exists(output_dat):
            logger.info(f"Output already exists, skipping: {output_dat}")
            continue

        create_boundary_fromcfmesh(file, time, rad_out, nb_th, nb_phi, eps, output_dat_temp, full_output=full_output)
        rotation(output_dat_temp, output_dat, angle, full_output=full_output)
        os.remove(output_dat_temp)
    """

    ########################################################################

    
    file = 'E:/GU V2/coconut_result_coriolis/result_2017-04-04_fullmhd/corona.CFmesh'
    logger.info(f"Processing file {file}")

    magnetogram_path = "E:/GU V2/magnetogram/mrzqs170404t1814c2189_268.fits.gz"

    angle, date_dt = compute_rotation_angle(magnetogram_path, date_hmi)
    logger.info(f"Computed rotation angle: {angle} degrees")
    logger.info(f"Using date for rotation computation: {date_dt.isoformat()}\n")

    timestamp_i = str(int(date_dt.timestamp()))
    output_dat_temp = f'E:/GU V2/coconut_result_coriolis/result_2017-04-04_fullmhd/dat/solar_wind_boundary_{timestamp_i}_temps.dat'
    output_dat = f'E:/GU V2/coconut_result_coriolis/result_2017-04-04_fullmhd/dat/solar_wind_boundary_{timestamp_i}.dat'

    time = date_dt.strftime("%Y-%m-%dT%H:%M:%S")

    create_boundary_fromcfmesh(file, time, rad_out, nb_th, nb_phi, eps, output_dat_temp, full_output=full_output)
    rotation(output_dat_temp, output_dat, angle, full_output=full_output)
    os.remove(output_dat_temp)

    ########################################################################

    file = 'E:/GU V2/coconut_result/result_2017-04-04/corona.CFmesh'
    logger.info(f"Processing file {file}")

    magnetogram_path = "E:/GU V2/magnetogram/mrzqs170404t1814c2189_268.fits.gz"

    angle, date_dt = compute_rotation_angle(magnetogram_path, date_hmi)
    logger.info(f"Computed rotation angle: {angle} degrees")
    logger.info(f"Using date for rotation computation: {date_dt.isoformat()}\n")

    timestamp_i = str(int(date_dt.timestamp()))
    output_dat_temp = f'E:/GU V2/coconut_result/result_2017-04-04/dat/solar_wind_boundary_{timestamp_i}_temps.dat'
    output_dat = f'E:/GU V2/coconut_result/result_2017-04-04/dat/solar_wind_boundary_{timestamp_i}.dat'

    time = date_dt.strftime("%Y-%m-%dT%H:%M:%S")

    create_boundary_fromcfmesh(file, time, rad_out, nb_th, nb_phi, eps, output_dat_temp, full_output=full_output)
    rotation(output_dat_temp, output_dat, angle, full_output=full_output)
    os.remove(output_dat_temp)

    ########################################################################

    file = 'E:/GU V2/coconut_result_coriolis/result_2011-09-04_fullmhd/corona.CFmesh'
    logger.info(f"Processing file {file}")

    magnetogram_path = "E:/GU V2/magnetogram/mrzqs110904t1154c2114_180.fits.gz"

    angle, date_dt = compute_rotation_angle(magnetogram_path, date_hmi)
    logger.info(f"Computed rotation angle: {angle} degrees")
    logger.info(f"Using date for rotation computation: {date_dt.isoformat()}\n")

    timestamp_i = str(int(date_dt.timestamp()))
    output_dat_temp = f'E:/GU V2/coconut_result_coriolis/result_2011-09-04_fullmhd/dat/solar_wind_boundary_{timestamp_i}_temps.dat'
    output_dat = f'E:/GU V2/coconut_result_coriolis/result_2011-09-04_fullmhd/dat/solar_wind_boundary_{timestamp_i}.dat'

    time = date_dt.strftime("%Y-%m-%dT%H:%M:%S")

    create_boundary_fromcfmesh(file, time, rad_out, nb_th, nb_phi, eps, output_dat_temp, full_output=full_output)
    rotation(output_dat_temp, output_dat, angle, full_output=full_output)
    os.remove(output_dat_temp)

    ########################################################################

    file = 'E:/GU V2/coconut_result_coriolis/result_2011-09-24_fullmhd/corona.CFmesh'
    logger.info(f"Processing file {file}")

    magnetogram_path = "E:/GU V2/magnetogram/mrzqs110924t1154c2115_276.fits.gz"

    angle, date_dt = compute_rotation_angle(magnetogram_path, date_hmi)
    logger.info(f"Computed rotation angle: {angle} degrees")
    logger.info(f"Using date for rotation computation: {date_dt.isoformat()}\n")

    timestamp_i = str(int(date_dt.timestamp()))
    output_dat_temp = f'E:/GU V2/coconut_result_coriolis/result_2011-09-24_fullmhd/dat/solar_wind_boundary_{timestamp_i}_temps.dat'
    output_dat = f'E:/GU V2/coconut_result_coriolis/result_2011-09-24_fullmhd/dat/solar_wind_boundary_{timestamp_i}.dat'

    time = date_dt.strftime("%Y-%m-%dT%H:%M:%S")

    create_boundary_fromcfmesh(file, time, rad_out, nb_th, nb_phi, eps, output_dat_temp, full_output=full_output)
    rotation(output_dat_temp, output_dat, angle, full_output=full_output)
    os.remove(output_dat_temp)

    ########################################################################

    
    file = 'E:/GU V2/coconut_result_coriolis/result_2012-03-06_fullmhd/corona.CFmesh'
    logger.info(f"Processing file {file}")

    magnetogram_path = "E:/GU V2/magnetogram/mrzqs120306t2354c2121_269.fits.gz"

    angle, date_dt = compute_rotation_angle(magnetogram_path, date_hmi)
    logger.info(f"Computed rotation angle: {angle} degrees")
    logger.info(f"Using date for rotation computation: {date_dt.isoformat()}\n")

    timestamp_i = str(int(date_dt.timestamp()))
    output_dat_temp = f'E:/GU V2/coconut_result_coriolis/result_2012-03-06_fullmhd/dat/solar_wind_boundary_{timestamp_i}_temps.dat'
    output_dat = f'E:/GU V2/coconut_result_coriolis/result_2012-03-06_fullmhd/dat/solar_wind_boundary_{timestamp_i}.dat'

    time = date_dt.strftime("%Y-%m-%dT%H:%M:%S")

    create_boundary_fromcfmesh(file, time, rad_out, nb_th, nb_phi, eps, output_dat_temp, full_output=full_output)
    rotation(output_dat_temp, output_dat, angle, full_output=full_output)
    os.remove(output_dat_temp)

    ########################################################################

    
    file = 'E:/GU V2/coconut_result_coriolis/result_2012-03-07_fullmhd/corona.CFmesh'
    logger.info(f"Processing file {file}")

    magnetogram_path = "E:/GU V2/magnetogram/mrzqs120307t1154c2121_262.fits.gz"

    angle, date_dt = compute_rotation_angle(magnetogram_path, date_hmi)
    logger.info(f"Computed rotation angle: {angle} degrees")
    logger.info(f"Using date for rotation computation: {date_dt.isoformat()}\n")

    timestamp_i = str(int(date_dt.timestamp()))
    output_dat_temp = f'E:/GU V2/coconut_result_coriolis/result_2012-03-07_fullmhd/dat/solar_wind_boundary_{timestamp_i}_temps.dat'
    output_dat = f'E:/GU V2/coconut_result_coriolis/result_2012-03-07_fullmhd/dat/solar_wind_boundary_{timestamp_i}.dat'

    time = date_dt.strftime("%Y-%m-%dT%H:%M:%S")

    create_boundary_fromcfmesh(file, time, rad_out, nb_th, nb_phi, eps, output_dat_temp, full_output=full_output)
    rotation(output_dat_temp, output_dat, angle, full_output=full_output)
    os.remove(output_dat_temp)

    ########################################################################

    file = 'E:/GU V2/coconut_result_coriolis/result_2012-05-10_fullmhd/corona.CFmesh'
    logger.info(f"Processing file {file}")

    magnetogram_path = "E:/GU V2/magnetogram/mrzqs120510t2354c2123_131.fits.gz"

    angle, date_dt = compute_rotation_angle(magnetogram_path, date_hmi)
    logger.info(f"Computed rotation angle: {angle} degrees")
    logger.info(f"Using date for rotation computation: {date_dt.isoformat()}\n")

    timestamp_i = str(int(date_dt.timestamp()))
    output_dat_temp = f'E:/GU V2/coconut_result_coriolis/result_2012-05-10_fullmhd/dat/solar_wind_boundary_{timestamp_i}_temps.dat'
    output_dat = f'E:/GU V2/coconut_result_coriolis/result_2012-05-10_fullmhd/dat/solar_wind_boundary_{timestamp_i}.dat'

    time = date_dt.strftime("%Y-%m-%dT%H:%M:%S")

    create_boundary_fromcfmesh(file, time, rad_out, nb_th, nb_phi, eps, output_dat_temp, full_output=full_output)
    rotation(output_dat_temp, output_dat, angle, full_output=full_output)
    os.remove(output_dat_temp)

    ########################################################################

    file = 'E:/GU V2/coconut_result_coriolis/result_2012-07-09_fullmhd/corona.CFmesh'
    logger.info(f"Processing file {file}")

    magnetogram_path = "E:/GU V2/magnetogram/mrzqs120709t2354c2125_057.fits.gz"

    angle, date_dt = compute_rotation_angle(magnetogram_path, date_hmi)
    logger.info(f"Computed rotation angle: {angle} degrees")
    logger.info(f"Using date for rotation computation: {date_dt.isoformat()}\n")

    timestamp_i = str(int(date_dt.timestamp()))
    output_dat_temp = f'E:/GU V2/coconut_result_coriolis/result_2012-07-09_fullmhd/dat/solar_wind_boundary_{timestamp_i}_temps.dat'
    output_dat = f'E:/GU V2/coconut_result_coriolis/result_2012-07-09_fullmhd/dat/solar_wind_boundary_{timestamp_i}.dat'

    time = date_dt.strftime("%Y-%m-%dT%H:%M:%S")

    create_boundary_fromcfmesh(file, time, rad_out, nb_th, nb_phi, eps, output_dat_temp, full_output=full_output)
    rotation(output_dat_temp, output_dat, angle, full_output=full_output)
    os.remove(output_dat_temp)

    ########################################################################

    file = 'E:/GU V2/coconut_result_coriolis/result_2012-09-23_fullmhd/corona.CFmesh'
    logger.info(f"Processing file {file}")

    magnetogram_path = "E:/GU V2/magnetogram/mrzqs120923t0544c2128_142.fits.gz"

    angle, date_dt = compute_rotation_angle(magnetogram_path, date_hmi)
    logger.info(f"Computed rotation angle: {angle} degrees")
    logger.info(f"Using date for rotation computation: {date_dt.isoformat()}\n")

    timestamp_i = str(int(date_dt.timestamp()))
    output_dat_temp = f'E:/GU V2/coconut_result_coriolis/result_2012-09-23_fullmhd/dat/solar_wind_boundary_{timestamp_i}_temps.dat'
    output_dat = f'E:/GU V2/coconut_result_coriolis/result_2012-09-23_fullmhd/dat/solar_wind_boundary_{timestamp_i}.dat'

    time = date_dt.strftime("%Y-%m-%dT%H:%M:%S")

    create_boundary_fromcfmesh(file, time, rad_out, nb_th, nb_phi, eps, output_dat_temp, full_output=full_output)
    rotation(output_dat_temp, output_dat, angle, full_output=full_output)
    os.remove(output_dat_temp)

    ########################################################################
    file = 'E:/GU V2/coconut_result_coriolis/result_2013-03-13_fullmhd/corona.CFmesh'
    logger.info(f"Processing file {file}")

    magnetogram_path = "E:/GU V2/magnetogram/mrzqs130313t1124c2134_046.fits.gz"

    angle, date_dt = compute_rotation_angle(magnetogram_path, date_hmi)
    logger.info(f"Computed rotation angle: {angle} degrees")
    logger.info(f"Using date for rotation computation: {date_dt.isoformat()}\n")

    timestamp_i = str(int(date_dt.timestamp()))
    output_dat_temp = f'E:/GU V2/coconut_result_coriolis/result_2013-03-13_fullmhd/dat/solar_wind_boundary_{timestamp_i}_temps.dat'
    output_dat = f'E:/GU V2/coconut_result_coriolis/result_2013-03-13_fullmhd/dat/solar_wind_boundary_{timestamp_i}.dat'

    time = date_dt.strftime("%Y-%m-%dT%H:%M:%S")

    create_boundary_fromcfmesh(file, time, rad_out, nb_th, nb_phi, eps, output_dat_temp, full_output=full_output)
    rotation(output_dat_temp, output_dat, angle, full_output=full_output)
    os.remove(output_dat_temp)

    ########################################################################

    file = 'E:/GU V2/coconut_result_coriolis/result_2013-04-08_fullmhd/corona.CFmesh'
    logger.info(f"Processing file {file}")

    magnetogram_path = "E:/GU V2/magnetogram/mrzqs130408t0604c2135_066.fits.gz"

    angle, date_dt = compute_rotation_angle(magnetogram_path, date_hmi)
    logger.info(f"Computed rotation angle: {angle} degrees")
    logger.info(f"Using date for rotation computation: {date_dt.isoformat()}\n")

    timestamp_i = str(int(date_dt.timestamp()))
    output_dat_temp = f'E:/GU V2/coconut_result_coriolis/result_2013-04-08_fullmhd/dat/solar_wind_boundary_{timestamp_i}_temps.dat'
    output_dat = f'E:/GU V2/coconut_result_coriolis/result_2013-04-08_fullmhd/dat/solar_wind_boundary_{timestamp_i}.dat'

    time = date_dt.strftime("%Y-%m-%dT%H:%M:%S")

    create_boundary_fromcfmesh(file, time, rad_out, nb_th, nb_phi, eps, output_dat_temp, full_output=full_output)
    rotation(output_dat_temp, output_dat, angle, full_output=full_output)
    os.remove(output_dat_temp)

    ########################################################################


    file = 'E:/GU V2/coconut_result_coriolis/result_2013-09-28_fullmhd/corona.CFmesh'
    logger.info(f"Processing file {file}")

    magnetogram_path = "E:/GU V2/magnetogram/mrzqs130928t1804c2142_292.fits.gz"

    angle, date_dt = compute_rotation_angle(magnetogram_path, date_hmi)
    logger.info(f"Computed rotation angle: {angle} degrees")
    logger.info(f"Using date for rotation computation: {date_dt.isoformat()}\n")

    timestamp_i = str(int(date_dt.timestamp()))
    output_dat_temp = f'E:/GU V2/coconut_result_coriolis/result_2013-09-28_fullmhd/dat/solar_wind_boundary_{timestamp_i}_temps.dat'
    output_dat = f'E:/GU V2/coconut_result_coriolis/result_2013-09-28_fullmhd/dat/solar_wind_boundary_{timestamp_i}.dat'

    time = date_dt.strftime("%Y-%m-%dT%H:%M:%S")

    create_boundary_fromcfmesh(file, time, rad_out, nb_th, nb_phi, eps, output_dat_temp, full_output=full_output)
    rotation(output_dat_temp, output_dat, angle, full_output=full_output)
    os.remove(output_dat_temp)

    ########################################################################

    file = 'E:/GU V2/coconut_result_coriolis/result_2014-01-04_fullmhd/corona.CFmesh'
    logger.info(f"Processing file {file}")

    magnetogram_path = "E:/GU V2/magnetogram/mrzqs140104t1804c2145_080.fits.gz"

    angle, date_dt = compute_rotation_angle(magnetogram_path, date_hmi)
    logger.info(f"Computed rotation angle: {angle} degrees")
    logger.info(f"Using date for rotation computation: {date_dt.isoformat()}\n")

    timestamp_i = str(int(date_dt.timestamp()))
    output_dat_temp = f'E:/GU V2/coconut_result_coriolis/result_2014-01-04_fullmhd/dat/solar_wind_boundary_{timestamp_i}_temps.dat'
    output_dat = f'E:/GU V2/coconut_result_coriolis/result_2014-01-04_fullmhd/dat/solar_wind_boundary_{timestamp_i}.dat'

    time = date_dt.strftime("%Y-%m-%dT%H:%M:%S")

    create_boundary_fromcfmesh(file, time, rad_out, nb_th, nb_phi, eps, output_dat_temp, full_output=full_output)
    rotation(output_dat_temp, output_dat, angle, full_output=full_output)
    os.remove(output_dat_temp)

    ########################################################################
    
    file = 'E:/GU V2/coconut_result_coriolis/result_2014-09-06_fullmhd/corona.CFmesh'
    logger.info(f"Processing file {file}")

    magnetogram_path = "E:/GU V2/magnetogram/mrzqs140906t1804c2154_085.fits.gz"

    angle, date_dt = compute_rotation_angle(magnetogram_path, date_hmi)
    logger.info(f"Computed rotation angle: {angle} degrees")
    logger.info(f"Using date for rotation computation: {date_dt.isoformat()}\n")

    timestamp_i = str(int(date_dt.timestamp()))
    output_dat_temp = f'E:/GU V2/coconut_result_coriolis/result_2014-09-06_fullmhd/dat/solar_wind_boundary_{timestamp_i}_temps.dat'
    output_dat = f'E:/GU V2/coconut_result_coriolis/result_2014-09-06_fullmhd/dat/solar_wind_boundary_{timestamp_i}.dat'

    time = date_dt.strftime("%Y-%m-%dT%H:%M:%S")

    create_boundary_fromcfmesh(file, time, rad_out, nb_th, nb_phi, eps, output_dat_temp, full_output=full_output)
    rotation(output_dat_temp, output_dat, angle, full_output=full_output)
    os.remove(output_dat_temp)




 
