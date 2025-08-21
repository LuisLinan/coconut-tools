"""Reads physical simulation data from a formatted text file."""
import numpy as np


def read_data(filename, reduced=False):
    """Reads physical simulation data from a formatted text file.

    Args:
        filename (str): Path to the file to be read.
        reduced (bool): Whether to only read vr, density, br, temperature.

    Returns:
        tuple: Contains the parsed data arrays including date, clt, lon, and selected fields.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()

    date = lines[1].strip()

    # Locate headers dynamically
    all_headers = {
        'clt': 'Colatitude grid points:\n',
        'lon': 'Longitude grid points:\n',
        'vr': 'vr\n',
        'vp': 'vp\n',
        'vt': 'vt\n',
        'density': 'number_density\n',
        'temp': 'temperature\n',
        'br': 'Br\n',
        'bp': 'Bp\n',
        'bt': 'Bt\n'
    }

    # Filter headers depending on mode
    if reduced:
        expected_headers = ['clt', 'lon', 'vr', 'density', 'temp', 'br']
    else:
        expected_headers = list(all_headers.keys())

    # Get line indices for existing headers only
    indices = {
        key: lines.index(all_headers[key]) for key in expected_headers if all_headers[key] in lines
    }

    idx_clt = indices['clt'] + 1
    idx_lon = indices['lon'] + 1

    nb_clt = int(lines[idx_clt - 2])
    nb_lon = int(lines[idx_lon - 2])

    clt = np.array([float(lines[idx_clt + i]) for i in range(nb_clt)])
    lon = np.array([float(lines[idx_lon + i]) for i in range(nb_lon)])

    def read_array(start, end):
        return np.array([
            [float(val) for val in line.split()] for line in lines[start:end]
        ]).reshape(nb_lon, nb_clt)

    if reduced:
        vr = read_array(indices['vr'] + 1, indices['density'])
        density = read_array(indices['density'] + 1, indices['temp'])
        temp = read_array(indices['temp'] + 1, indices['br'])
        br = read_array(indices['br'] + 1, len(lines))
        vlon = vclt = blon = bclt = None
    else:
        vr = read_array(indices['vr'] + 1, indices['vp'])
        vlon = read_array(indices['vp'] + 1, indices['vt'])
        vclt = read_array(indices['vt'] + 1, indices['density'])
        density = read_array(indices['density'] + 1, indices['temp'])
        temp = read_array(indices['temp'] + 1, indices['br'])
        br = read_array(indices['br'] + 1, indices['bp'])
        blon = read_array(indices['bp'] + 1, indices['bt'])
        bclt = read_array(indices['bt'] + 1, len(lines))

    return date, clt, lon, vr / 1000, vlon / 1000, vclt / 1000, density, br, bclt, blon, temp