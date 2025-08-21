"""
Utilities for generating and downloading magnetogram input files for COCONUT simulations.

This module supports the generation of COCONUT-compatible magnetic boundary files
based on real solar data from various sources (e.g., WSO, GONG, ADAPT, HMI).

Main capabilities:
- Construct output filenames and download magnetogram data using Sunpy or custom rules,
- Handle spherical harmonic resolution (`lmax`),
- Provide helper functions to parse timestamps and build mesh grids.

Used to prepare magnetic input for running COCONUT coronal MHD simulations.
"""
import os
import sunpy.coordinates.sun
import sunpy.util.net
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import sunpy.coordinates.sun
import sunpy.util.net
from astropy.io import fits
from scipy import interpolate
from scipy import special as scisp
import matplotlib.pyplot as plt
from logger_config import setup_logger

logger = setup_logger(__name__)


def parse_date_from_filename(name, fmt, split_token):
    """Parse date from filename using a given format and delimiter.

    Args:
        name (str): Filename.
        fmt (str): Format string to parse the date.
        split_token (str): Token to split the filename before parsing.

    Returns:
        datetime: Parsed datetime object.
    """
    try:
        return datetime.strptime(name.split(split_token)[0], fmt)
    except Exception as e:
        logger.warning(f"Skipping file {name} due to parsing error: {e}")
        return None

def generate_output_and_map_names(date, map_type, output_dir, lmax, method_used):
    """Generate output filename and download magnetogram file based on map type.

    Args:
        date (str): Date string in ISO format (e.g., '2020-12-07T15:00:00').
        map_type (str): Map type ('WSO', 'GONG', 'ADAPT', 'HMI').
        output_dir (str): Directory to store output file.
        lmax (int): Spherical harmonic maximum degree.
        method_used (str) : method used for filtering ( e.g NLD )

    Returns:
        tuple: (output_name (str), local_file (str))
    """
    date_datetime = datetime.fromisoformat(date)
    year, month, day = date_datetime.year, date_datetime.month, date_datetime.day

    cr_number = int(sunpy.coordinates.sun.carrington_rotation_number(date_datetime))
    logger.info(f"Carrington rotation number: {cr_number}")

    output_name = {
        'WSO': f"{output_dir}map_wso_lmax{lmax}_cr{cr_number}_{method_used}.dat",
        'GONG': f"{output_dir}map_gong_lmax{lmax}_{date[:10]}_{method_used}.dat",
        'ADAPT': f"{output_dir}map_adapt_lmax{lmax}_{date[:10]}_{method_used}.dat",
        'HMI_polfil': f"{output_dir}map_hmi_polfil_lmax{lmax}_cr{cr_number}_{method_used}.dat",
        'HMI_small': f"{output_dir}map_hmi_small_lmax{lmax}_cr{cr_number}_{method_used}.dat",
    }.get(map_type)

    if not output_name:
        raise ValueError(f"Unsupported map_type: {map_type}")

    if map_type == 'WSO':
        map_name = f'WSO.{cr_number}.txt'
        remote_file = f"http://wso.stanford.edu/synoptic/{map_name}"

    elif map_type == 'GONG':
        file_id = 'mrzqs'
        remote_dir = f"https://gong.nso.edu/data/magmap/QR/{file_id[2:]}/{year}{month:02d}/{file_id}{str(year)[2:]}{month:02d}{day:02d}/"
        soup = BeautifulSoup(requests.get(remote_dir).text, "html.parser")
        file_names = [a.get("href") for a in soup.find_all("a") if file_id in a.get("href")]

        time_deltas = []
        for name in file_names:
            parsed_date = parse_date_from_filename(name, file_id + "%y%m%dt%H%M", "c")
            if parsed_date:
                delta = (parsed_date - date_datetime).total_seconds()
                time_deltas.append(delta)
            else:
                time_deltas.append(np.inf)

        if not time_deltas or all(np.isinf(time_deltas)):
            raise RuntimeError("No valid GONG file found on the remote server.")

        map_name = file_names[np.argmin(np.abs(time_deltas))]
        remote_file = remote_dir + map_name

    elif map_type == 'ADAPT':
        remote_dir = f"https://gong.nso.edu/adapt/maps/gong/{year}/"
        soup = BeautifulSoup(requests.get(remote_dir).text, "html.parser")
        file_id = 'adapt40311'
        file_names = [a.get("href") for a in soup.find_all("a") if file_id in a.get("href")]

        time_deltas = []
        for name in file_names:
            try:
                parsed_date = datetime.strptime(name.split("_")[2], "%Y%m%d%H%M")
                delta = (parsed_date - date_datetime).total_seconds()
                time_deltas.append(delta)
            except Exception as e:
                logger.warning(f"Skipping file {name} due to parsing error: {e}")
                time_deltas.append(np.inf)

        if not time_deltas or all(np.isinf(time_deltas)):
            raise RuntimeError("No valid ADAPT file found on the remote server.")

        map_name = file_names[np.argmin(np.abs(time_deltas))]
        remote_file = remote_dir + map_name

    elif map_type == 'HMI_small':
        map_name = f"hmi.Synoptic_Mr_small.{cr_number}.fits"
        remote_file = f"http://jsoc.stanford.edu/data/hmi/synoptic/{map_name}"
    elif map_type == 'HMI_polfil':
        map_name = f'hmi.Synoptic_Mr_polfil.{cr_number}.fits'
        remote_file = f"http://jsoc.stanford.edu/data/hmi/synoptic/{map_name}"


    local_file = os.path.join(output_dir, map_name)

    if not os.path.exists(local_file):
        local_file = sunpy.util.net.download_file(remote_file, directory=output_dir, overwrite=True)
        logger.info(f"Downloaded map: {local_file}")
    else:
        logger.info(f"Map already exists locally: {local_file}")

    logger.info(f"Output file: {output_name}")
    logger.info(f"Downloaded map: {local_file}")

    return output_name, local_file

def build_theta_phi(theta, phi):
    """Build 2D meshgrids from 1D theta and phi arrays.

    Args:
        theta (ndarray): 1D array of theta values (colatitude).
        phi (ndarray): 1D array of phi values (longitude).

    Returns:
        tuple: (Theta, Phi) meshgrids.
    """
    return np.tile(theta, (len(phi), 1)).T, np.tile(phi, (len(theta), 1))

def read_magnetogram(file_path, map_type, adapt_map=0):
    """Read magnetic field map file and extract Br, Theta, Phi grids.

    Args:
        file_path (str): Path to the magnetogram file.
        map_type (str): Type of the map ('WSO', 'GONG', 'ADAPT', 'HMI_small', 'HMI_pofil').
        adapt_map (int, optional): Index for ADAPT map. Defaults to 0.

    Returns:
        tuple: (Br_map, Theta, Phi) arrays.
    """
    logger.info('Reading file')

    if map_type == 'ADAPT':
        input_data = fits.getdata(file_path, ext=0)
        Br_map = input_data[adapt_map, ::-1, :]
        nb_th, nb_phi = Br_map.shape
        theta = np.linspace(0., np.pi, nb_th)
        phi = np.linspace(0., 2.0*np.pi, nb_phi)
        Theta, Phi = build_theta_phi(theta, phi)

    elif map_type.lower() == 'wso':
        with open(file_path, 'r') as fwso:
            line = fwso.readline()
            while line.strip() == '':
                line = fwso.readline()
            header = line.split()
            lat_type = 'sinlat' if 'sine' in header else 'lat'
            nb_th = int(header[1])
            nb_phi = 73
            nb_lines = 4 * nb_phi
            nb_thplus = 4
            nb_th2 = nb_th + 2 * nb_thplus
            Br_read = np.zeros((nb_th, nb_phi))
            fwso.readline()
            idx_ph = nb_phi
            for _ in range(nb_lines):
                line = fwso.readline()
                split_line = line.split()
                if split_line[0].startswith('C'):
                    idx_th = 0
                    idx_ph -= 1
                    for val in split_line[1:]:
                        Br_read[idx_th, idx_ph] = float(val)
                        idx_th += 1
                else:
                    for val in split_line:
                        Br_read[idx_th, idx_ph] = float(val)
                        idx_th += 1

        if lat_type == 'lat':
            Br_ext = np.pad(Br_read, ((nb_thplus, nb_thplus), (0, 0)), mode='edge')
            Br_map = Br_ext * 0.01
            nb_th = Br_map.shape[0]
            theta = (np.linspace(-90., 90., nb_th) + 90.) * np.pi / 180.
        else:
            Br_data = Br_read[::-1, :] * 0.01
            sinlat = np.linspace(-14.5/15., 14.5/15., nb_th)
            theta_map = np.arcsin(sinlat) + np.pi/2.
            theta = np.linspace(0., np.pi, nb_th)
            phi = np.linspace(0., 360., nb_phi) * np.pi / 180.
            fbr = interpolate.RectBivariateSpline(theta_map, phi, Br_data)
            Br_map = fbr(theta, phi)[::-1, :]

        phi = np.linspace(0., 360., nb_phi) * np.pi / 180.
        Theta, Phi = build_theta_phi(theta, phi)

    else:
        input_data = fits.getdata(file_path)
        Br_data = input_data[::-1, :]
        nb_th, nb_phi = Br_data.shape
        sinlat = np.linspace(-1., 1., nb_th)
        theta = np.arcsin(sinlat) + np.pi/2.
        phi = np.linspace(0., 2.0*np.pi, nb_phi)
        Theta, Phi = build_theta_phi(theta, phi)
        Br_map = np.nan_to_num(Br_data)

    logger.info("End of reading file")


    return Br_map, Theta, Phi

def project_and_reconstruct(Br, Theta, Phi, lmax, amp=1, alpha=0):
    """Project Br on spherical harmonics and reconstruct Br_mode.

    Args:
        Br (ndarray): Original radial field.
        Theta (ndarray): Colatitude grid.
        Phi (ndarray): Longitude grid.
        lmax (int): Maximum spherical harmonic degree.
        amp (float): Amplitude factor for reconstruction.
        alpha (float): Enhances the filtering for higher frequencies.

    Returns:
        tuple: (Br_mode, coefbr)
    """
    logger.info('Beginning of projection')
    nb_th, nb_phi = Br.shape
    nb_modes_tot = int((lmax + 1) * (lmax + 2) / 2 - 1)

    dtheta = np.tile(np.gradient(Theta[:, 0]), (nb_phi, 1)).T
    dphi = np.tile(np.gradient(Phi[0, :]), (nb_th, 1))

    coefbr = np.zeros(nb_modes_tot, dtype=complex)
    mod = 0
    for l in range(1, lmax + 1):
        logger.info(f"l = {l}")
        for m in range(0, l + 1):
            ylm = scisp.sph_harm(m, l, Phi, Theta) / (1+alpha*l**2*(l+1)**2)
            integrand = Br * np.conj(ylm) * np.sin(Theta) * dtheta * dphi
            coefbr[mod] = np.sum(integrand)
            mod += 1
    logger.info('End of projection')

    logger.info('Reconstructing Br')
    Br_mode = np.zeros_like(Br)
    mod = 0
    for l in range(1, lmax + 1):
        logger.info(f"l = {l}")
        for m in range(0, l + 1):
            ylm = scisp.sph_harm(m, l, Phi, Theta)
            Br_mode += np.real(coefbr[mod] * ylm)
            mod += 1

    Br_mode /= 2.2
    Br_mode *= amp
    logger.info('End of reconstructing Br')
    return Br_mode, coefbr

def write_bc_file(output_name, Br_mode, theta, phi, r_st):
    """Write boundary condition file.

    Args:
        output_name (str): Path to output file.
        Br_mode (ndarray): Reconstructed radial field.
        theta (ndarray): 1D theta grid.
        phi (ndarray): 1D phi grid.
        r_st (float): Spherical radius.
    """
    logger.info("Writing BC file")
    nb_th, nb_phi = Br_mode.shape
    with open(output_name, 'w') as F:
        F.write('1 \n')
        F.write(f'!PHOTOSPHERE {(nb_th - 2) * nb_phi + 2} \n')
        for j in range(nb_th):
            for k in range(nb_phi):
                if ((j == 0 or j == nb_th - 1) and k != 0):
                    continue
                xcoord = r_st * np.sin(theta[j]) * np.cos(phi[k])
                ycoord = r_st * np.sin(theta[j]) * np.sin(phi[k])
                zcoord = r_st * np.cos(theta[j])
                F.write(f"{xcoord:.16e} {ycoord:.16e} {zcoord:.16e} {Br_mode[j, k]:.16e} \n")
    logger.info("End of writing BC file")

def plot_maps(Br, Br_mode, theta, phi, map_type, visu_type, output_path='output_map.png'):
    """Plot original and reconstructed Br maps.

    Args:
        Br (ndarray): Original radial magnetic field.
        Br_mode (ndarray): Reconstructed radial magnetic field.
        theta (ndarray): 1D theta grid.
        phi (ndarray): 1D phi grid.
        map_type (str): Type of the input map ('WSO', 'GONG', etc.).
        visu_type (str): Visualization style ('lat' or 'sinlat').
        lat_type (str, optional): Latitude type for WSO ('lat' or 'sinlat').
        output_path (str): Path where the figure will be saved.
    """
    logger.info("Plotting maps")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    if map_type == 'ADAPT':
        visu_type = 'lat'

    lat = 90. - 180. * theta / np.pi
    longi = 180. * phi / np.pi
    Lat, Long = np.meshgrid(lat, longi, indexing='ij')
    Sinlat = np.sin(np.radians(Lat))
    Sinlong = Long

    vmax1 = np.max(np.abs(Br))
    vmax1= 100
    vmax2 = np.max(np.abs(Br_mode))
    vmax2 = 30
    # Plot original map
    if visu_type == 'lat':
        im1 = ax1.imshow(
            Br[::-1], aspect='auto', origin='lower', cmap='seismic',
            extent=[Long.min(), Long.max(), Lat.min(), Lat.max()],
            vmin=-vmax1, vmax=vmax1
        )
        ax1.set_ylabel('Latitude', fontsize=14)
    else:
        im1 = ax1.imshow(
            Br[::-1], aspect='auto', origin='lower', cmap='seismic',
            extent=[Sinlong.min(), Sinlong.max(), Sinlat.min(), Sinlat.max()],
            vmin=-vmax1, vmax=vmax1
        )
        ax1.set_ylabel('Sine Latitude', fontsize=14)

    ax1.set_title('Original magnetogram', fontsize=16)
    ax1.set_xticks(np.arange(0., 360., 60.))
    ax1.tick_params(axis='both', which='major', labelsize=12)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Br [G]', fontsize=14)
    cbar1.ax.tick_params(labelsize=12)

    # Plot processed input map
    if visu_type == 'lat':
        extent_map = [Long.min(), Long.max(), Lat.min(), Lat.max()]
        ylabel = 'Latitude'
    else:
        extent_map = [Sinlong.min(), Sinlong.max(), Sinlat.min(), Sinlat.max()]
        ylabel = 'Sine Latitude'

    im2 = ax2.imshow(
        Br_mode[::-1], aspect='auto', origin='lower', cmap='seismic',
        extent=extent_map, vmin=-vmax2, vmax=vmax2
    )
    ax2.set_title('Processed input magnetogram', fontsize=16)
    ax2.set_ylabel(ylabel, fontsize=14)
    ax2.set_xlabel('Longitude', fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Br [G/2.2]', fontsize=14)
    cbar2.ax.tick_params(labelsize=12)

    plt.savefig(output_path)
    plt.close()



if __name__ == "__main__":
    # --- Example runs ---
    """
    configs = [
        {
            "date": '2020-12-07T15:00:00', "map_type": 'GONG',
            "lmax": 15, "amp": 1, "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "../", "output_path_fig": "../hmi_20201207.png",
            "alpha" : 0
        },
        {
            "date": '2022-03-11T12:00:00', "map_type": 'GONG',
            "lmax": 15, "amp": 0.8, "write_map": False, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "../", "output_path_fig": "../gong_20220311.png",
            "alpha": 0
        },
        {
            "date": '2023-08-15T00:00:00', "map_type": 'ADAPT', "adapt_map": 4,
            "lmax": 15, "amp": 1.2, "write_map": True, "show_map": False,
            "visu_type": "sinlat",
            "output_dir": "../", "output_path_fig": "../adapt_20230815.png",
            "alpha": 0
        },
        {
            "date": '2024-09-12T06:00:00', "map_type": 'WSO',
            "lmax": 15, "amp": 1.0, "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "../", "output_path_fig": "../wso_20240912.png",
            "alpha": 0
        }
    ]
    """

    configs = [
        {
            "date": '2013-03-13T12:00:00', "map_type": 'GONG',
            "lmax": 50, "amp": 1, "write_map": False, "show_map": False,
            "visu_type": "sinlat",
            "output_dir": "./", "output_path_fig": "./hmi_20201207.png",
            "alpha": 3*10**(-6)
        }]

    for config in configs:
        date = config["date"]
        map_type = config["map_type"]
        output_dir = config.get("output_dir", "../")
        output_path_fig = config.get("output_path_fig", f"{output_dir}/{map_type.lower()}_map.png")
        lmax = config.get("lmax", 20)
        amp = config.get("amp", 1)
        r_st = 1.0
        adapt_map = config.get("adapt_map", 6)

        write_map = config.get("write_map", True)
        show_map = config.get("show_map", True)
        visu_type = config.get("visu_type", "sinlat")

        alpha = config.get("alpha", 0)

        output_name, local_file = generate_output_and_map_names(date, map_type, output_dir, lmax, method_used="sph")
        Br, Theta, Phi = read_magnetogram(local_file, map_type, adapt_map)

        Br_mode, coefbr = project_and_reconstruct(Br, Theta, Phi, lmax, amp, alpha)

        if write_map:
            write_bc_file(output_name, Br_mode, Theta[:, 0], Phi[0, :], r_st)

        if show_map:
            plot_maps(Br, Br_mode, Theta[:, 0], Phi[0, :], map_type, visu_type, output_path=output_path_fig)
