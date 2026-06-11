"""
Compare Coconut Tomography with Tomographic Inversion

This script provides functions to compare a 3D MHD simulation result from COCONUT with tomography-based electron density maps.
It reads a .vtu simulation file and a tomography .dat file, performs interpolation, identifies the current sheet,
and generates comparative visual plots.

Usage:
    python tomography.py

Note:
    Please cite these papers if you plan to use the images or data for your own work.

    Morgan 2015: https://iopscience.iop.org/article/10.1088/0067-0049/219/2/23/meta
    Morgan 2019: https://iopscience.iop.org/article/10.3847/1538-4365/ab125d/meta
    Morgan 2020: https://iopscience.iop.org/article/10.3847/1538-4357/ab7e32/meta
"""
import os
from scipy.io import readsav
import numpy as np
import pyvista as pv
from vtk.util.numpy_support import vtk_to_numpy
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib.gridspec as gridspec
from coconut_tools.tools.download_tomo_file import download_tomography_file
from coconut_tools.tools.logger_config import setup_logger

logger = setup_logger(__name__)

def line(dens: np.ndarray, clt: np.ndarray, lon: np.ndarray) -> list:
    """
    Identify the current sheet location based on maximum density per longitude.

    Args:
        dens (np.ndarray): The density map.
        clt  (np.ndarray): Colatitude array.
        lon  (np.ndarray): Longitude array.

    Returns:
        list: Latitudes where the density peaks for each longitude.
    """
    return [clt[np.argmax(np.flip(dens, axis=0)[:, i])] - np.pi / 2.0 for i in range(len(lon))]

def compare_tomography_with_simulation(vtufile: str, datfile: str, output: str) -> None:
    """
    Compare electron density between a COCONUT MHD simulation and a tomographic reconstruction.

    Args:
        vtufile (str): Path to the COCONUT output (.vtu).
        datfile (str): Path to the tomography .dat file.
        output (str): Path to save the output figure.

    Returns:
        None
    """
    logger.info("Reading tomography file...")
    f = readsav(datfile)
    dens_tomo = f['dens'][1:-1, :]
    lon_tomo = f['lon_rad']
    clt_tomo = f['colat_rad'][1:-1]
    r_tomo = f['rmain']

    lat, lon = np.meshgrid(clt_tomo, lon_tomo, indexing='ij')
    x_1 = r_tomo * np.sin(lat.flatten()) * np.cos(lon.flatten())
    y_1 = r_tomo * np.sin(lat.flatten()) * np.sin(lon.flatten())
    z_1 = r_tomo * np.cos(lat.flatten())
    grid_apres = np.stack((x_1, y_1, z_1), axis=1)

    logger.info("Reading simulation file...")
    mesh = pv.read(vtufile)
    points_loc = vtk_to_numpy(mesh.GetPoints().GetData())
    rho = mesh.get_array('rho')
    uni_grid_avant, index_avant = np.unique(points_loc, axis=0, return_index=True)

    logger.info("Interpolating simulation density onto tomographic grid...")
    ri = RBFInterpolator(uni_grid_avant, rho[index_avant], kernel='linear', neighbors=50)
    dens_int = ri(grid_apres).reshape(len(clt_tomo), len(lon_tomo))

    # Normalize
    dens_int = (dens_int - np.min(dens_int)) / (np.max(dens_int) - np.min(dens_int))
    dens_tomo = (dens_tomo - np.min(dens_tomo)) / (np.max(dens_tomo) - np.min(dens_tomo))

    # Current sheet lines
    logger.info("Extracting current sheet profiles...")
    listcolatitude = line(dens_tomo, clt_tomo, lon_tomo)
    listcolatitude_smooth = savgol_filter(listcolatitude, 501, 10)
    listcolatitude_coco = line(dens_int, clt_tomo, lon_tomo)
    listcolatitude_coco_smooth = savgol_filter(listcolatitude_coco, 501, 10)

    logger.info("Max error: %.2f°", np.degrees(np.max(np.abs(np.array(listcolatitude_coco_smooth) - np.array(listcolatitude_smooth)))))
    logger.info("Mean error: %.2f°", np.degrees(np.mean(np.abs(np.array(listcolatitude_coco_smooth) - np.array(listcolatitude_smooth)))))

    logger.info("Generating plot...")
    fig = plt.figure(tight_layout=True, figsize=(15, 7))
    spec = gridspec.GridSpec(ncols=6, nrows=2, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0:3])
    ax2 = fig.add_subplot(spec[0, 3:])
    ax3 = fig.add_subplot(spec[1, 1:5])

    extent = [np.degrees(lon_tomo[0]), np.degrees(lon_tomo[-1]),
              np.degrees(np.min(clt_tomo) - np.pi / 2), np.degrees(np.max(clt_tomo) - np.pi / 2)]

    im1 = ax1.imshow(dens_int, cmap='gist_stern', origin='upper', extent=extent)
    fig.colorbar(im1, ax=ax1).set_label('Density [norm.]', fontsize=12)
    ax1.contour(np.degrees(lon_tomo), np.degrees(clt_tomo - np.pi / 2.0), np.flip(dens_int, axis=0),
                levels=[0.2, 0.4, 0.6, 0.8], colors='k', linewidths=0.5, linestyles='dashed')
    ax1.set_title(f'COCONUT at R={r_tomo:.2f}$R_\odot$')
    ax1.set_xlabel('Longitude [°]')
    ax1.set_ylabel('Latitude [°]')

    im2 = ax2.imshow(dens_tomo, cmap='gist_stern', origin='upper', extent=extent)
    fig.colorbar(im2, ax=ax2).set_label('Density [norm.]', fontsize=12)
    ax2.contour(np.degrees(lon_tomo), np.degrees(clt_tomo - np.pi / 2.0), np.flip(dens_tomo, axis=0),
                levels=[0.2, 0.4, 0.6, 0.8], colors='k', linewidths=0.5, linestyles='dashed')
    ax2.set_title(f'Tomography at R={r_tomo:.2f}$R_\odot$')
    ax2.set_xlabel('Longitude [°]')
    ax2.set_ylabel('Latitude [°]')

    ax3.set_title(f'Maximum of density at R={r_tomo:.2f}$R_\odot$')
    ax3.plot(np.degrees(lon_tomo), np.degrees(listcolatitude_smooth), label='Tomography')
    ax3.plot(np.degrees(lon_tomo), np.degrees(listcolatitude_coco_smooth), label='COCONUT')
    ax3.set_xlabel('Longitude [°]')
    ax3.set_ylabel('Latitude [°]')
    ax3.legend()
    ax3.set_xlim(extent[0], extent[1])

    plt.savefig(output, dpi=300)
    logger.info("Plot saved to %s", output)

if __name__ == '__main__':
    date = "20190704"
    altitude = "5-0"
    output_dir = "C:/Users/luisl/Desktop/Tinatin"
    datfile = os.path.join(output_dir, f"tomo_sta_cor2a_{date}_{altitude}.dat")

    if not os.path.exists(datfile):
        download_tomography_file(date, altitude, output_dir)

    vtufile = "C:/Users/luisl/Desktop/hmi/corona-mhd_0.vtu"
    output = os.path.join(output_dir, "tomo_3.pdf")
    compare_tomography_with_simulation(vtufile, datfile, output)
