"""
Local Weighted Filtering Interface for Radial Magnetogram Field (Br)

This script applies a local adaptive filter based on spatial and radiometric
proximity, using a Gaussian kernel controlled by `alpha` and `Rn`.

Author: Jose Murteira
Cleaned and modularized by: Luis
"""

import numpy as np
import scipy.ndimage
import logging
from coconut_tools.magnetogram.local_weigh_filter import filter3
from coconut_tools.magnetogram.sph_filtering import (
    read_magnetogram,
    generate_output_and_map_names,
    write_bc_file,
    plot_maps
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def filter_radial_field_weighted(
    Br: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    alpha_factor: float,
    Rn: float,
    sig: float = 0.0,
    write_gaussian_prepass: bool = False
):
    """
    Apply local weighted filtering to radial magnetic field map.

    Args:
        Br (np.ndarray): Input magnetic field map.
        phi (np.ndarray): 1D array of longitudes in radians.
        theta (np.ndarray): 1D array of latitudes in radians.
        alpha_factor (float): Alpha controlling kernel sharpness.
        Rn (float): Radius of neighborhood in physical units.
        sig (float): Sigma of optional Gaussian smoothing.
        write_gaussian_prepass (bool): Whether to export Gaussian-smoothed version.

    Returns:
        np.ndarray: Filtered Br field.
    """
    dlong = phi[1] / (2 * np.pi)
    dlat = theta[1] / np.pi
    R_sun = 696.34e6

    dx = dlong * R_sun
    dy = dlat * R_sun
    delta_var = max(dx, dy)

    Br_smoothed = scipy.ndimage.gaussian_filter(Br, sig) if sig > 0 else Br.copy()

    if write_gaussian_prepass:
        np.save("Br_gaussian_prepass.npy", Br_smoothed)

    logger.info("Running local filter with alpha=%.2f, Rn=%.2f, dx=%.2f, dy=%.2f", alpha_factor, Rn, dx, dy)
    Br_filtered = filter3(Br_smoothed, delta_var, delta_var, alpha_factor, Rn)

    return Br_filtered

if __name__ == "__main__":
    """
    configs = [
        {
            "date": '2020-12-07T15:00:00', "map_type": 'HMI_small',
            "lmax": 15, "amp": 1, "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "../", "output_path_fig": "../hmi_20201207_weighted.png",
            "alpha": 1.4, "Rn": 5.0, "sig": 1.0
        },
        {
            "date": '2022-03-11T12:00:00', "map_type": 'GONG',
            "lmax": 15, "amp": 0.8, "write_map": False, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "../", "output_path_fig": "../gong_20220311_weighted.png",
            "alpha": 1.2, "Rn": 6.0, "sig": 1.0
        },
        {
            "date": '2023-08-15T00:00:00', "map_type": 'ADAPT', "adapt_map": 4,
            "lmax": 15, "amp": 1.2, "write_map": True, "show_map": False,
            "visu_type": "sinlat",
            "output_dir": "../", "output_path_fig": "../adapt_20230815_weighted.png",
            "alpha": 1.0, "Rn": 5.0, "sig": 1.5
        },
        {
            "date": '2024-09-12T06:00:00', "map_type": 'WSO',
            "lmax": 15, "amp": 1.0, "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "../", "output_path_fig": "../wso_20240912_weighted.png",
            "alpha": 1.1, "Rn": 7.0, "sig": 0.0
        }
    ]
    """
    configs = [
        {
            "date": '2013-03-13T12:00:00', "map_type": 'HMI_small',
            "lmax": 15, "amp": 1, "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "./", "output_path_fig": "./hmi_20201207.png",
            "alpha": 1.4, "Rn" : 2, "sig": 1.5
        }]

    for config in configs:
        date = config["date"]
        map_type = config["map_type"]
        output_dir = config.get("output_dir", "../")
        output_path_fig = config.get("output_path_fig", f"{output_dir}/{map_type.lower()}_filtered.png")
        lmax = config.get("lmax", 20)
        amp = config.get("amp", 1)
        r_st = 1.0
        adapt_map = config.get("adapt_map", 6)

        write_map = config.get("write_map", True)
        show_map = config.get("show_map", True)
        visu_type = config.get("visu_type", "sinlat")

        alpha = config.get("alpha", 1.0)
        Rn = config.get("Rn", 5.0)
        sig = config.get("sig", 0.0)

        output_name, local_file = generate_output_and_map_names(date, map_type, output_dir, lmax)
        Br, Theta, Phi = read_magnetogram(local_file, map_type, adapt_map)

        Br_filtered = filter_radial_field_weighted(Br, Phi[0, :], Theta[:, 0], alpha, Rn, sig)

        if write_map:
            write_bc_file(output_name, Br_filtered, Theta[:, 0], Phi[0, :], r_st)

        if show_map:
            plot_maps(Br, Br_filtered, Theta[:, 0], Phi[0, :], map_type, visu_type, output_path=output_path_fig)
