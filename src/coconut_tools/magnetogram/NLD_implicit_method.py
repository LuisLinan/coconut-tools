"""
Filtering interface for radial magnetic field map (Br)

Applies a nonlinear diffusion filter followed by optional Gaussian smoothing.
Encapsulates preprocessing for Perona-Malik filtering of magnetogram data.

Author: Jose Murteira
Cleaned and modularized by: Luis
"""
import logging

import numpy as np
import scipy.ndimage
from coconut_tools.magnetogram.nonlinear_diffusion_filter import nonlinearDiffusionFilter
from coconut_tools.magnetogram.sph_filtering import (
    read_magnetogram,
    generate_output_and_map_names,
    write_bc_file,
    plot_maps
)
from coconut_tools.logger_config import setup_logger

logger = setup_logger(__name__)

def filter_radial_field(
    Br: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    iterations: int = 3,
    apply_gaussian: bool = True,
    gaussian_sigma: float = 1.0,
    dx_override: float = 1.0,
    dy_override: float = 1.0,
    tau: float = 1.0
):
    """
    Apply nonlinear diffusion filtering to the Br magnetogram field.

    Args:
        Br (np.ndarray): 2D array of the radial magnetic field.
        phi (np.ndarray): Array of longitudes (in radians).
        theta (np.ndarray): Array of latitudes (in radians).
        iterations (int): Number of iterations for nonlinear diffusion.
        apply_gaussian (bool): Whether to apply Gaussian filtering before diffusion.
        gaussian_sigma (float): Sigma used in Gaussian smoothing.
        dx_override (float): Spatial resolution in x-direction (default: 1.0).
        dy_override (float): Spatial resolution in y-direction (default: 1.0).
        tau (float): Time step for the nonlinear diffusion.

    Returns:
        Tuple[np.ndarray, float]: Filtered Br field and final time step.
    """

    logger.info("Begin Filtering")

    dx = dx_override
    dy = dy_override

    if apply_gaussian:
        Br_smoothed = scipy.ndimage.gaussian_filter(Br, gaussian_sigma)
    else:
        Br_smoothed = Br.copy()

    Br_filtered, timestep = nonlinearDiffusionFilter(Br_smoothed, dx, dy, iterations, tau)

    logger.info("End Filtering")
    return Br_filtered, timestep


if __name__ == "__main__":
    # --- Example runs ---

    configs = [

        {
            "date": '2011-09-04T12:00:00', "map_type": 'HMI_small',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/hmi_20110904T120000.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2011-09-24T12:00:00', "map_type": 'HMI_small',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/hmi_20110924T120000.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2012-03-06T23:00:00', "map_type": 'HMI_small',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/hmi_20120306T230000.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2012-03-07T12:00:00', "map_type": 'HMI_small',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/hmi_20120307T120000.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2012-05-10T23:30:00', "map_type": 'HMI_small',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/hmi_20120510T233000.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2012-07-09T23:30:00', "map_type": 'HMI_small',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/hmi_20120709T233000.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2012-09-23T06:00:00', "map_type": 'HMI_small',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/hmi_20120923T060000.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2013-04-08T06:00:00', "map_type": 'HMI_small',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/hmi_20130408T060000.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2013-09-28T18:00:00', "map_type": 'HMI_small',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/hmi_20130928T180000.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2013-03-13T12:00:00', "map_type": 'HMI_small',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/hmi_20130313T120000.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2014-01-04T18:00:00', "map_type": 'HMI_small',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/hmi_20140104T180000.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2014-09-06T18:00:00', "map_type": 'HMI_small',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/hmi_20140906T180000.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2017-04-04T18:00:00', "map_type": 'HMI_small',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/hmi_20170404T180000.png",
            "tau": 5, "iterations": 7
        },

        {
            "date": '2011-09-04T12:00:00', "map_type": 'GONG',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/GONG_20110904T120000.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2011-09-24T12:00:00', "map_type": 'GONG',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/GONG_20110924T120000.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2012-03-06T23:00:00', "map_type": 'GONG',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/GONG_20120306T230000.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2012-03-07T12:00:00', "map_type": 'GONG',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/GONG_20120307T120000.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2012-05-10T23:30:00', "map_type": 'GONG',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/GONG_20120510T233000.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2012-07-09T23:30:00', "map_type": 'GONG',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/GONG_20120709T233000.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2012-09-23T06:00:00', "map_type": 'GONG',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/GONG_20120923T060000.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2013-04-08T06:00:00', "map_type": 'GONG',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/GONG_20130408T060000.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2013-09-28T18:00:00', "map_type": 'GONG',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/GONG_20130928T180000.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2013-03-13T12:00:00', "map_type": 'GONG',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/GONG_20130313T120000.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2014-01-04T18:00:00', "map_type": 'GONG',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/GONG_20140104T180000.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2014-09-06T18:00:00', "map_type": 'GONG',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/GONG_20140906T180000.png",
            "tau": 5, "iterations": 7
        },
        {
            "date": '2017-04-04T18:00:00', "map_type": 'GONG',
            "write_map": True, "show_map": True,
            "visu_type": "sinlat",
            "output_dir": "E:/GU V2/magnetogram/",
            "output_path_fig": "E:/GU V2/image/GONG_20170404T180000.png",
            "tau": 5, "iterations": 7
        },
    ]

    for config in configs:
        try:
            date = config["date"]
            map_type = config["map_type"]
            output_dir = config.get("output_dir", "../")
            output_path_fig = config.get("output_path_fig", f"{output_dir}/{map_type.lower()}_map.png")
            lmax = config.get("lmax", None)
            r_st = 1.0
            adapt_map = config.get("adapt_map", 6)

            write_map = config.get("write_map", True)
            show_map = config.get("show_map", True)
            visu_type = config.get("visu_type", "sinlat")

            tau  = config.get("tau", 5)
            iterations = config.get("iterations", 7)

            output_name, local_file = generate_output_and_map_names(date, map_type, output_dir, lmax, "NLD")
            Br, Theta, Phi = read_magnetogram(local_file, map_type, adapt_map)

            # Use phi and theta vectors from the 2D grid
            Br_filtered, timestep = filter_radial_field(Br, Phi[0, :], Theta[:, 0], iterations=7,tau=5)

            seuil = 40
            max_val = np.max(np.abs(Br_filtered))
            if max_val > seuil:
                scalaire = seuil / max_val
                Br_filtered = Br_filtered * scalaire

            if write_map:
                write_bc_file(output_name, Br_filtered, Theta[:, 0], Phi[0, :], r_st)

            if show_map:
                plot_maps(Br, Br_filtered, Theta[:, 0], Phi[0, :], map_type, visu_type, output_path=output_path_fig)
        except:
            logging.warning(f'fail to find {config["date"]} and {config["map_type"]}')
            continue