"""Writers for COCONUT photospheric boundary-condition files."""

import os

import numpy as np

from coconut_tools.magnetogram.core.coordinates import _validate_theta_axis
from coconut_tools.tools.logger_config import setup_logger

logger = setup_logger(__name__)


def write_bc_file(output_name, Br_mode, theta, phi, r_st):
    """Write one photospheric surface as ``x y z Br`` COCONUT data."""
    logger.info("Writing BC file")
    output_dir = os.path.dirname(output_name)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    Br_mode = np.asarray(Br_mode)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if Br_mode.ndim != 2 or Br_mode.shape != (theta.size, phi.size):
        raise ValueError("Br_mode shape must match the theta and phi axes.")
    _validate_theta_axis(theta)

    nb_th, nb_phi = Br_mode.shape
    pole_tolerance = 1.0e-12
    polar_rows = np.isclose(theta, 0.0, atol=pole_tolerance, rtol=0.0) | np.isclose(
        theta,
        np.pi,
        atol=pole_tolerance,
        rtol=0.0,
    )
    number_of_points = nb_th * nb_phi - int(np.count_nonzero(polar_rows)) * (
        nb_phi - 1
    )

    with open(output_name, "w") as boundary_file:
        boundary_file.write("1 \n")
        boundary_file.write(f"!PHOTOSPHERE {number_of_points} \n")
        for theta_index in range(nb_th):
            for phi_index in range(nb_phi):
                if polar_rows[theta_index] and phi_index != 0:
                    continue
                xcoord = (
                    r_st
                    * np.sin(theta[theta_index])
                    * np.cos(phi[phi_index])
                )
                ycoord = (
                    r_st
                    * np.sin(theta[theta_index])
                    * np.sin(phi[phi_index])
                )
                zcoord = r_st * np.cos(theta[theta_index])
                boundary_file.write(
                    f"{xcoord:.16e} {ycoord:.16e} {zcoord:.16e} "
                    f"{Br_mode[theta_index, phi_index]:.16e} \n"
                )
    logger.info("End of writing BC file")
