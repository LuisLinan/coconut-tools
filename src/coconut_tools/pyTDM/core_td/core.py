"""
TDm object class to inject a flux rope (TDm or RBSL) into a COCONUT simulation.

This class loads configuration parameters, interpolates the background magnetic field,
constructs the flux rope field (TDm or RBSL), superposes it to the background field,
and exports the result to VTK format. Optionally updates CFmesh for Coolfluid.

Author:
    Luis Linan
"""

import numpy as np
import os
import math

from coconut_tools.tools.logger_config import setup_logger
from coconut_tools.pyTDM.core_td import formula_TDm
from coconut_tools.pyTDM.core_td.COCONUT import coconut
from coconut_tools.pyTDM.core_td.COCONUT import RBSL_util
from coconut_tools.pyTDM.core_td import myconfig as mc

from pyevtk.hl import gridToVTK
from scipy.interpolate import griddata

logger = setup_logger(__name__)

class TDm:

    def __init__(self, name):
        """Initialize the TDm object."""
        self.setup = dict()
        self.is_setup = False
        self.param_loaded = False
        self.name = name

    def __repr__(self):
        """String representation of the TDm object."""
        description = f'TDm flux rope "{self.name}"\n'
        if self.param_loaded:
            for key, value in self.setup.items():
                description += f'{key} : {value}\n'
        return description

    def add_TDm_coconut(self, path: str, cfmesh: bool) -> None:
        """
        Add a TDm or RBSL flux rope to a COCONUT simulation.

        Args:
            path (str): Path to the configuration .ini file.
            cfmesh (bool): If True, regenerate the CFmesh via Coolfluid.
        """
        # Read configuration
        path_file, path_tdm, name, solver, flux = mc.readconfig_path(path)
        theta_0, phi_0, alpha_0, d, a, R, Delta, zeta, case, geometry = mc.readconfig(path)
        nb_proc, nb_r, nb_th, nb_phi, eps = mc.readconfig_coconut(path)

        points, br, bt, bp = coconut.read_CFmesh(path_file)
        # points, br, bt, bp = coconut.read_vtu(path_file, nb_proc)  # Alternative

        # Interpolation grid
        logger.info('Interpolation...')
        x1 = np.logspace(0, np.log10(25), nb_r)
        x2 = np.linspace(eps, np.pi - eps, nb_th)
        x3 = np.linspace(eps, 2.0 * np.pi - eps, nb_phi)

        X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing='ij')
        grid_x = X1 * np.sin(X2) * np.cos(X3)
        grid_y = X1 * np.sin(X2) * np.sin(X3)
        grid_z = X1 * np.cos(X2)

        # Interpolate background magnetic field
        brr = griddata(points, br, (grid_x, grid_y, grid_z), method='nearest')
        btr = griddata(points, bt, (grid_x, grid_y, grid_z), method='nearest')
        bpr = griddata(points, bp, (grid_x, grid_y, grid_z), method='nearest')
        logger.info('Interpolation done.')

        # Evaluate B_amb at flux rope center
        r0 = 1 + R - d
        r_loc = np.argmin(np.abs(x1 - r0))
        t_loc = np.argmin(np.abs(x2 - theta_0))
        p_loc = np.argmin(np.abs(x3 - phi_0))
        B_amb = np.sqrt(brr[r_loc, t_loc, p_loc]**2 + btr[r_loc, t_loc, p_loc]**2 + bpr[r_loc, t_loc, p_loc]**2)

        logger.info(f'name: {name}, zeta: {zeta}, B_amp: {zeta * B_amb}')

        if flux == 'RBSL':
            nfr, xc, xh, hh_fr, ll_fr, F_flx = mc.readconfig_rbsl(path)

            deg2rad = math.pi / 180.0
            cen_lon_fr = phi_0 * deg2rad
            cen_lat_fr = theta_0 * deg2rad
            angle_fr = math.pi / 3.0
            hh_fr /= 699.5
            ll_fr *= deg2rad
            a = 0.1
            F_flx = F_flx * 1e20 / (2.0 * (6.955e10)**2)

            Br_fr, Bth_fr, Bph_fr = RBSL_util.RBSL_setup(
                nfr, nb_r, nb_th, nb_phi,
                x1, x2, x3,
                X1, X2, X3,
                grid_x, grid_y, grid_z,
                cen_lon_fr, cen_lat_fr,
                xc, xh, angle_fr,
                hh_fr, ll_fr,
                a, F_flx
            )

            brr += Br_fr
            btr += Bth_fr
            bpr += Bph_fr

        else:
            B_cart = formula_TDm.TDm_setup(
                x1, x2, x3,
                alpha_0, theta_0, phi_0,
                d, a, R, zeta, B_amb, Delta,
                case=case, geometry=geometry
            )
            brr += B_cart[0]
            btr += B_cart[1]
            bpr += B_cart[2]

        # Convert to Cartesian
        bxr = np.sin(X2) * np.cos(X3) * brr + np.cos(X2) * np.cos(X3) * btr - np.sin(X3) * bpr
        byr = np.sin(X2) * np.sin(X3) * brr + np.cos(X2) * np.sin(X3) * btr + np.cos(X3) * bpr
        bzr = np.cos(X2) * brr - np.sin(X2) * btr

        # Save to VTK
        brr, btr, bpr = map(np.asfortranarray, (brr, btr, bpr))
        bxr, byr, bzr = map(np.asfortranarray, (bxr, byr, bzr))

        os.makedirs(path_tdm, exist_ok=True)
        vtsfile = os.path.join(path_tdm, name)
        logger.info('Saving to VTK...')

        pointData = {
            'B_cart': (bxr, byr, bzr),
            'B_sph': (brr, btr, bpr)
        }
        if flux != 'RBSL':
            pointData['B_TDm'] = (B_cart[0], B_cart[1], B_cart[2])

        gridToVTK(vtsfile, grid_x, grid_y, grid_z, pointData=pointData)

        if cfmesh:
            coconut.restart_coolfluid(
                path_file, vtsfile, path_tdm,
                alpha_0, theta_0, phi_0, d, a, R, name,
                method='rbf', save=True
            )

        return
