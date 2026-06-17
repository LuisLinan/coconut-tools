# -*- coding: utf-8 -*-
"""
Configuration module for pyTDm-Coconut coupling.

This module provides functions to create and read `.ini` config files used
to define and control the injection of a flux rope (TDm or RBSL) into a
COCONUT simulation. The configuration includes spatial parameters, numerical
resolutions, and solver settings.

Typical usage example:
    createconfig(path_file=..., ...)
    nb_proc, nb_r, nb_th, nb_phi, eps = readconfig_coconut('myconfig.ini')
"""

import configparser
import os
import logging
from typing import Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def createconfig(path_file: str, path_tdm: str, path_save: str, name: str, theta: float, phi: float, alpha: float,
                 D: float, A: float, R: float, delta: float, zeta: float, case_tdm: str, geometry: str, solver: str,
                 flux: str,
                 nb_proc: int = 108, nb_r: int = 200, nb_th: int = 200, nb_phi: int = 200, eps: float = 0.01,
                 nfr: int = 100, xc: float = 0.5, xh: float = 0.5, hh_fr: float = 120, ll_fr: float = 35,
                 F_flx: float = 20) -> None:
    """
    Create a configuration file for pyTDm and COCONUT simulations.

    Args:
        path_file (str): Path to the input data.
        path_tdm (str): Path to save the TDM output.
        path_save (str): Directory where the config file will be saved.
        name (str): Case name.
        theta (float): Colatitude of the flux rope.
        phi (float): Longitude of the flux rope.
        alpha (float): Tilt angle.
        D (float): Distance from the solar surface.
        A (float): Small radius of the flux rope.
        R (float): Large radius of the flux rope.
        delta (float): Twist parameter.
        zeta (float): Instability parameter.
        case_tdm (str): Case identifier.
        geometry (str): Coordinate system ('spherical' or 'cartesian').
        solver (str): Solver used ('COCONUT' or 'pluto').
        flux (str): Type of flux rope ('Tdm' or 'RBSL').
        nb_proc (int): Number of processors.
        nb_r (int): Radial resolution.
        nb_th (int): Colatitude resolution.
        nb_phi (int): Longitude resolution.
        eps (float): Smoothing parameter.
        nfr (int): Number of flux rope field lines.
        xc (float): Center x-coordinate (RBSL).
        xh (float): Half-width (RBSL).
        hh_fr (float): Height of flux rope (RBSL).
        ll_fr (float): Footpoint length (RBSL).
        F_flx (float): Total flux (RBSL).

    Returns:
        None
    """
    configfile_name = os.path.join(path_save, f"{name}config.ini")

    if not os.path.isfile(configfile_name):
        configfile = open(configfile_name, 'w')
        config = configparser.ConfigParser()

        config.add_section('Path')
        config.set('Path', 'path_files', path_file)
        config.set('Path', 'path_tdm', path_tdm)
        config.set('Path', 'name', name)

        config.add_section('General')
        config.set('General', 'solver', solver)
        config.set('General', 'flux', flux)
        config.set('General', 'Case_tdm', case_tdm)

        config.add_section('COCONUT')
        config.set('COCONUT', 'nb_proc', str(nb_proc))
        config.set('COCONUT', 'nb_r', str(nb_r))
        config.set('COCONUT', 'nb_th', str(nb_th))
        config.set('COCONUT', 'nb_phi', str(nb_phi))
        config.set('COCONUT', 'eps', str(eps))

        config.add_section('RBSL')
        config.set('RBSL', 'nfr', str(nfr))
        config.set('RBSL', 'xc', str(xc))
        config.set('RBSL', 'xh', str(xh))
        config.set('RBSL', 'hh_fr', str(hh_fr))
        config.set('RBSL', 'll_fr', str(ll_fr))
        config.set('RBSL', 'F_flx', str(F_flx))

        config.add_section('GEOMETRY')
        config.set('GEOMETRY', 'D', str(D))
        config.set('GEOMETRY', 'A', str(A))
        config.set('GEOMETRY', 'R', str(R))
        config.set('GEOMETRY', 'DELTA', str(delta))
        config.set('GEOMETRY', 'ZETA', str(zeta))
        config.set('GEOMETRY', 'geometry', geometry)

        config.add_section('POSITION')
        config.set('POSITION', 'theta_0', str(theta))
        config.set('POSITION', 'phi_0', str(phi))
        config.set('POSITION', 'alpha_0', str(alpha))

        config.write(configfile)
        configfile.close()
        logger.info(f"Config file created: {configfile_name}")


def readconfig_coconut(path: str) -> Tuple[int, int, int, int, float]:
    """Read COCONUT-specific parameters from a config file."""
    config = configparser.ConfigParser()
    config.read(path)

    try:
        return (
            int(config['COCONUT']['nb_proc']),
            int(config['COCONUT']['nb_r']),
            int(config['COCONUT']['nb_th']),
            int(config['COCONUT']['nb_phi']),
            float(config['COCONUT']['eps']),
        )
    except KeyError as e:
        logger.error(f"Missing COCONUT parameter: {e}")
        raise


def readconfig_rbsl(path: str) -> Tuple[int, float, float, float, float, float]:
    """Read RBSL-specific parameters from a config file."""
    config = configparser.ConfigParser()
    config.read(path)

    try:
        return (
            int(config['RBSL']['nfr']),
            float(config['RBSL']['xc']),
            float(config['RBSL']['xh']),
            float(config['RBSL']['hh_fr']),
            float(config['RBSL']['ll_fr']),
            float(config['RBSL']['F_flx']),
        )
    except KeyError as e:
        logger.error(f"Missing RBSL parameter: {e}")
        raise


def readconfig_path(path: str) -> Tuple[str, str, str, str, str]:
    """Read path and general solver information from a config file."""
    config = configparser.ConfigParser()
    config.read(path)

    try:
        return (
            config['Path']['path_files'],
            config['Path']['path_tdm'],
            config['Path']['name'],
            config['General']['solver'],
            config['General']['flux']
        )
    except KeyError as e:
        logger.error(f"Missing path/general parameter: {e}")
        raise


def readconfig(path: str) -> Tuple[float, float, float, float, float, float, float, float, str, str]:
    """Read geometry and position parameters from a config file."""
    config = configparser.ConfigParser()
    config.read(path)

    try:
        return (
            float(config['POSITION']['theta_0']),
            float(config['POSITION']['phi_0']),
            float(config['POSITION']['alpha_0']),
            float(config['GEOMETRY']['D']),
            float(config['GEOMETRY']['A']),
            float(config['GEOMETRY']['R']),
            float(config['GEOMETRY']['DELTA']),
            float(config['GEOMETRY']['ZETA']),
            config['General']['Case_tdm'],
            config['GEOMETRY']['geometry'],
        )
    except KeyError as e:
        logger.error(f"Missing geometry/position parameter: {e}")
        raise


if __name__ == '__main__':


    createconfig(
        path_file='C:/Users/luisl/Desktop/cfmesh/',
        path_tdm='C:/Users/luisl/Desktop/cfmesh/',
        path_save='C:/Users/luisl/Desktop/cfmesh/config/',
        name='zeta10',
        theta=1.57, phi=3.14, alpha=0, D=0.15, A=0.10, R=0.3, delta=0.01,
        zeta=10, case_tdm='first', geometry='spherical', solver='COCONUT',
        flux='Tdm', nb_proc=270, nb_r=200, nb_th=200, nb_phi=200, eps=0.01,
        nfr=100, xc=0.5, xh=0.5, hh_fr=120, ll_fr=35, F_flx=20
    )


