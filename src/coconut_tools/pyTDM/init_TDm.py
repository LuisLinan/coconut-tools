"""
Module for injecting a flux rope (TDm or RBSL) into a COCONUT simulation.

This script reads multiple configuration files (`.ini`) and executes a processing
routine for each, injecting magnetic structures into the simulation environment.
It is designed to operate within the pyTDm framework, especially focusing on
COCONUT simulations, and manages the modification of CFmesh files if requested.

Typical usage example:
    execute_configs(path='/path/to/configs/', cfmesh=True)

This script expects config files with appropriate `.ini` format and uses
the pyTDm library to parse and process them.

Attributes:
    None

"""

import glob
import os
import shutil
import logging

from coconut_tools.pyTDM.core_td import myconfig as myc
from coconut_tools.pyTDM.core_td import core as pytdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def execute_configs(path: str = None, cfmesh: bool = False) -> None:
    """
    Reads and executes pyTDm configuration files for injecting magnetic structures
    into COCONUT simulations, with optional CFmesh modification.

    Args:
        path (str): Path where the configuration `.ini` files are located.
        cfmesh (bool): If True, a new `corona.CFmesh` will be created for COCONUT.

    Raises:
        TypeError: If `path` is not specified.

    Returns:
        None
    """

    if path is None:
        raise TypeError("The path to the configuration files must be specified.")

    path_to_move = os.path.join(path, 'used')
    os.makedirs(path_to_move, exist_ok=True)

    config_paths = sorted(glob.glob(os.path.join(path, '*.ini')))

    for config in config_paths:
        logger.info(f'Initiating analysis for config file: {config}')

        _, _, name, solver, _ = myc.readconfig_path(config)
        TDm = pytdm.TDm(name)

        if solver == 'COCONUT':
            TDm.add_TDm_coconut(config, cfmesh)
        else:
            logger.warning("Unsupported solver: please use the Regnault pytdm implementation instead.")

        shutil.move(config, os.path.join(path_to_move, os.path.basename(config)))

    logger.info('All config files have been processed.')


if __name__ == '__main__':
    # Example execution block
    execute_configs(path='C:/Users/luisl/Desktop/cfmesh/config/', cfmesh=True)

