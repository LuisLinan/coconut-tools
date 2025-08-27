"""
Utility to merge `.vtu` files from multiple processors into a single file for ParaView visualization.

When COCONUT (or other MPI-based MHD solvers) produces separate `.vtu` files per processor,
this tool collects them into a unified `.pvtu` file readable by ParaView.

Features:
- Supports distributed output folder structures (e.g. one folder per timestep),
- Automatically generates `.pvtu` headers referencing each processor's `.vtu` file,
- Can be integrated in batch visualization workflows.

Intended for post-processing and visualization of 3D MHD simulation results.
"""

import os
from os.path import exists
from multiprocessing import Pool
import pyvista as pv
from coconut_tools.logger_config import setup_logger

logger = setup_logger(__name__)


def initialize(index: int, input_dir: str, output_dir: str, timestep: int, start_time: int, nb_proc: int = 900) -> None:
    """Merge distributed VTU files from Coolfluid into a single VTU file per time step.

    Args:
        index (int): Snapshot index.
        input_dir (str): Path to the input directory containing distributed VTU files.
        output_dir (str): Path to the output directory to save merged VTU files.
        timestep (int): Time increment between snapshots.
        start_time (int): Starting snapshot index.
        nb_proc (int, optional): Number of processors used. Defaults to 900.

    Returns:
        None
    """
    output_filename = os.path.join(output_dir, f"corona-mhd_{index + start_time:04d}.vtu")

    if exists(output_filename):
        logger.info(f"Snapshot already exists: {output_filename}")
        return

    filenames = [
        os.path.join(input_dir, f"corona-flow0-P{j}-iter_{start_time * timestep + timestep * index}.vtu")
        #os.path.join(input_dir, f"corona-flow0-P{j}.vtu")
        for j in range(nb_proc)
    ]

    logger.info(f"Reading files for snapshot {index + start_time:04d}")

    try:
        mesh = pv.read(filenames)
        merged = mesh.combine()
    except Exception as e:
        logger.warning(f"Failed to merge snapshot {index + start_time:04d}: {e}")
        return

    logger.info(f"Saving merged snapshot to {output_filename}")
    merged.save(output_filename)

    for f in filenames:
        try:
            os.remove(f)
        except Exception as e:
            logger.warning(f"Failed to remove file {f}: {e}")

def merge_all_snapshots(
    input_dir: str,
    output_dir: str,
    start_time: int,
    timestep: int,
    nbmax: int,
    num_processes: int = 5,
    nb_proc: int = 900
) -> None:
    """Launch parallel merging of distributed VTU snapshots.

    Args:
        input_dir (str): Path to the input VTU directory.
        output_dir (str): Path to the output VTU directory.
        start_time (int): Starting snapshot index.
        timestep (int): Time increment between snapshots.
        nbmax (int): Total number of snapshots to process.
        num_processes (int, optional): Number of parallel processes. Defaults to 5.
        nb_proc (int, optional): Number of distributed files per snapshot. Defaults to 900.

    Returns:
        None
    """
    args = [
        (idx, input_dir, output_dir, timestep, start_time, nb_proc)
        for idx in range(nbmax)
    ]

    with Pool(processes=num_processes) as pool:
        pool.starmap(initialize, args)


if __name__ == "__main__":
    merge_all_snapshots(
        input_dir="/scratch/leuven/342/vsc34280/time-evolving/run/results/",
        output_dir="/scratch/leuven/342/vsc34280/time-evolving/vtu/",
        start_time=0,
        timestep=1,
        nbmax=1,
        num_processes=5
    )
