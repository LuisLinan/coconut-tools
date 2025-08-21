"""
Module for comparing convergence of COCONUT in-situ plasma quantities across different simulations.

This script reads convergence data from multiple COCONUT simulation cases, extracts key physical quantities
(e.g., density, velocity components, magnetic field components), and plots their evolution with
respect to the number of solver iterations.

Usage:
    Configure the `input_dir`, `output_dir`, `filename`, and `label_dict` directly in the `__main__` block.
    Run the script to generate comparison plots saved to the specified output directory.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

def compare_convergence(
    input_dir: Path,
    output_dir: Path,
    filename: str,
    label_dict: Dict[str, str],
    figsize: Tuple[int, int] = (10, 7)
) -> None:
    """Compare convergence of quantities across COCONUT simulations.

    Args:
        input_dir (Path): Base directory containing simulation result folders.
        output_dir (Path): Directory where plots will be saved.
        filename (str): Name of the convergence data file.
        label_dict (Dict[str, str]): Mapping from subfolder names to legend labels.
        figsize (Tuple[int, int], optional): Size of the figure. Defaults to (10, 7).

    Returns:
        None
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    quantities = {
        'n': 'Convergence of density',
        'Vx': 'Convergence of Vx', 'Vy': 'Convergence of Vy', 'Vz': 'Convergence of Vz',
        'Bx': 'Convergence of Bx', 'By': 'Convergence of By', 'Bz': 'Convergence of Bz',
        'residual': 'Convergence residuals'
    }

    headers = ["iter", "n", "Vx", "Vy", "Vz", "Bx", "By", "Bz", "residual"]

    for quantity, title in quantities.items():
        fig, ax = plt.subplots(figsize=figsize)

        for folder_name, label in label_dict.items():
            file_path = input_dir / folder_name / filename

            with file_path.open('r') as f:
                lines = f.readlines()
            data = np.loadtxt(lines[2:])
            df = pd.DataFrame(data[:, :9], columns=headers).set_index("iter")

            ax.plot(df.index, df[quantity], linewidth=1, label=label)

        ax.set_xlabel('Iterations')
        ax.set_ylabel(quantity)
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / f"comparison_{quantity}.png", dpi=200)
        plt.close()

if __name__ == "__main__":
    input_dir = Path("/run/media/luis/TOSHIBA EXT/coolfluid/Test_conver/")
    output_dir = input_dir / "plot"
    filename = "convergence.plt-P0.FlowNamespace"

    label_dict = {
        'test1': 'MaxIter 10, AbsNorm -4',
        'test2': 'MaxIter 10, AbsNorm -3',
        'test3': 'MaxIter 20, AbsNorm -4',
        'test4': 'MaxIter 5, AbsNorm -1'
    }

    compare_convergence(input_dir, output_dir, filename, label_dict, figsize=(10, 10))
