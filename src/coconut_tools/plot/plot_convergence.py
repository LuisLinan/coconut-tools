"""
Plot COCONUT convergence residuals.

The script reads one or several
`convergence.plt-P0.FlowNamespace` files and produces a single
`convergence.png` figure containing the residual evolution of the nine
COCONUT primitive variables.

For multiple simulations, each panel compares the different runs.
For a single simulation, the legend is automatically omitted.
"""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

def compare_convergence(
    input_dir: Path,
    output_dir: Path,
    filename: str,
    label_dict: Dict[str, str],
    figsize: Tuple[int, int] = (14, 11),
) -> None:
    """Compare COCONUT convergence residuals across simulations.

    The residuals stored in `convergence.plt-P0.FlowNamespace`
    follow the order:

        rho, Vx, Vy, Vz, Bx, By, Bz, p, psi

    A single figure containing nine panels is generated, one for each
    primitive-variable residual.

    Args:
        input_dir:
            Base directory containing the simulation folders.

        output_dir:
            Directory where `convergence.png` will be written.

        filename:
            Name of the COCONUT convergence file.

        label_dict:
            Mapping between simulation folder names and labels used in
            the plot legend.

            Example for multiple simulations:

                {
                    "test1": "MaxIter 10, AbsNorm -4",
                    "test2": "MaxIter 10, AbsNorm -3",
                }

            For a single file located directly in `input_dir`:

                {
                    ".": "COCONUT"
                }

        figsize:
            Size of the complete figure.

    Returns:
        None
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    quantities = {
        "rho": r"Density $\rho$",
        "Vx": r"Velocity $V_x$",
        "Vy": r"Velocity $V_y$",
        "Vz": r"Velocity $V_z$",
        "Bx": r"Magnetic field $B_x$",
        "By": r"Magnetic field $B_y$",
        "Bz": r"Magnetic field $B_z$",
        "p": r"Pressure $p$",
        "psi": r"Divergence cleaning $\psi$",
    }

    # Column order in convergence.plt-P0.FlowNamespace
    headers = [
        "iter",
        "rho",
        "Vx",
        "Vy",
        "Vz",
        "Bx",
        "By",
        "Bz",
        "p",
        "psi",
        "CFL",
        "PhysTime",
        "DT",
        "WallTime",
        "MemUsage",
    ]

    simulations = {}

    # ------------------------------------------------------------------
    # Read convergence files
    # ------------------------------------------------------------------

    for folder_name, label in label_dict.items():

        file_path = input_dir / folder_name / filename

        if not file_path.exists():
            print(f"Warning: convergence file not found: {file_path}")
            continue

        try:
            data = np.loadtxt(file_path, skiprows=2)
        except (ValueError, OSError) as exc:
            print(f"Warning: unable to read {file_path}: {exc}")
            continue

        # np.loadtxt returns a 1-D array if the file contains one row.
        if data.ndim == 1:
            data = data[np.newaxis, :]

        # iter + 9 residuals are required.
        if data.shape[1] < 10:
            print(
                f"Warning: {file_path} contains only "
                f"{data.shape[1]} columns. At least 10 are required."
            )
            continue

        # Some COCONUT versions may contain fewer diagnostic columns
        # after the nine residuals.
        n_columns = min(data.shape[1], len(headers))

        df = pd.DataFrame(
            data[:, :n_columns],
            columns=headers[:n_columns],
        ).set_index("iter")

        simulations[label] = df

    if not simulations:
        raise RuntimeError(
            "No valid COCONUT convergence files were found."
        )

    # ------------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        3,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten()

    for ax, (quantity, title) in zip(axes, quantities.items()):

        for label, df in simulations.items():

            if quantity not in df.columns:
                continue

            iterations = df.index.to_numpy(dtype=float)
            residuals = df[quantity].to_numpy(dtype=float)

            # Remove NaN / Inf and COOLFluiD divergence sentinel values
            # such as -1.7976931e+308.
            mask = (
                np.isfinite(iterations)
                & np.isfinite(residuals)
                & (residuals > -100.0)
                & (residuals < 100.0)
            )

            ax.plot(
                iterations[mask],
                residuals[mask],
                linewidth=1.4,
                label=label,
            )

        ax.set_title(
            title,
            fontsize=11,
            pad=8,
        )

        ax.grid(
            True,
            linestyle="--",
            linewidth=0.5,
            alpha=0.5,
        )

        ax.tick_params(
            axis="both",
            labelsize=9,
        )

        ax.yaxis.set_major_formatter(
    FuncFormatter(lambda value, pos: rf"$10^{{{int(value)}}}$")
)

    # ------------------------------------------------------------------
    # Axis labels
    # ------------------------------------------------------------------

    for ax in axes[6:9]:
        ax.set_xlabel(
            "Iterations",
            fontsize=10,
        )

    for ax in axes[::3]:
        ax.set_ylabel("Residual magnitude")

    # ------------------------------------------------------------------
    # Main title
    # ------------------------------------------------------------------

    fig.suptitle(
        "COCONUT convergence residuals",
        fontsize=16,
        y=0.975,
    )

    # ------------------------------------------------------------------
    # Shared legend
    # ------------------------------------------------------------------

    handles, labels = axes[0].get_legend_handles_labels()

    # A legend is useful only when several simulations are compared.
    if len(simulations) > 1 and handles:

        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=min(len(labels), 4),
            frameon=False,
            fontsize=10,
        )

        bottom_margin = 0.075

    else:
        bottom_margin = 0.045

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    fig.tight_layout(
        rect=(
            0.03,
            bottom_margin,
            0.99,
            0.95,
        )
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    output_file = output_dir / "convergence.png"

    fig.savefig(
        output_file,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Convergence plot saved to: {output_file}")


if __name__ == "__main__":

    input_dir = Path(
        r"C:\Users\luisl\Desktop\coconut"
    )

    output_dir = input_dir / "plot"

    filename = "convergence.plt-P0.FlowNamespace"

    # --------------------------------------------------------------
    # Example 1: compare several simulations
    # --------------------------------------------------------------

    label_dict = {
        ".": "COCONUT",
    }

    compare_convergence(
        input_dir=input_dir,
        output_dir=output_dir,
        filename=filename,
        label_dict=label_dict,
        figsize=(14, 11),
    )