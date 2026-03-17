"""
This module provides plotting utilities for analyzing and comparing the
magnetic, thermodynamic, and velocity outputs from COCONUT simulation results.

Functions included:
    - plot_boundary_profil: Plot boundary magnetic and plasma profiles from dat files.
    - Surface_2D_onetime: Plot 2D slices of key physical quantities at a single time from dat file.
    - create_plot_B: Plot magnetic field components for solar minimum and maximum conditions.
    - create_plot_comparison: Compare multiple simulation runs with different parameters (e.g., ζ).
    - create_plot_max_quantities_vs_b0: Compare the scaling of max V, B, and density vs B0 from different runs.

Note:
    Some plotting functions assume that the relevant simulation data has been
    preprocessed and saved into `.hdf5` files. These HDF5 files must contain
    the expected datasets (e.g., 'B', 'V', 'rho', 'time', etc.). Cf. create_hdf script.

    This design allows for efficient reuse of simulation outputs and avoids
    recomputing or reloading raw binary data multiple times.

"""

import os
import math
from typing import Literal
from coconut_tools.read_dat_files import read_data
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from coconut_tools.logger_config import setup_logger
from pathlib import Path

logger = setup_logger(__name__)

def m_to_rsun(x: float) -> float:
    """Convert a distance expressed in meters to solar radii.

    The conversion assumes 1 astronomical unit equals 215 solar radii,
    which implies 1 Rsun = 149597870700 / 215 meters.

    Args:
        x (float): Distance in meters.

    Returns:
        float: Distance in solar radii.
    """
    AU = 149597870700.0
    RSUN_PER_AU = 215.0
    return x * RSUN_PER_AU / AU


def plot_boundary_profil(inputdir, outputfile, label_dict, color_map):
    """Plot magnetic and thermodynamic quantities from HDF5 files.

    Args:
        inputdir (str): Path to the directory containing the HDF5 data files.
        outputfile (str): Path where the resulting figure will be saved.
        label_dict (dict): Dictionary mapping filenames (str) to legend labels (str).
        color_map (dict): Dictionary mapping labels to their corresponding colors.

    Notes:
        - The function expects each HDF5 file to contain datasets:
          'vr', 'vlon', 'vclt', 'br', 'blon', 'bclt', 'density', 'temperature', 'date'.
        - Vertical lines and annotations are added when the color is 'purple'.
        - The plot contains four subplots:
            1. Velocity [km/s]
            2. Magnetic field [nT]
            3. Density [m⁻³]
            4. Temperature [K]
    """

    import h5py
    fig, axs = plt.subplots(4, 1, figsize=(8, 12), constrained_layout=True)
    fig.suptitle(r'Magnetic and thermodynamic quantities from hdf files', fontsize=16)

    min_time, max_time = float('inf'), float('-inf')
    vlines_hours = [2, 4.2, 13]
    annotation_positions = {'Sheath': 1.8, 'ME': 4.0}
    y_annotation = 650

    for idx, (filename, label) in enumerate(label_dict.items()):
        with h5py.File(inputdir + filename, 'r') as hdf:
            # Extract data
            vr, vlon, vclt = hdf['vr'][:], hdf['vlon'][:], hdf['vclt'][:]
            br, blon, bclt = hdf['br'][:], hdf['blon'][:], hdf['bclt'][:]
            density, temperature = hdf['rho'][:], hdf['T'][:]
            date = list(hdf['time'][:])

        # Compute derived quantities
        b = np.sqrt(br**2 + blon**2 + bclt**2) * 1e9
        v = np.sqrt(vr**2 + vlon**2 + vclt**2)
        hours_since_first = [(t - date[0]) / 3600.0 for t in date]
        min_time = min(min_time, min(hours_since_first))
        max_time = max(max_time, max(hours_since_first))

        linestyle = '-' if idx < 3 else '--'
        color = color_map[label]

        # Plot each quantity
        for ax, ydata, ylabel in zip(
            axs,
            [v, b, density, temperature],
            ['V [km/s]', 'B [nT]', r'Density [$m^{-3}$]', 'Temperature [K]']
        ):
            ax.plot(hours_since_first, ydata, label=label, color=color,
                    linestyle=linestyle, linewidth=2)

        # Optional vertical markers and annotations
        if color == "purple":
            for ax in axs:
                for x in vlines_hours:
                    ax.axvline(x=x, color=color, linestyle='--', linewidth=1)

            axs[0].text(x=annotation_positions['Sheath'], y=y_annotation, s='Sheath',
                        rotation=90, ha='center', va='center', fontsize=12)
            axs[0].text(x=annotation_positions['ME'], y=y_annotation, s='ME',
                        rotation=90, ha='center', va='center', fontsize=12)

    # Set labels and styling
    ylabels = ['V [km/s]', 'B [nT]', r'Density [$m^{-3}$]', 'Temperature [K]']
    for i, ax in enumerate(axs):
        ax.set_xlim([min_time, max_time])
        ax.set_ylabel(ylabels[i], fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)

    axs[3].set_xlabel('Time [h]', fontsize=14)

    # Add legends
    for ax in axs:
        ax.legend(fontsize=10, loc='upper right')
    axs[0].legend(loc='lower right', prop={'size': 10})

    plt.savefig(outputfile)
    plt.close()

def Surface_2D_onetime(
    inputfile: str,
    outputfile: str,
    mode: Literal["all", "reduced"] = "all",
    extended: bool = False,
    showP: bool = False,
) -> None:
    """Plot thermodynamic and magnetic quantities from a COCONUT output file.

    Args:
        inputfile (str): Path to the input directory containing COCONUT binary output.
        outputfile (str): Path where the figure should be saved.
        mode (Literal["all", "reduced"], optional): "all" plots 8 variables, "reduced"
            plots only density, temperature, Br, and Vr. Default is "all".
        extended (bool, optional): If True, read radius from the .dat file. Default is False.
        showP (bool, optional): If True, also plot pressure and include it in the saved figure.
            Default is False.

    Returns:
        None
    """
    import cmocean as cm

    clat_ticks = [-90, -45, 0, 45, 90]
    lon_ticks = [-180, -135, -90, -45, 0, 45, 90, 135, 180]

    reduced = (mode == "reduced")
    date, r, clt, lon, vr, vlon, vclt, density, br, bclt, blon, temp = read_data(
        inputfile, reduced=reduced, extended=extended
    )

    vars_to_plot = {
        "number_density": (density.T, cm.cm.thermal, "Density [$m^{-3}$]"),
        "temperature": (temp.T, cm.cm.haline, "Temperature [K]"),
        "Br": (br.T * 1e9, cm.cm.balance, "Br [nT]"),
        "Vr": (vr.T, cm.cm.matter_r, "Vr [km/s]"),
    }

    if not reduced:
        if any(x is None for x in (bclt, blon, vclt, vlon)):
            raise ValueError(
                "Expected full vector fields (Bclt/Blon/Vclt/Vlon) but some are missing (None)."
            )
        vars_to_plot.update(
            {
                "Bclt": (bclt.T * 1e9, cm.cm.balance, "Bclt [nT]"),
                "Blon": (blon.T * 1e9, cm.cm.balance, "Blon [nT]"),
                "Vclt": (vclt.T, cm.cm.balance, "Vclt [km/s]"),
                "Vlon": (vlon.T, cm.cm.balance, "Vlon [km/s]"),
            }
        )

    if reduced:
        plot_order = ["number_density", "temperature", "Br", "Vr"]
    else:
        plot_order = ["number_density", "temperature", "Br", "Bclt", "Blon", "Vr", "Vclt", "Vlon"]

    if showP:
        kB = 1.38e-23
        pressure = (density.T * temp.T) * kB
        vars_to_plot["pressure"] = (pressure, cm.cm.haline, "Pressure [Pa]")

    tit = r"Magnetic and thermodynamic quantities from COCONUT"
    if extended and r is not None:
        tit += "\n" + r"at r={0:.2f} R$_\odot$".format(m_to_rsun(r))

    ncols = 2
    n_main = len(plot_order)
    nrows_main = math.ceil(n_main / ncols)
    total_rows = nrows_main + (1 if showP else 0)

    fig = plt.figure(figsize=(15, 2.8 * total_rows), constrained_layout=True)
    gs = fig.add_gridspec(total_rows, ncols)
    fig.suptitle(tit, size=18)

    def _format_axes(ax):
        ax.set_xticks(lon_ticks)
        ax.set_yticks(clat_ticks)
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.set_ylabel("Latitude [deg]", fontsize=11)


    def _plot_panel(ax, data, cmap, label, show_xlabel: bool):
        data_shifted = np.roll(data, data.shape[1] // 2, axis=1)
        im = ax.imshow(
            data_shifted,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            extent=[lon_ticks[0], lon_ticks[-1], clat_ticks[-1], clat_ticks[0]],
        )
        _format_axes(ax)
        if show_xlabel:
            ax.set_xlabel("Longitude [deg]", fontsize=11)

        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cbar.set_label(label, fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.yaxis.offsetText.set_fontsize(10)
    # Main panels
    for i, name in enumerate(plot_order):
        row = i // ncols
        col = i % ncols
        ax = fig.add_subplot(gs[row, col])

        data, cmap, label = vars_to_plot[name]

        # If pressure exists, only pressure gets xlabel (cleaner).
        show_xlabel = (not showP) and (row == nrows_main - 1)
        _plot_panel(ax, data, cmap, label, show_xlabel=show_xlabel)

    # Pressure panel: create it spanning both columns (so layout reserves space cleanly),
    # then shrink width and center it.
    if showP:
        # Sous grille sur la dernière ligne pour centrer la pression sans set_position
        sub = gs[nrows_main, :].subgridspec(
            1, 3, width_ratios=[1.0, 1.25, 1.0]
        )

        # Axes vides gauche et droite pour réserver l espace et centrer le panneau
        fig.add_subplot(sub[0, 0]).axis("off")
        fig.add_subplot(sub[0, 2]).axis("off")

        # Panneau pression centré, taille comparable aux autres
        axp = fig.add_subplot(sub[0, 1])

        data, cmap, label = vars_to_plot["pressure"]
        _plot_panel(axp, data, cmap, label, show_xlabel=True)

        plt.savefig(outputfile, dpi=300)
    else:
        plt.savefig(outputfile, dpi=300)

    if showP:
        plt.show()

    plt.close(fig)


def Surface_2D_onetime_file_axes(
    inputfile: str,
    outputfile: str,
    mode: Literal["all", "reduced"] = "all",
    extended: bool = False,
    showP: bool = False,
    invert_yaxis: bool = True,
) -> None:
    """Plot 2D maps using colatitude/longitude axes read from the input file.

    This version uses the exact clt/lon grids from the .dat file to avoid
    implicit flipping. Axes are labeled in degrees of colatitude/longitude.
    """
    import cmocean as cm

    reduced = (mode == "reduced")
    date, r, clt, lon, vr, vlon, vclt, density, br, bclt, blon, temp = read_data(
        inputfile, reduced=reduced, extended=extended
    )

    clt_deg = np.degrees(clt)
    lon_deg = np.degrees(lon)
    lon_grid, clt_grid = np.meshgrid(lon_deg, clt_deg, indexing="xy")

    vars_to_plot = {
        "number_density": (density.T, cm.cm.thermal, "Density [$m^{-3}$]"),
        "temperature": (temp.T, cm.cm.haline, "Temperature [K]"),
        "Br": (br.T * 1e9, cm.cm.balance, "Br [nT]"),
        "Vr": (vr.T, cm.cm.matter_r, "Vr [km/s]"),
    }

    if not reduced:
        if any(x is None for x in (bclt, blon, vclt, vlon)):
            raise ValueError(
                "Expected full vector fields (Bclt/Blon/Vclt/Vlon) but some are missing (None)."
            )
        vars_to_plot.update(
            {
                "Bclt": (bclt.T * 1e9, cm.cm.balance, "Bclt [nT]"),
                "Blon": (blon.T * 1e9, cm.cm.balance, "Blon [nT]"),
                "Vclt": (vclt.T, cm.cm.balance, "Vclt [km/s]"),
                "Vlon": (vlon.T, cm.cm.balance, "Vlon [km/s]"),
            }
        )

    if reduced:
        plot_order = ["number_density", "temperature", "Br", "Vr"]
    else:
        plot_order = ["number_density", "temperature", "Br", "Bclt", "Blon", "Vr", "Vclt", "Vlon"]

    if showP:
        kB = 1.38e-23
        pressure = (density.T * temp.T) * kB
        vars_to_plot["pressure"] = (pressure, cm.cm.haline, "Pressure [Pa]")

    title = r"Magnetic and thermodynamic quantities from COCONUT"
    if extended and r is not None:
        title += "\n" + r"at r={0:.2f} R$_\odot$".format(m_to_rsun(r))

    ncols = 2
    n_main = len(plot_order)
    nrows_main = math.ceil(n_main / ncols)
    total_rows = nrows_main + (1 if showP else 0)

    fig = plt.figure(figsize=(15, 2.8 * total_rows), constrained_layout=True)
    gs = fig.add_gridspec(total_rows, ncols)
    fig.suptitle(title, size=18)

    def _format_axes(ax):
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.set_ylabel("Colatitude [deg]", fontsize=11)
        if invert_yaxis:
            ax.invert_yaxis()

    def _plot_panel(ax, data, cmap, label, show_xlabel: bool):
        im = ax.pcolormesh(
            lon_grid,
            clt_grid,
            data,
            cmap=cmap,
            shading="auto",
        )
        _format_axes(ax)
        if show_xlabel:
            ax.set_xlabel("Longitude [deg]", fontsize=11)

        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cbar.set_label(label, fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.yaxis.offsetText.set_fontsize(10)

    for i, name in enumerate(plot_order):
        row = i // ncols
        col = i % ncols
        ax = fig.add_subplot(gs[row, col])
        data, cmap, label = vars_to_plot[name]
        show_xlabel = (not showP) and (row == nrows_main - 1)
        _plot_panel(ax, data, cmap, label, show_xlabel=show_xlabel)

    if showP:
        sub = gs[nrows_main, :].subgridspec(1, 3, width_ratios=[1.0, 1.25, 1.0])
        fig.add_subplot(sub[0, 0]).axis("off")
        fig.add_subplot(sub[0, 2]).axis("off")
        axp = fig.add_subplot(sub[0, 1])

        data, cmap, label = vars_to_plot["pressure"]
        _plot_panel(axp, data, cmap, label, show_xlabel=True)

        plt.savefig(outputfile, dpi=300)
    else:
        plt.savefig(outputfile, dpi=300)

    if showP:
        plt.show()

    plt.close(fig)


def create_plot_B(
    path_min: str,
    path_max: str,
    save_path: str = None,
    scale_factor: float = 0.402,
    show: bool = False
) -> None:
    """Plot magnetic field components for solar minimum and maximum from COCONUT hdf5 files.

    Args:
        path_min (str): Path to the HDF5 file for solar minimum data.
        path_max (str): Path to the HDF5 file for solar maximum data.
        save_path (str, optional): Path where the plot should be saved as PDF. If None, no file is saved.
        scale_factor (float): Time scaling factor to convert simulation time to hours.
        show (bool): Whether to display the plot using matplotlib.

    Returns:
        None
    """
    import h5py
    fig, ax = plt.subplots(2, 1, figsize=(7, 10))

    for i, (title, filepath, sheath_indices, me_indices) in enumerate([
        ("Solar minimum", path_min, (57, 65), (67, 178)),
        ("Solar maximum", path_max, (29, 36), (40, 105))
    ]):
        with h5py.File(filepath, "r") as fd:
            # Read magnetic field data
            B = fd['B'][:]
            Bx = fd['Bx'][:]
            By = fd['By'][:]
            Bz = fd['Bz'][:]
            time = fd['time'][:] * scale_factor

        ax[i].set_title(title)
        ax[i].plot(time, B, linewidth=1, color='black', linestyle="--", label='B')
        ax[i].plot(time, Bx, linewidth=1, label='Bx')
        ax[i].plot(time, By, linewidth=1, label='By')
        ax[i].plot(time, Bz, linewidth=1, label='Bz')
        ax[i].legend(frameon=True, prop={'size': 10}, loc='upper right').get_frame().set_edgecolor('black')

        # Highlight sheath and magnetic ejecta (ME) intervals
        t_sheath = time[sheath_indices[1]] - time[sheath_indices[0]]
        t_me = time[me_indices[1]] - time[me_indices[0]]
        ax[i].axvspan(time[sheath_indices[0]], time[sheath_indices[1]], facecolor='darkviolet', alpha=0.5)
        ax[i].axvspan(time[me_indices[0]], time[me_indices[1]], facecolor='plum', alpha=0.8)

        logger.info(f"{title}: sheath duration = {t_sheath:.2f} h, ME duration = {t_me:.2f} h")

        ax[i].set_ylabel('Magnetic field [nT]')
        ax[i].set_xlabel('Time [h]')
        ax[i].set_xlim(time[0], time[-1])
        ax[i].xaxis.set_minor_locator(AutoMinorLocator(5))

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    if show:
        plt.show()

    plt.close()

def create_plot_comparison(
    data_indices: dict,
    input_dir: str,
    output_dir: str,
    output_filename: str = "comp_zeta_max.pdf",
    time_scale: float = 0.402
) -> None:
    """Plot velocity, magnetic field, and density comparisons from multiple COCONUT simulation outputs.

    Args:
        data_indices (dict): Dictionary mapping data keys (e.g. file index) to labels.
        input_dir (str): Directory containing the HDF5 files named like 'Data_{index}.hdf5'.
        output_dir (str): Directory where the plot will be saved.
        output_filename (str): Name of the output PDF file. Default is 'comp_zeta_max.pdf'.
        time_scale (float): Scaling factor to convert simulation time to hours.

    Returns:
        None
    """
    import h5py
    fig, ax = plt.subplots(3, 1, figsize=(7, 10), sharex=False)

    # Setup colormap
    cm = plt.get_cmap('gist_rainbow')
    num_colors = len(data_indices)
    color_cycle = [cm(1. * i / num_colors) for i in range(num_colors)]
    for axis in ax:
        axis.set_prop_cycle(color=color_cycle)

    for idx, (key, label) in enumerate(data_indices.items()):
        file_path = os.path.join(input_dir, f"Data_{key}.hdf5")
        if not os.path.isfile(file_path):
            logger.warning(f"File not found: {file_path}")
            continue

        with h5py.File(file_path, "r") as fd:
            B = fd['B'][:]
            V = fd['V'][:]
            rho = fd['rho'][:]
            time = fd['time'][:] * time_scale

        # Plot the variables
        ax[0].plot(time, V, linewidth=1, label=label)
        ax[1].plot(time, B, linewidth=1, label=label)
        ax[2].plot(time, rho, linewidth=1, label=label)

    # Set axis labels
    ylabels = ['V [km/s]', 'B [nT]', 'Density [g/cm³]']
    for i in range(3):
        ax[i].set_ylabel(ylabels[i])
        ax[i].set_xlim(time[0], time[-1])
        ax[i].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[i].ticklabel_format(useOffset=False)
        ax[i].legend(frameon=True, prop={'size': 8}, loc='upper right', ncol=3,
                     title=r"$\zeta$").get_frame().set_edgecolor('black')

    ax[2].set_xlabel('Time [h]')

    # Save plot
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    logger.info(f"Comparison plot saved to {output_path}")

def create_plot_max_quantities_vs_b0(
    dict_folder_min: dict,
    dict_folder_max: dict,
    input_dir: str,
    output_dir: str,
    time_limit_index: int = 50,
    output_filename: str = "max_quantities.pdf"
) -> None:
    """Plot maximum V, B, and density as functions of B0 for solar min and max cases.

    Args:
        dict_folder_min (dict): Dictionary mapping zeta -> B0 for solar minimum.
        dict_folder_max (dict): Dictionary mapping zeta -> B0 for solar maximum.
        input_dir (str): Base directory where 'min_result/plot/' and 'max_result/plot/' subfolders are located.
        output_dir (str): Directory to save the resulting plot.
        time_limit_index (int): Limit index used for zeta in [7, 8, 9] to restrict data range.
        output_filename (str): Name of the PDF output file.

    Returns:
        None
    """
    import h5py
    cm = plt.get_cmap('gist_rainbow')

    B0min, B0max = [], []
    maxVl, maxrhol, maxBl = [], [], []
    maxVlmax, maxrholmax, maxBlmax = [], [], []

    fig, ax = plt.subplots(3, 1, figsize=(7, 10), sharex=False)

    for zeta in range(2, 21):
        # Solar minimum
        if zeta in dict_folder_min:
            B0min.append(dict_folder_min[zeta])
            fpath = os.path.join(input_dir, "min_result", "plot", f"Data_{zeta}.hdf5")
            with h5py.File(fpath, "r") as fd:
                Bl = fd["B"][:]
                Vl = fd["V"][:]
                rho = fd["rho"][:]
                maxVl.append(np.max(Vl))
                maxBl.append(np.max(Bl))
                maxrhol.append(np.max(rho))

        # Solar maximum
        if zeta in dict_folder_max:
            B0max.append(dict_folder_max[zeta])
            fpath = os.path.join(input_dir, "max_result", "plot", f"Data_{zeta}.hdf5")
            with h5py.File(fpath, "r") as fd:
                Bl = fd["B"][:]
                Vl = fd["V"][:]
                rho = fd["rho"][:]

                if zeta in [7, 8, 9]:
                    maxVlmax.append(np.max(Vl[:time_limit_index]))
                    maxBlmax.append(np.max(Bl[:time_limit_index]))
                    maxrholmax.append(np.max(rho[:time_limit_index]))
                else:
                    maxVlmax.append(np.max(Vl))
                    maxBlmax.append(np.max(Bl))
                    maxrholmax.append(np.max(rho))

    # --- Linear fits and plots ---
    def fit_and_plot(ax, x, y, color, label=None):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        try:
            from sklearn.linear_model import LinearRegression  # local import
            model = LinearRegression().fit(x.reshape(-1, 1), y)
            slope = float(model.coef_[0]);
            intercept = float(model.intercept_)
        except Exception:
            slope, intercept = np.polyfit(x, y, 1)
        y_fit = slope * np.array(x) + intercept
        ax.plot(x, y_fit, linestyle='--', color=color, label=label)
        return model

    fit_and_plot(ax[0], B0min, maxVl, 'blue')
    fit_and_plot(ax[0], B0max, maxVlmax, 'orange')

    fit_and_plot(ax[1], B0min, maxBl, 'blue')
    fit_and_plot(ax[1], B0max, maxBlmax, 'orange')

    # --- Polynomial fits (degree 2) for density ---
    def polyfit_and_plot(ax, x, y, color):
        coeffs = np.polyfit(x, y, 2)
        fit_y = coeffs[0] * np.array(x) ** 2 + coeffs[1] * np.array(x) + coeffs[2]
        ax.plot(x, fit_y, linestyle='--', color=color)
        logger.info(f"Polynomial fit coefficients (color {color}): {coeffs}")
        return coeffs

    polyfit_and_plot(ax[2], B0min, maxrhol, 'blue')
    polyfit_and_plot(ax[2], B0max, maxrholmax, 'orange')

    # --- Scatter points ---
    ax[0].scatter(B0min, maxVl, label="Minimum", marker="+", s=75)
    ax[0].scatter(B0max, maxVlmax, label="Maximum", marker="o", s=40)

    ax[1].scatter(B0min, maxBl, label="Minimum", marker="+", s=75)
    ax[1].scatter(B0max, maxBlmax, label="Maximum", marker="o", s=40)

    ax[2].scatter(B0min, maxrhol, label="Minimum", marker="+", s=75)
    ax[2].scatter(B0max, maxrholmax, label="Maximum", marker="o", s=40)

    # --- Axis labels and formatting ---
    ylabels = ['Max V [km/s]', 'Max B [nT]', 'Max Density [g/cm³]']
    for i in range(3):
        ax[i].set_ylabel(ylabels[i])
        ax[i].set_xlim(min(B0min), max(B0max))
        ax[i].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[i].ticklabel_format(useOffset=False)
        ax[i].legend(frameon=True, prop={'size': 10}, loc='upper left').get_frame().set_edgecolor('black')

    ax[2].set_xlabel(r'$B_{0}$ [G]')

    # --- Save plot ---
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    logger.info(f"Max quantities plot saved to {output_path}")

if __name__ == "__main__":
    """
    input_dir = Path("E:/euhforia/dat8/")
    output_dir = Path("E:/euhforia/image/inner_boundary/dat8/")

    output_dir.mkdir(parents=True, exist_ok=True)

    

    for dat_file in input_dir.glob("solar_wind_boundary_*.dat"):
    #    # extrait le timestamp
    #    timestamp = dat_file.stem.replace("solar_wind_boundary_", "")

        output_file = output_dir / f"surface.png"

        Surface_2D_onetime_file_axes(
            inputfile=str(dat_file),
            outputfile=str(output_file),
            mode="all",
            extended=True,
            showP=False
        )

        output_file = output_dir / f"surface2.png"
        Surface_2D_onetime(
            inputfile=str(dat_file),
            outputfile=str(output_file),
            mode="all",
            extended=True,
            showP=False
        )

    input_dir = Path("E:/euhforia/dat4/")
    output_dir = Path("E:/euhforia/image/inner_boundary/dat4/")

    output_dir.mkdir(parents=True, exist_ok=True)

    

    for dat_file in input_dir.glob("solar_wind_boundary_*.dat"):
    #    # extrait le timestamp
    #    timestamp = dat_file.stem.replace("solar_wind_boundary_", "")

        output_file = output_dir / f"surface.png"
        Surface_2D_onetime_file_axes(
            inputfile=str(dat_file),
            outputfile=str(output_file),
            mode="reduced",
            extended=True,
            showP=False
        )

        output_file = output_dir / f"surface2.png"
        Surface_2D_onetime(
            inputfile=str(dat_file),
            outputfile=str(output_file),
            mode="reduced",
            extended=True,
            showP=False
        )

    input_dir = Path("C:/Users/luisl/Desktop/fluxrope/")
    output_dir = Path("C:/Users/luisl/Desktop/fluxrope/")

    output_dir.mkdir(parents=True, exist_ok=True)

    

    for dat_file in input_dir.glob("solar_wind_boundary_*.dat"):
    #    # extrait le timestamp
    #    timestamp = dat_file.stem.replace("solar_wind_boundary_", "")

        output_file = output_dir / f"surface.png"

        Surface_2D_onetime_file_axes(
            inputfile=str(dat_file),
            outputfile=str(output_file),
            mode="all",
            extended=True,
            showP=False
        )

        output_file = output_dir / f"surface2.png"

        Surface_2D_onetime(
            inputfile=str(dat_file),
            outputfile=str(output_file),
            mode="all",
            extended=True,
            showP=False
        )
    input_dir = Path("E:/euhforia/dat4/")
    output_dir = Path("E:/euhforia/image/inner_boundary/dat4/")

    output_dir.mkdir(parents=True, exist_ok=True)

    

    for dat_file in input_dir.glob("solar_wind_boundary_*.dat"):
    #    # extrait le timestamp
    #    timestamp = dat_file.stem.replace("solar_wind_boundary_", "")

        output_file = output_dir / f"surface.png"
        Surface_2D_onetime(
            inputfile=str(dat_file),
            outputfile=str(output_file),
            mode="reduced",
            extended=True,
            showP=False
        )

    """
    input_dir = "E:/euhforia/CFcase_large/result_cme/test_hdf_output/"

    label_dict = {
        "CFData_cme.hdf5": "CME",
    }
    color_map = {
        "CME": "blue",
    }

    output_path = "E:/euhforia/CFcase_large/result_cme/test_plot_boundary.png"

    plot_boundary_profil(
        inputdir=input_dir,
        outputfile=output_path,
        label_dict=label_dict,
        color_map=color_map
    )
    
    """


    create_plot_comparison(
        data_indices={3: "ζ=3", 4: "ζ=4", 5: "ζ=5", 6: "ζ=6"},
        input_dir="/run/media/luis/TOSHIBA EXT/Coolfluid_bj/max_result/plot/",
        output_dir="/run/media/luis/TOSHIBA EXT/Coolfluid_bj/plots/",
        output_filename="comparison_maximum.pdf"
    )

    create_plot_B(
        path_min="/run/media/luis/TOSHIBA EXT/Coolfluid_bj/min_result/plot/Data_16.hdf5",
        path_max="/run/media/luis/TOSHIBA EXT/Coolfluid_bj/max_result/plot/Data_6.hdf5",
        save_path="/run/media/luis/TOSHIBA EXT/Coolfluid_bj/B.pdf",
        show=True
    )

    create_plot_max_quantities_vs_b0(
        dict_folder_min={z: b0 for z, b0 in zip(range(2, 21), np.linspace(1, 20, 19))},
        dict_folder_max={z: b0 for z, b0 in zip(range(2, 21), np.linspace(1.5, 25, 19))},
        input_dir="/run/media/luis/TOSHIBA EXT/Coolfluid_bj/",
        output_dir="/run/media/luis/TOSHIBA EXT/Coolfluid_bj/plots/"
    )
    """
