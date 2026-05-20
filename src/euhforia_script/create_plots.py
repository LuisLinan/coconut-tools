#!/usr/bin/env python
#
# This file is part of EUHFORIA.
#
# Copyright 2016, 2017, 2018 Jens Pomoell
#
# EUHFORIA is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# EUHFORIA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EUHFORIA. If not, see <http://www.gnu.org/licenses/>.

import argparse
import datetime
import glob
import os
import re
import sys
from matplotlib import colors
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, DayLocator, HourLocator
from matplotlib.ticker import MultipleLocator

import euhforia.core.constants as constants

# EUHFORIA imports
import euhforia.core.io
import euhforia.orbit
import euhforia.plot.colormap
import euhforia.plot.slice

from pathlib import Path

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - fallback for non-MPI environments
    MPI = None


def _get_mpi_context():
    """Return the current MPI communicator, rank, and world size."""
    if MPI is None:
        return None, 0, 1

    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def _partition_work(items, rank, size):
    """Assign items to ranks using a round-robin distribution."""
    return items[rank::size]

def expand_inputs(items, patterns=("*.vtu", "*.vts", "*.vtk", "*.h5")):
    out = []
    for it in items:
        p = Path(it)
        if p.is_dir():
            for pat in patterns:
                out.extend(sorted(p.glob(pat)))
        else:
            out.append(p)
    # enlever doublons en gardant l’ordre
    seen = set()
    uniq = []
    for p in out:
        s = str(p)
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq


def plot_slice_figure(
    integrator,
    data_dir,
    date,
    R,
    data,
    variable,
    meridional_slice,
    levels,
    cmap,
    heliospheric_objects,
    colorbar_ticks,
    output_dir,
    filename_prefix,
    lowres,
    plot_time_series=False,
    add_time_series=None,
):
    fig_size = (12, 7) if plot_time_series else (12, 7)
    fig = plt.figure(figsize=fig_size)

    if plot_time_series:
        gs = matplotlib.gridspec.GridSpec(
            3, 3, width_ratios=[4, 2.0, np.pi], height_ratios=[3, 1, 1]
        )
    else:
        gs = matplotlib.gridspec.GridSpec(
            3, 3, width_ratios=[4, 2.0, np.pi], height_ratios=[3, 1, 1]
        )
        #gs = matplotlib.gridspec.GridSpec(1, 3, width_ratios=[4, 2.0, np.pi])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax_t = plt.subplot(gs[2])

    if plot_time_series and add_time_series is not None:
        ax3 = plt.axes((0.135, 0.08, 0.67, 0.14))
        add_time_series(ax3)

    euhforia.plot.slice.equatorial_and_meridional_and_transversal(
        integrator,
        data_dir,
        date,
        R,
        data,
        variable=variable,
        meridional_slice=meridional_slice,
        levels=levels,
        cmap=cmap,
        heliospheric_objects=heliospheric_objects,
        colorbar_ticks=colorbar_ticks,
        fig=fig,
        ax=(ax1, ax2, ax_t),
    )

    ax2.set_xlim([0, data.grid.indomain_edge_coords.r[-1] / constants.astronomical_unit])



    if lowres:
        plt.savefig(output_dir + filename_prefix + "_" + date, dpi=72, bbox_inches = matplotlib.transforms.Bbox.from_extents(1,1.8,11.7,6.4))
    else:
        plt.savefig(output_dir + filename_prefix + "_" + date, dpi=200,bbox_inches = matplotlib.transforms.Bbox.from_extents(1,1.8,11.7,6.4))
        plt.savefig(output_dir + filename_prefix + "_" + date + ".pdf", dpi=300, bbox_inches = matplotlib.transforms.Bbox.from_extents(1,1.8,11.7,6.4))



    plt.close("all")

if __name__ == "__main__":

    #
    # Parse command line arguments
    #
    parser = argparse.ArgumentParser()

    # Where the data is located
    parser.add_argument(
        "files",
        type=str,
        # metavar="path",
        nargs="*",
        help="files to process",
    )

    # Where to save plots and data
    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        metavar="path",
        help="dir. where output is saved (default: %(default)s)",
    )
    
    # Wich integrator to use
    
    parser.add_argument(
        "--integrator_for_3D_IMF_line",
        default="parker",
        type=str,
        help="How to trace 3D IMF line, (parker, numpa, integral)"
    )
    
    # Which meridional slice to plot
    parser.add_argument(
        "--meridional_plane",
        default="Earth",
        type=str,
        help="Name of object determining which meridional \
                              plane to plot (default: %(default)s)",
    )

    # Which in-situ data to plot
    parser.add_argument(
        "--insitu", default="None", type=str, help="In-situ data to plot: ACE NRT, OMNI, None"
    )

    # Plot log density?
    parser.add_argument("--logn", action="store_true", help="Plot density on log scale")

    parser.add_argument("--lowres", action="store_true", default=True)

    args = parser.parse_args()
    comm, rank, size = _get_mpi_context()
    
    """
    # Display help message if no files given
    if len(args.files) == 0:
        parser.print_help()
        exit()
    """
    
    # Get directory of data
    data_dir = os.path.dirname(os.path.abspath(args.files[0]))
    
    # Get integrator 
    integrator = args.integrator_for_3D_IMF_line
    #
    # Create list of file expressions of type xxx_dateTtime*
    #
    file_expressions = []
    dates = []
    
    args = parser.parse_args()

    inputs = expand_inputs(args.files, patterns=("*.npy","*.npz"))

    print("raw args.files =", args.files)
    print("expanded inputs =", [str(p) for p in inputs])

    if not inputs:
        raise SystemExit("No input files found (check directory and patterns).")


    for f in inputs:
        f = str(f)
        # Get the date
        match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}", f)

        # Skip if a file matching the regex does not exist
        if match is not None:

            # Get date of output
            date = match.group()

            dates.append(date)

            # Split the file name
            fs = f.split(date)

            # Create the file expression
            file_expressions.append(fs[0] + date + "*")

    # Remove duplicate and sort
    file_expressions = sorted(list(set(file_expressions)))
    dates = sorted(list(set(dates)))

    if not file_expressions:
        raise SystemExit("No dated input files found (expected YYYY-MM-DDTHH-MM-SS in file names).")

    assigned_file_expressions = _partition_work(file_expressions, rank, size)

    if size > 1:
        print(
            f"[rank {rank}/{size}] assigned {len(assigned_file_expressions)} "
            f"of {len(file_expressions)} timestamp groups"
        )
        sys.stdout.flush()

    if not assigned_file_expressions:
        print(f"[rank {rank}/{size}] no work assigned")
        sys.stdout.flush()
        raise SystemExit(0)

    # Parse first and last date
    first_date = datetime.datetime.strptime(dates[0], "%Y-%m-%dT%H-%M-%S")
    last_date = datetime.datetime.strptime(dates[-1], "%Y-%m-%dT%H-%M-%S")
    delta_t = (last_date - first_date).days

    #
    # Instantiate planets and spacecraft
    #
    mercury = euhforia.orbit.Mercury()
    venus = euhforia.orbit.Venus()
    earth = euhforia.orbit.Earth()
    mars = euhforia.orbit.Mars()
    sta = euhforia.orbit.STA()
    stb = euhforia.orbit.STB()
    #psp = euhforia.orbit.PSP()
    #solo =  euhforia.orbit.SOLO()
    
    
    heliospheric_objects = (mercury, venus, earth, mars, sta, stb) # , psp, solo)
    # virtual spacecraft
    vs = getattr(euhforia.orbit, args.meridional_plane)()

    #
    # Virtual spacecraft time series data
    #

    # Determine which time series file to open
    simulation_time_series_file = None
    for f in glob.glob(data_dir + "/*.dsv"):
        if args.meridional_plane in f:
            simulation_time_series_file = f

    # Load the data if .dsv files have been found
    df = None
    if simulation_time_series_file is not None:
        print("Loading", simulation_time_series_file)

        df = pd.read_csv(simulation_time_series_file, sep=r"\s+", parse_dates=["date"])

        # If dsv file exists but no data is stored, make it None
        df = df if len(df) > 1 else None

    #
    # Spacecraft in-situ data
    #

    # Which in-situ data to plot
    insitu_dataset = args.insitu.lower()

    # Plot in-situ data?
    plot_time_series = False if insitu_dataset == "none" else True
    plot_insitu = plot_time_series
    if first_date == last_date:
        plot_time_series = False

    # Load in-situ data
    if plot_time_series:

        if insitu_dataset == "default":
            import euhforia.insitu.rt

            insitu = euhforia.insitu.rt.ACERealTimeData(data_dir + "/ace_data/")
        elif insitu_dataset == "omni":
            import euhforia.insitu.omni

            insitu = euhforia.insitu.omni.OMNIData()
        else:
            raise ValueError("Unknown in-situ dataset " + insitu_dataset)

        insitu.start_time = first_date  # df['date'].iloc[0]
        insitu.end_time = last_date  # df['date'].iloc[-1]

        print("Loading", insitu.label, "data")
        try:
            insitu.retrieve()
        except OSError:
            plot_insitu = False

    for idx, f in enumerate(assigned_file_expressions):

        if size > 1:
            print(f"[rank {rank}/{size}] Processing {f}")
        else:
            print("Processing", f)
        sys.stdout.flush()

        # Load data
        data = euhforia.core.io.load_heliospheric_data(f)
        
        date = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}", f).group()
        
        
        # Meridional plane to slice
        lon_slice = vs.position(data.datetime.item(0))[2]
        
        # eart distance 
        R = earth.position(data.datetime)[0]

        def add_vr_time_series(ax3):
            if df is not None:
                ax3.plot(df["date"], df["vr[km/s]"], "b", label="EUHFORIA")

            if plot_insitu:
                ax3.plot(
                    insitu.df["date"],
                    insitu.df["flow_speed"] / 1e3,
                    "-r",
                    label=insitu.label,
                    lw=2,
                    alpha=0.7,
                )
            ax3.plot([data.datetime.item(0), data.datetime.item(0)], [0, 4000], "-k", lw=2, alpha=0.5)

            ax3.set_ylim((200, 1000))
            ax3.set_xlim((first_date, last_date))

            ax3.set_ylabel("Speed [km/s]")

            ax3.yaxis.set_tick_params(labelsize=9)
            ax3.xaxis.set_tick_params(labelsize=12)

            ax3.yaxis.set_major_locator(MultipleLocator(200))
            ax3.yaxis.set_minor_locator(MultipleLocator(100))

            if delta_t < 4:
                ax3.xaxis.set_minor_locator(HourLocator(np.arange(0, 25)))
                ax3.xaxis.set_major_formatter(DateFormatter("%H:%M"))
                ax3.xaxis.set_tick_params(labelsize=10)
            else:
                ax3.xaxis.set_minor_locator(HourLocator(np.arange(0, 25, 6)))
                ax3.xaxis.set_major_formatter(DateFormatter("%b %d"))

            ax3.legend(loc="upper right", bbox_to_anchor=(1.2, 1.05), fontsize=8)

        plot_slice_figure(
            integrator=integrator,
            data_dir=data_dir,
            date=date,
            R=R,
            data=data,
            variable="vr",
            meridional_slice=lon_slice,
            levels=np.linspace(200, 700, 120),
            cmap=euhforia.plot.colormap.citrus,
            heliospheric_objects=heliospheric_objects,
            colorbar_ticks=np.linspace(200, 700, 8),
            output_dir=args.output_dir,
            filename_prefix="vr",
            lowres=args.lowres,
            plot_time_series=plot_time_series,
            add_time_series=add_vr_time_series
        )
        
        
        #
        # Create plot of the scaled number density
        #
        nscaled = np.zeros(data.n.shape)
        
        for idx, r in enumerate(data.grid.center_coords.r):
            nscaled[idx, :, :] = data.n[idx, :, :] * (r / constants.astronomical_unit) ** 2

        data.add_variable(nscaled, name=r"$n \, (r / 1 \mathrm{AU})^2$", unit="cm$^{-3}$")
        
        def add_nscaled_time_series(ax3):
            if args.logn:
                if df is not None:
                    ax3.semilogy(df["date"], df["n[1/cm^3]"], "b", label="EUHFORIA")

                if plot_insitu:
                    ax3.semilogy(
                        insitu.df["date"],
                        insitu.df["proton_number_density"] / 1e6,
                        "-r",
                        label=insitu.label,
                        lw=2,
                        alpha=0.7,
                    )
                ax3.semilogy([data.datetime.item(0), data.datetime.item(0)], [1e-2, 4000], "-k", lw=2, alpha=0.5)
                ax3.set_ylim((0.5, 100))

            else:
                if df is not None:
                    ax3.plot(df["date"], df["n[1/cm^3]"], "b", label="EUHFORIA")

                if plot_insitu:
                    ax3.plot(
                        insitu.df["date"],
                        insitu.df["proton_number_density"] / 1e6,
                        "-r",
                        label=insitu.label,
                        lw=2,
                        alpha=0.7,
                    )
                ax3.plot([data.datetime.item(0), data.datetime.item(0)], [0, 4000], "-k", lw=2, alpha=0.5)
                ax3.set_ylim((0, 30))

                ax3.yaxis.set_major_locator(MultipleLocator(10))
                ax3.yaxis.set_minor_locator(MultipleLocator(5))

            ax3.set_xlim((first_date, last_date))

            ax3.set_ylabel("$n$ [cm$^{-3}$]")

            ax3.yaxis.set_tick_params(labelsize=9)
            ax3.xaxis.set_tick_params(labelsize=12)

            if delta_t < 4:
                ax3.xaxis.set_minor_locator(HourLocator(np.arange(0, 25)))
                ax3.xaxis.set_major_formatter(DateFormatter("%H:%M"))
                ax3.xaxis.set_tick_params(labelsize=10)
            else:
                ax3.xaxis.set_minor_locator(HourLocator(np.arange(0, 25, 6)))
                ax3.xaxis.set_major_formatter(DateFormatter("%b %d"))

            ax3.legend(loc="upper right", bbox_to_anchor=(1.2, 1.05), fontsize=8)

        plot_slice_figure(
            integrator=integrator,
            data_dir=data_dir,
            date=date,
            R=R,
            data=data,
            variable=r"$n \, (r / 1 \mathrm{AU})^2$",
            meridional_slice=lon_slice,
            levels=np.linspace(15, 30, 120),
            cmap=euhforia.plot.colormap.citrus,
            heliospheric_objects=heliospheric_objects,
            colorbar_ticks=np.linspace(15, 40, 8),
            output_dir=args.output_dir,
            filename_prefix="nscaled",
            lowres=args.lowres,
            plot_time_series=plot_time_series,
            add_time_series=add_nscaled_time_series
        )

        #
        # Create plot of the Bclt
        #

        #bcltscaled = np.zeros(data.Bclt.shape)
        #for idx, r in enumerate(data.grid.center_coords.r):
        #    bcltscaled[idx, :, :] = data.Bclt[idx, :, :] * (r / constants.astronomical_unit) ** 2

        #data.add_variable(bcltscaled, name=r"$Bclt \, (r / 1 \mathrm{AU})^2$", unit="nT")
        #
        # Create plot of the Bclt
        #
        plot_slice_figure(
            integrator=integrator,
            data_dir=data_dir,
            date=date,
            R=R,
            data=data,
            variable="Bclt",  # $Bclt \, (r / 1 \mathrm{AU})^2$
            meridional_slice=lon_slice,
            levels=np.linspace(-0.005, 0.005, 120),
            cmap="RdBu",  # euhforia.plot.colormap.citrus,
            heliospheric_objects=heliospheric_objects,
            colorbar_ticks=np.linspace(-0.005, 0.005, 8),
            output_dir=args.output_dir,
            filename_prefix="bclt",
            lowres=args.lowres,
            plot_time_series=False,
        )

        #
        # Create plot of the temperature reconstructed from pressure and number density
        #
        temperature = np.full(data.P.shape, np.nan)
        valid_density = data.n > 0.0
        temperature[valid_density] = data.P[valid_density] / (data.n[valid_density] * 1e6 * constants.kB)
        data.add_variable(temperature, name="T", unit="K")

        plot_slice_figure(
            integrator=integrator,
            data_dir=data_dir,
            date=date,
            R=R,
            data=data,
            variable="T",
            meridional_slice=lon_slice,
            levels=np.linspace(0.0, 2.0e6, 120),
            cmap=euhforia.plot.colormap.citrus,
            heliospheric_objects=heliospheric_objects,
            colorbar_ticks=np.linspace(0.0, 2.0e6, 8),
            output_dir=args.output_dir,
            filename_prefix="T",
            lowres=args.lowres,
            plot_time_series=False,
        )

        #
        # Create plot of the Br
        #

        plot_slice_figure(
            integrator=integrator,
            data_dir=data_dir,
            date=date,
            R=R,
            data=data,
            variable="Br",  # $Bclt \, (r / 1 \mathrm{AU})^2$
            meridional_slice=lon_slice,
            levels=np.linspace(-10, 10, 120),
            cmap="RdBu_r",  # euhforia.plot.colormap.citrus,
            heliospheric_objects=heliospheric_objects,
            colorbar_ticks=np.linspace(-10, 10, 8),
            output_dir=args.output_dir,
            filename_prefix="br",
            lowres=args.lowres,
            plot_time_series=False,
        )

        #
        # Create plot of the Blon
        #

        plot_slice_figure(
            integrator=integrator,
            data_dir=data_dir,
            date=date,
            R=R,
            data=data,
            variable="Blon",  # $Bclt \, (r / 1 \mathrm{AU})^2$
            meridional_slice=lon_slice,
            levels=np.linspace(-4, 4, 120),
            cmap="RdBu",  # euhforia.plot.colormap.citrus,
            heliospheric_objects=heliospheric_objects,
            colorbar_ticks=np.linspace(-4, 4, 8),
            output_dir=args.output_dir,
            filename_prefix="blon",
            lowres=args.lowres,
            plot_time_series=False,
        )

        #
        # Create plot of the Vlon
        #
        plot_slice_figure(
            integrator=integrator,
            data_dir=data_dir,
            date=date,
            R=R,
            data=data,
            variable="vlon",
            meridional_slice=lon_slice,
            levels=np.linspace(-30, 30, 120),
            cmap="RdBu",
            heliospheric_objects=heliospheric_objects,
            colorbar_ticks=np.linspace(-30, 30, 8),
            output_dir=args.output_dir,
            filename_prefix="vlon",
            lowres=args.lowres,
            plot_time_series=False,
        )

        #
        # Create plot of the Vclt
        #
        plot_slice_figure(
            integrator=integrator,
            data_dir=data_dir,
            date=date,
            R=R,
            data=data,
            variable="vclt",
            meridional_slice=lon_slice,
            levels=np.linspace(-50, 50, 120),
            cmap="RdBu",
            heliospheric_objects=heliospheric_objects,
            colorbar_ticks=np.linspace(-50, 50, 8),
            output_dir=args.output_dir,
            filename_prefix="vclt",
            lowres=args.lowres,
            plot_time_series=False,
        )
