# This file is part of EUHFORIA.
#
# Copyright 2016, 2017 Jens Pomoell
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


"""Module for plotting slices of the 3D output data.
"""

import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import scipy.interpolate
from matplotlib import pyplot as plt, gridspec, lines

import coco.core.grid
import euhforia.core.constants as constants
from euhforia.plot import marker



def transversal(R, d, variable, **kwargs):
    
    element_type = coco.core.grid.get_element_type(d.grid, d.data[variable])
    interpolator = scipy.interpolate.RegularGridInterpolator(
        d.grid.get_coordinates(element_type), d.data[variable], method="linear"
    )
    
    heliospheric_objects = kwargs.get("heliospheric_objects", [])
    
    # déinition de l'interpolation 
    lon = d.grid.indomain_edge_coords.lon # longitude 
    clt = d.grid.indomain_edge_coords.clt # latitude
    
    interpolation_coordinates = np.vstack(np.meshgrid(R, clt,lon, indexing="ij")).reshape(3, -1).T # reshape ?
    interpolated_data = interpolator(interpolation_coordinates).reshape((len(clt), len(lon)))
    
    # coordonné a impliqué
    cltmesh, lonmesh = np.meshgrid(clt, lon, indexing="ij")
    Rmesh = R/constants.astronomical_unit
    phi =  lonmesh
    teta = -(cltmesh-np.pi/2)
    
    # génération figure
    fig = kwargs.get("fig", None)
    if fig is None:
        fig = plt.figure(figsize=kwargs.get("figsize", (11, 6)))

    ax = kwargs.get("ax", None)
    if ax is None:
        ax = fig.add_subplot(1, 1, 1)

    
    levels = get_contour_levels(kwargs.get("levels", None), interpolated_data)
    cax = ax.contourf(
        teta,
        phi,
        interpolated_data,
        levels,
        cmap=kwargs.get("cmap", "plasma"),
        extend=kwargs.get("extend", "both"),
    )
    
    # add electric line
    element_type = coco.core.grid.get_element_type(d.grid, d.data['Br'])
    interpolator = scipy.interpolate.RegularGridInterpolator(
        d.grid.get_coordinates(element_type), d.data['Br'], method="linear"
    )
    interpolated_data = interpolator(interpolation_coordinates).reshape((len(clt), len(lon)))
         
    ax.contour(teta, phi, interpolated_data, [0], colors ='w', linewidths = 1)
    
    for i in range(len(clt)):
        for j in range(len(lon)):
            if (i < len(clt)-1 and i > 0) and (j < len(lon)-2 and j > 0) :
                interpolated_data[i][j] = 0
            
            if (i == len(clt)-1 or i == 0) and interpolated_data[i][j] <= 0: 
                interpolated_data[i][j] = -1                
            if (i == len(clt)-1 or i == 0) and interpolated_data[i][j] >= 0:
                interpolated_data[i][j] = 1
            """
            if (i == len(clt)-1 or i == 0) and j > 0 and interpolated_data[i][j]*interpolated_data[i][j-1] < 0:
                ax.plot(clt[i],lon[j], maker ='|', markersize = 18)
            """
        if interpolated_data[i][0] >= 0 :
           interpolated_data[i][0] = 1 
        if interpolated_data[i][len(lon)-2] >= 0 :
           interpolated_data[i][len(lon)-2] = 1 
           
        if interpolated_data[i][0] <= 0 :
           interpolated_data[i][0] = -1 
        if interpolated_data[i][len(lon)-2] <= 0 :
           interpolated_data[i][len(lon)-2] = -1
        """   
        if (j == len(clt)-2 or j == 0) and interpolated_data[i][j]*interpolated_data[i-1][j] < 0:
            ax.plot(clt[i],lon[j], maker ='_', markersize = 4)
        """   

             
    ax.contour(teta, phi, interpolated_data, [-0.999], colors ='b', linestyles ='solid', linewidths = 2)
    ax.contour(teta, phi, interpolated_data, [0.999], colors ='r', linestyles ='solid', linewidths = 2)
    
    plt.xticks(ticks = [-np.pi/2,-1, 0, 1, np.pi/2], labels = ['S90', '', '0°', '', 'N90'])
    plt.yticks(ticks = [-np.pi, -3, -2, -1, 0, 1, 2, 3, np.pi], labels = ['E90', '', '', '','0°', '', '', '', 'W90'])
    
    options = {"color": kwargs.get("guide_color", "0.35"), "alpha": 1}
    ax.set_xlim((-np.pi/2, np.pi/2)) # lat
    ax.set_ylim((-np.pi, np.pi)) # lon
    ax.grid(**options)
    ax.set_xlabel("$Latitude$ [Radian]")
    ax.set_ylabel("$Longitude$ [Radian]")
    
    return fig, ax, cax
    
def equatorial(d, variable, **kwargs):
    """Plot data in heliographic equatorial plane
    """

    #
    # Compute data in equatorial plane
    #

    # Create interpolator to data
    element_type = coco.core.grid.get_element_type(d.grid, d.data[variable])
    interpolator = scipy.interpolate.RegularGridInterpolator(
        d.grid.get_coordinates(element_type), d.data[variable], method="linear"
    )
    
    # Define interpolation points
    r_max = round(d.grid.indomain_edge_coords.r[-1]/constants.astronomical_unit, 1)
    r = kwargs.get("radial_coordinates", np.linspace(0.1, r_max, 256)*constants.astronomical_unit)
    r = np.unique(np.clip(r, d.grid.indomain_edge_coords.r[0], d.grid.indomain_edge_coords.r[-1]))
    
    
    clt = 0.5*np.pi

    lon = d.grid.indomain_edge_coords.lon   # lon :angle de -pi a pi
                                            # r : longeur de 0.1 à 2
            

    
    # Array of points for RegularGridInterpolator
    interpolation_coordinates = np.vstack(np.meshgrid(r, clt, lon, indexing="ij")).reshape(3, -1).T # céation du cercle
    
    # Interpolate
    interpolated_data = interpolator(interpolation_coordinates).reshape((len(r), len(lon)))
    
    # Compute x,y coordinates of plot
    r_plot = r/constants.astronomical_unit

    rmesh, lonmesh = np.meshgrid(r_plot, lon, indexing="ij")

    x = rmesh * np.cos(lonmesh)
    y = rmesh * np.sin(lonmesh)
    
    #
    # Plot
    #

    # Figure handles
    fig = kwargs.get("fig", None)
    if fig is None:
        fig = plt.figure(figsize=kwargs.get("figsize", (11, 6)))

    ax = kwargs.get("ax", None)
    if ax is None:
        ax = fig.add_subplot(1, 1, 1)

    # Contour levels
    levels = get_contour_levels(kwargs.get("levels", None), interpolated_data) #erreur

    # Create the contour plot
    cax = ax.contourf(
        x,
        y,
        interpolated_data,
        levels,
        cmap=kwargs.get("cmap", "plasma"),
        extend=kwargs.get("extend", "both"),
    )

    # Add colorbar
    if kwargs.get("colorbar", True):

        locator = matplotlib.ticker.MaxNLocator(8, min_n_ticks=1)
        default_ticks = locator.tick_values(levels[0], levels[-1])

        cbar = fig.colorbar(cax, ax=ax, ticks=kwargs.get("colorbar_ticks", default_ticks))

    #
    # Add guides
    #

    if kwargs.get("guides", True):

        options = {"color": kwargs.get("guide_color", "0.35"), "alpha": 0.5}

        r_min = r_plot[0]
        r_max = r_plot[-1]

        # Draw circles at given radial coordinates
        circle_radii = np.asarray(kwargs.get("circle_radii", tuple(r + 1 for r in range(round(r_max)))))

        for circle_radius in circle_radii[(circle_radii >= r_min) & (circle_radii <= r_max)]:
            circle = plt.Circle((0.0, 0.0), radius=circle_radius, fill=False, **options)
            ax.add_patch(circle)

        # Draw dashed circles at given radial coordinates
        dashed_circle_radii = np.asarray(kwargs.get("dashed_circle_radii", tuple(r + 0.5 for r in range(round(r_max)))))

        for circle_radius in dashed_circle_radii[(dashed_circle_radii >= r_min) & (dashed_circle_radii <= r_max)]:
            circle = plt.Circle((0.0, 0.0), radius=circle_radius, fill=False, ls="dashed", **options)
            ax.add_patch(circle)

        # Draw y = 0 line
        line = plt.Line2D((-r_max, r_max), (0.0, 0.0), **options)
        ax.add_line(line)

        # Draw x = 0 line
        line = plt.Line2D((0.0, 0.0), (-r_max, r_max), **options)
        ax.add_line(line)

        # Draw lon = 45 deg diagonal line
        line = plt.Line2D(
            (-r_max/np.sqrt(2.0), r_max/np.sqrt(2.0)), (-r_max/np.sqrt(2.0), r_max/np.sqrt(2.0)), **options
        )
        ax.add_line(line)

        # Draw lon = -45 deg diagonal line
        line = plt.Line2D(
            (-r_max/np.sqrt(2.0), r_max/np.sqrt(2.0)), (r_max/np.sqrt(2.0), -r_max/np.sqrt(2.0)), **options
        )
        ax.add_line(line)

        # Cover the lines above in the middle
        circle = plt.Circle((0.0, 0.0), radius=r_min, color="w", fill=True, zorder=4)
        ax.add_patch(circle)

    ax.set_aspect("equal")

    ax.set_xlabel("$x$ [HEEQ AU]")
    ax.set_ylabel("$y$ [HEEQ AU]")

    if kwargs.get("write_date", True):
        ax.text(0.01, 1.025, "{0: <40}".format(d.datetime.item(0).strftime("%Y-%m-%d %H:%M")), transform=ax.transAxes)

    if kwargs.get("colorbar", True):

        unit_str = str("  [" + d.units[variable] + "]") if d.units[variable] is not None else ""

        colorbar_default_label = str(variable) + unit_str

        cbar.ax.set_ylabel(kwargs.get("colorbar_label", colorbar_default_label), labelpad=10.0) 
    
    # add electric line
    element_type = coco.core.grid.get_element_type(d.grid, d.data['Br'])
    interpolator = scipy.interpolate.RegularGridInterpolator(
        d.grid.get_coordinates(element_type), d.data['Br'], method="linear"
    )
    interpolated_data = interpolator(interpolation_coordinates).reshape((len(r), len(lon)))
    
    ax.contour(x, y, interpolated_data, [0], colors ='w', linewidths = 1)  
    
    for i in range(len(r)):
        for j in range(len(lon)):
            if i < len(r)-1:
                interpolated_data[i][j] = 0
            if i == len(r)-1 and interpolated_data[i][j] <= 0: 
                interpolated_data[i][j] = -1                
            if i == len(r)-1 and interpolated_data[i][j] >= 0:
                interpolated_data[i][j] = 1
    c = plt.Circle((0.0, 0.0),radius = r_max, linewidth = 2, fill =False)
    ax.add_patch(c)       
    ax.contour(x, y, interpolated_data, [-0.99999], colors ='b', linestyles ='solid', linewidths = 2)
    ax.contour(x, y, interpolated_data, [0.999999], colors ='r', linestyles ='solid', linewidths = 2)
            
    return fig, ax, cax


def meridional(d, variable, lon=0.0, **kwargs):
    """Plot data in a meridional plane
    """

    #
    # Compute data in meridional plane
    #

    # Create interpolator to data
    element_type = coco.core.grid.get_element_type(d.grid, d.data[variable])
    interpolator = scipy.interpolate.RegularGridInterpolator(
        d.grid.get_coordinates(element_type), d.data[variable], method="linear"
    )

    # Define interpolation points
    r_max = round(d.grid.indomain_edge_coords.r[-1]/constants.astronomical_unit, 1)
    r = kwargs.get("radial_coordinates", np.linspace(0.1, r_max, 256)*constants.astronomical_unit)
    r = np.unique(np.clip(r, d.grid.indomain_edge_coords.r[0], d.grid.indomain_edge_coords.r[-1]))

    clt = d.grid.indomain_edge_coords.clt
    
    # Wrap longitude
    if lon > d.grid.indomain_edge_coords.lon[-1]:
        lon -= 2.0*np.pi
    if lon < d.grid.indomain_edge_coords.lon[0]:
        lon += 2.0*np.pi

    # Array of points for RegularGridInterpolator   
    interpolation_coordinates = np.vstack(np.meshgrid(r, clt, lon, indexing="ij")).reshape(3, -1).T
    
    # Interpolate
    interpolated_data = interpolator(interpolation_coordinates).reshape((len(r), len(clt)))

    # Compute x,y coordinates of plot
    r_plot = r/constants.astronomical_unit

    rmesh, cltmesh = np.meshgrid(r_plot, clt, indexing="ij")
    
    x = rmesh*np.sin(cltmesh)
    y = rmesh*np.cos(cltmesh)

    #
    # Plot
    #

    # Figure handles
    fig = kwargs.get("fig", None)
    if fig is None:
        fig = plt.figure(figsize=kwargs.get("figsize", (11, 6)))

    ax = kwargs.get("ax", None)
    if ax is None:
        ax = fig.add_subplot(1, 1, 1)

    levels = get_contour_levels(kwargs.get("levels", None), interpolated_data)

    # Plot contour
    cax = ax.contourf(
        x,
        y,
        interpolated_data,
        levels,
        cmap=kwargs.get("cmap", "plasma"),
        extend=kwargs.get("extend", "both"),
    )

    # Add colorbar
    if kwargs.get("colorbar", True):

        locator = matplotlib.ticker.MaxNLocator(8, min_n_ticks=1)
        default_ticks = locator.tick_values(levels[0], levels[-1])

        cbar = fig.colorbar(cax, ax=ax, ticks=kwargs.get("colorbar_ticks", default_ticks))
    
    #
    # Add guides
    #
    
    if kwargs.get("guides", True):

        options = {"color": kwargs.get("guide_color", "0.35"), "alpha": 0.5}

        r_min = r_plot[0]
        r_max = r_plot[-1]

        # Draw arcs at given radial coordinates
        circle_radii = np.asarray(kwargs.get("circle_radii", tuple(r + 1 for r in range(round(r_max)))))

        for circle_radius in circle_radii[(circle_radii >= r_min) & (circle_radii <= r_max)]:
            arc = matplotlib.patches.Arc(
                (0.0, 0.0),
                width=2.0*circle_radius,
                height=2.0*circle_radius,
                angle=0.0,
                theta1=90.0 - d.grid.indomain_edge_coords.clt[-1]*180.0/np.pi,
                theta2=90.0 - d.grid.indomain_edge_coords.clt[0]*180.0/np.pi,
                fill=False,
                **options
            )
            ax.add_patch(arc)

        # Draw dashed arcs at given radial coordinates
        dashed_circle_radii = np.asarray(kwargs.get("dashed_circle_radii", tuple(r + 0.5 for r in range(round(r_max)))))

        for circle_radius in dashed_circle_radii[(dashed_circle_radii >= r_min) & (dashed_circle_radii <= r_max)]:
            arc = matplotlib.patches.Arc(
                (0.0, 0.0),
                width=2.0*circle_radius,
                height=2.0*circle_radius,
                angle=0.0,
                theta1=90.0 - d.grid.indomain_edge_coords.clt[-1]*180.0/np.pi,
                theta2=90.0 - d.grid.indomain_edge_coords.clt[0]*180.0/np.pi,
                fill=False,
                ls="dashed",
                **options
            )
            ax.add_patch(arc)

        # Draw line of heliographic equator
        line = plt.Line2D((r_min, r_max), (0.0, 0.0), **options)
        ax.add_line(line)

        # Draw line 30 deg to North from heliographic equator
        line = plt.Line2D(
            (r_min*np.cos(30.0*np.pi/180.0), r_max*np.cos(30.0*np.pi/180.0)),
            (r_min*np.sin(30.0*np.pi/180.0), r_max*np.sin(30.0*np.pi/180.0)),
            **options
        )
        ax.add_line(line)

        # Draw line 30 deg to South from heliographic equator
        line = plt.Line2D(
            (r_min*np.cos(-30.0*np.pi/180.0), r_max*np.cos(-30.0*np.pi/180.0)),
            (r_min*np.sin(-30.0*np.pi/180.0), r_max*np.sin(-30.0*np.pi/180.0)),
            **options
        )
        ax.add_line(line)
    
    ax.set_ylim((-r_plot[-1], r_plot[-1]))

    ax.set_aspect("equal")
    
    if np.isclose(lon, 0.0):
        ax.set_xlabel("$x$ [HEEQ AU]")
    else:
        ax.set_xlabel("$\mathrm{Distance}$ [AU]")

    ax.set_ylabel("$z$ [HEEQ AU]")

    if kwargs.get("write_date", True):
        ax.text(0, 1.025, "{0: <40}".format(d.datetime.item(0).strftime("%Y-%m-%d %H:%M")), transform=ax.transAxes)

    if kwargs.get("colorbar", True):

        unit_str = str("  [" + d.units[variable] + "]") if d.units[variable] is not None else ""
        colorbar_default_label = str(variable) + unit_str

        cbar.ax.set_ylabel(kwargs.get("colorbar_label", colorbar_default_label), labelpad=10.0)
    
    # add electric line
    element_type = coco.core.grid.get_element_type(d.grid, d.data['Br'])
    interpolator = scipy.interpolate.RegularGridInterpolator(
        d.grid.get_coordinates(element_type), d.data['Br'], method="linear"
    )
    interpolated_data = interpolator(interpolation_coordinates).reshape((len(r), len(clt)))
    ax.contour(x, y, interpolated_data, [0], colors ='w', linewidths = 1)  # 
    
    arc = matplotlib.patches.Arc(
        (0.0, 0.0),
        width=r_max*2,
        height=r_max*2,
        angle=0.0,
        theta1=90.0 - d.grid.indomain_edge_coords.clt[-1]*180.0/np.pi,
        theta2=90.0 - d.grid.indomain_edge_coords.clt[0]*180.0/np.pi,
        fill=False,
        linewidth =2
    )
    ax.add_patch(arc)
    
    for j in range(len(clt)):
        for i in range(len(r)):
            
            if (i < len(r)-1 and i > 0) and (j < len(clt)-1 and j > 0) :
                interpolated_data[i][j] = 0
            
            #ok #pour tout les angles: tracer l'arc de cercle
            if (i == len(r)-1 or i == 0) and interpolated_data[i][j] <= 0: 
                interpolated_data[i][j] = -1                
            if (i == len(r)-1 or i == 0) and interpolated_data[i][j] >= 0:
                interpolated_data[i][j] = 1
            
            #pour clt =0 et clt =pi trancer les barres
            if (j == len(clt)-1 or j == 0) and interpolated_data[i][j] <= 0: 
                interpolated_data[i][j] = -1                
            if (j == len(clt)-1 or j == 0) and interpolated_data[i][j] >= 0:
                interpolated_data[i][j] = 1
    
    
    ax.contour(x, y, interpolated_data, [-0.999], colors ='b', linestyles ='solid', linewidths = 2)
    ax.contour(x, y, interpolated_data, [0.9999], colors ='r', linestyles ='solid', linewidths = 2)
    return fig, ax, cax


def equatorial_and_meridional_and_transversal(integrator, data_dir, date, R, d, variable, meridional_slice, levels, **kwargs):
    """Plot equatorial and meridional plane in same figure
    """

    #
    # Create figure
    #
    fig = kwargs.get("fig", None)
    if fig is None:
        fig = plt.figure(figsize=kwargs.get("figsize", (10, 5.5)))

    ax = kwargs.get("ax", None)
    if ax is None:
        gs = matplotlib.gridspec.GridSpec(3, 3, width_ratios=[4, 2.0, np.pi], height_ratios=[3, 1, 1])
        
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax_t = plt.subplot(gs[2])
    else:
        ax1 = ax[0]
        ax2 = ax[1]
        ax_t = ax[2]

    cmap = kwargs.get("cmap", "plasma")
    colorbar_ticks = kwargs.get("colorbar_ticks", None)

    #
    # Equatorial plot
    #
    fig, ax1, cax = equatorial(
        d,
        variable,
        fig=fig,
        ax=ax1,
        levels=levels,
        cmap=cmap,
        colorbar=False,
        write_date=kwargs.get("write_date", True),
    )

    #
    # Meridional plot
    #
    fig, ax2, cax = meridional(
        d, variable, fig=fig, ax=ax2, lon=meridional_slice, levels=levels, cmap=cmap, colorbar=False, write_date=False
    )
    
    #
    # transversal plot
    #
    fig, ax_t, cax = transversal(
        R, d, variable, fig=fig, ax=ax_t, levels=levels, cmap=cmap, colorbar=False, write_date=False
    )
    # ax2.set_xlim((0.0, 1.0));

    #
    # Add colorbar
    #
    cbar = fig.colorbar(cax, ax=ax_t, pad=0.25, ticks=colorbar_ticks, fraction=0.06)

    unit_str = str("  [" + d.units[variable] + "]") if d.units[variable] is not None else ""
    cbar.ax.set_ylabel(str(variable) + unit_str, labelpad=10.0)
    
    #
    # Add planets and SC
    #
    heliospheric_objects = kwargs.get("heliospheric_objects", [])
    
    
    #
    # add magnetic lines
    #    

    if integrator == 'parker':
        import euhforia.plot.magnetic_lines_parker
        fig_e, fig_m = euhforia.plot.magnetic_lines_parker.lines(d, heliospheric_objects=heliospheric_objects)
    if integrator == 'numpa':
        import euhforia.plot.magnetic_lines_integral_numpa
        fig_e, fig_m = euhforia.plot.magnetic_lines_integral_numpa.lines(data_dir, date, d, heliospheric_objects=heliospheric_objects)
    if integrator == 'integral':
        import euhforia.plot.magnetic_lines_integral
        fig_e, fig_m = euhforia.plot.magnetic_lines_integral.lines(data_dir, date, d, heliospheric_objects=heliospheric_objects)
                                                               
    i = 1 # rajouter artificiellement sta et stb        
    for obj in heliospheric_objects:
        r, lat, lon = obj.position(d.datetime)
        
        # Don't plot if object is further than outer boundary of data
        if r > max(d.grid.domain.r):
            continue

        ax1.plot(
            r * np.cos(lon) / constants.astronomical_unit,
            r * np.sin(lon) / constants.astronomical_unit,
            label=obj.name,
            markersize=7,
            marker=marker.style[obj.name]["Marker"],
            color=marker.style[obj.name]["color"],
        )
         
        x, y = fig_e[i-1][0], fig_e[i-1][1] # le truc à faire

        ax1.plot(x, y, linestyle = '-', color = 'black')
        ax1.plot(x, y, linestyle="dashed", dashes=(2,2) ,color = 'w')
        
        if kwargs.get("show_legend", True):
            ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=4, numpoints=1, fancybox=True, fontsize=7)

        if np.isclose(lon, meridional_slice):
            ax2.plot(
                r*np.cos(lat)/constants.astronomical_unit,
                r*np.sin(lat)/constants.astronomical_unit,
                label=obj.name,
                markersize=7,
                marker=marker.style[obj.name]["Marker"],
                color=marker.style[obj.name]["color"],
            )
            
        x, y = fig_m[2][0], fig_m[2][1] 

        ax2.plot(x, y, linestyle = '-', color = 'black')
        ax2.plot(x, y, linestyle="dashed", dashes=(2,2) ,color = 'w')
        
        l1 = lines.Line2D([], [], linewidth=4, linestyle="solid", color='w')
        l2 = lines.Line2D([], [], linewidth=4, linestyle="dotted", color='black')
        l3 = lines.Line2D([], [], linewidth=2, color='w')
        
        ax2.legend(handles=((l1,l2),l3), labels=("3D IMF Line","Current Sheet"), handlelength=3, loc="lower center", bbox_to_anchor=(0.3, -0.5), facecolor ='lightgrey')
        
        l4 = lines.Line2D([], [], linewidth=7, color='r')
        l5 = lines.Line2D([], [], linewidth=7, color='b')
        
        ax_t.legend(handles=(l4,l5), labels=("+","-"), handlelength=2, loc="lower center", bbox_to_anchor=(0.1, -0.5), facecolor ='lightgrey', title = 'IMF polarity')
        
        #plt.legend(handles=[l3], labels=("label",), handlelength=3, loc="lower right",bbox_to_anchor=(1.5, -0.5), facecolor ='lightgrey')
        
        if r == R or i ==6 or i==5:
            
            ax_t.plot(
                lat,
                lon,
                label=obj.name,
                markersize=7,
                marker=marker.style[obj.name]["Marker"],
                color=marker.style[obj.name]["color"],
            )
        
        i += 1
    
    
    return fig, ax1, ax2, ax_t


def equatorial_and_meridional(d, variable, meridional_slice, levels, **kwargs):
    """Plot equatorial and meridional plane in same figure
    """

    #
    # Create figure
    #
    fig = kwargs.get("fig", None)
    if fig is None:
        fig = plt.figure(figsize=kwargs.get("figsize", (10, 5.5)))

    ax = kwargs.get("ax", None)
    if ax is None:
        gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[1.38, 1.0])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
    else:
        ax1 = ax[0]
        ax2 = ax[1]

    cmap = kwargs.get("cmap", "plasma")
    colorbar_ticks = kwargs.get("colorbar_ticks", None)
    #cbar_tick_size = kwargs.get("colorbar_ticks_size", cbar_tick_size)

    #
    # Equatorial plot
    #
    fig, ax1, cax = equatorial(
        d,
        variable,
        fig=fig,
        ax=ax1,
        levels=levels,
        cmap=cmap,
        colorbar=False,
        write_date=kwargs.get("write_date", True),
    )

    #
    # Meridional plot
    #//
    fig, ax2, cax = meridional(
        d, variable, fig=fig, ax=ax2, lon=meridional_slice, levels=levels, cmap=cmap, colorbar=False, write_date=False
    )

    # ax2.set_xlim((0.0, 1.0));

    #
    # Add colorbar
    #
    cbar = fig.colorbar(cax, ax=ax2, pad=0.1, ticks=colorbar_ticks, fraction=0.08, extend="both") #default fraction=0.06

    unit_str = str("  [" + d.units[variable] + "]") if d.units[variable] is not None else ""
    cbar.ax.set_ylabel(str(variable) + unit_str, labelpad=10.0)

    #
    # Add planets and SC
    #
    heliospheric_objects = kwargs.get("heliospheric_objects", [])

    for obj in heliospheric_objects:

        r, lat, lon = obj.position(d.datetime)

        # Don't plot if object is further than outer boundary of data
        if r > max(d.grid.domain.r):
            continue

        ax1.plot(
            r * np.cos(lon) / constants.astronomical_unit,
            r * np.sin(lon) / constants.astronomical_unit,
            label=obj.name,
            markersize=7,
            marker=marker.style[obj.name]["Marker"],
            color=marker.style[obj.name]["color"],
        )

        if kwargs.get("show_legend", True):
            ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.23), ncol=4, numpoints=1, fancybox=True, fontsize=7)

        if np.isclose(lon, meridional_slice):
            ax2.plot(
                r*np.cos(lat)/constants.astronomical_unit,
                r*np.sin(lat)/constants.astronomical_unit,
                label=obj.name,
                markersize=7,
                marker=marker.style[obj.name]["Marker"],
                color=marker.style[obj.name]["color"],
            )

    return fig, ax1, ax2



def get_contour_levels(levels, data):
    """Determine contour levels for slice plots
    """

    if levels is None:
        locator = matplotlib.ticker.MaxNLocator(7 + 1, min_n_ticks=1)
        lev = locator.tick_values(data.min(), data.max())
        levels = np.linspace(lev[0], lev[-1], 64)
    elif isinstance(levels, (list, tuple, np.ndarray)):
        levels = np.asarray(levels)
    else:
        num_levels = int(levels)

        locator = matplotlib.ticker.MaxNLocator(num_levels + 1, min_n_ticks=1)
        lev = locator.tick_values(data.min(), data.max())
        levels = np.linspace(lev[0], lev[-1], num_levels)

    return levels
