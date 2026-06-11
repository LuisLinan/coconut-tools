import numpy as np
import pyvista as pv
import pandas as pd
import os
from bisect import bisect_left
from itertools import cycle
import cmocean
from cmap import Colormap
import re
from glob import glob
from datetime import datetime
import cv2
from coconut_tools.tools.color import citrus, citrus_low


def get_sorted_datetimes(folder_path):
    file_list = os.listdir(folder_path)

    vts_files = [f for f in file_list if f.endswith('.vts')]

    file_datetime_tuples = []
    for file_name in vts_files:
        datetime_str = file_name.split('_')[1].replace('T', ' ').replace('-', ':').split('.')[0]
        file_datetime = datetime.strptime(datetime_str, '%Y:%m:%d %H:%M:%S')
        file_datetime_tuples.append((file_datetime, file_name))

    file_datetime_tuples.sort()

    sorted_datetimes = [t[0] for t in file_datetime_tuples]
    sorted_file = [t[1] for t in file_datetime_tuples]

    return sorted_datetimes, sorted_file


def extract_number(filename):
    match = re.search(r'data_(\d+)\.vts', filename)
    if match:
        return int(match.group(1))
    return None


def read_dsv_file(file_path):
    # The .dsv files are whitespace-delimited, not comma-delimited.
    data = pd.read_csv(file_path, sep=r"\s+")
    data['date'] = pd.to_datetime(data['date'])
    return data


def get_mpl_cmap(name):
    return Colormap(name).to_mpl()


def create_dictplanet(folder):
    planet_data = {}

    for file_name in os.listdir(folder):
        if file_name.endswith(".dsv"):
            # Extraire le nom de la planète/satellite
            planet_name = file_name.split('_')[1].split('.')[0]

            # Lire le fichier et structurer les données
            file_path = os.path.join(folder, file_name)
            data = read_dsv_file(file_path)

            # Sauvegarder les données dans le dictionnaire
            position_data = data[['date', 'r[AU]', 'clt[rad]', 'lon[rad]', 'vr[km/s]']]
            planet_data[planet_name] = position_data.set_index('date').to_dict('index')
    return planet_data


def slice2D(filevtu, filevts, dsvdirectory, target_date, outputdir, idx, ecart, angle, save=False, variable='Vr'):
    if save:
        # Construction des chemins
        bmp_file = f'{outputdir}pyvista_{variable}_{idx:04d}.bmp'
        png_file = f'{outputdir}pyvista_{variable}_{idx:04d}.png'

        # Vérification de l'existence
        if os.path.exists(bmp_file) and os.path.exists(png_file):
            print(f"Files {bmp_file} and {png_file} already exist. Skipping.")
            return
        
    mesh = pv.read(filevtu)
    radii = np.sqrt(mesh.points[:, 0] ** 2 + mesh.points[:, 1] ** 2 + mesh.points[:, 2] ** 2)
    mask = radii <= 20.5
    mesh = mesh.extract_points(mask)

    transform = pv.transformations.axis_angle_rotation([0, 0, 1], angle)
    mesh.transform(transform)

    clip_sphere = pv.Sphere(radius=1.01, center=(0, 0, 0))

    # Clipping du mesh avec la sphère
    clipped_mesh = mesh.clip_surface(clip_sphere)

    planet_data = create_dictplanet(dsvdirectory)

    planet, data = list(planet_data.items())[0]

    # Trouver la position la plus proche pour la première planète à la target_date
    dates = list(data.keys())
    nearest_date_index = find_nearest_date_index(dates, target_date)
    nearest_date = dates[nearest_date_index]

    planet_positions = {}
    for planet, data in planet_data.items():
        try:
            planet_positions[planet] = data[nearest_date]
        except:
            continue

    mesh_vts = pv.read(filevts)

    conversion_factor = 215
    mesh_vts.points *= conversion_factor

    inner_radius_vts = 21.5
    radii_vts = np.sqrt(mesh_vts.points[:, 0] ** 2 + mesh_vts.points[:, 1] ** 2 + mesh_vts.points[:, 2] ** 2)
    mask_vts = radii_vts > inner_radius_vts
    mesh_vts = mesh_vts.extract_points(mask_vts)

    outer_radius = 230
    radii = np.sqrt(mesh_vts.points[:, 0] ** 2 + mesh_vts.points[:, 1] ** 2)
    mask = radii < outer_radius
    mesh_vts = mesh_vts.extract_points(mask)

    if variable == 'density':

        # EUHFORIA cm-3 to m-3
        n = mesh_vts.cell_data['n']
        rho = n * 1e6

        cell_centers = mesh_vts.cell_centers()
        cell_points = cell_centers.points
        radii_squared = np.sum(cell_points[:, :2] ** 2, axis=1)

        #vmin=np.min(rho*radii_squared)
        #vmax=np.max(rho*radii_squared)

        vmin = 2.39e11
        vmax = 1.85e13

        mesh_vts.cell_data['rho'] = rho * radii_squared

        # COCONUT vtu g/m-3 to m-3
        n = mesh.point_data['rho']
        rho = n / 1.67e-30
        radii_squared = np.sum(mesh.points[:, :2] ** 2, axis=1)

        mesh.point_data['rho'] = rho * radii_squared
    elif variable == 'T':

        n = mesh_vts.cell_data['n']

        # cm-3 to m-3
        rho = n * 1e6

        # P  ( Pa ? )
        P = mesh_vts.cell_data['P']

        vmin = 5.3e4
        vmax = 3.7e6

        # T = P/(nkb)
        mesh_vts.cell_data['T'] = 1.27 * P / (rho * 1.38e-23)
    elif variable == 'Bclt':
        mesh.point_data['Bclt'] = mesh.point_data['Btheta'] * 1e5 
    elif variable == 'Blon':
        mesh.point_data['Blon'] = mesh.point_data['Bphi'] * 1e5
    elif variable == 'Br':
        mesh.point_data['Br'] = (mesh.point_data['Br'] * 1e5)

    slice_vts = mesh_vts.slice(normal='z', origin=(0, 0, 0))
    # slice_vts_m = mesh_vts.slice(normal='y', origin=(0, 0, 0))
    # slice_vts = slice_vts.smooth(n_iter=50, relaxation_factor=0.01)

    # Extraire la coupe équatoriale à z = 0
    plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1))

    # Filtrer le maillage pour ne conserver que les points où z = 0
    slice_equatorial = mesh.slice(normal='z', origin=(0, 0, 0))
    #slice_equatorial_m = mesh.slice(normal='y', origin=(0, 0, 0))

    color_palette = cycle(["blue", "red", "green", "magenta", "purple", "orange", "cyan", "lime", "yellow", "pink"])
    color_dict = {planet: next(color_palette) for planet in planet_positions.keys()}
    orbit_line_width = 3.2
    orbit_opacity = 0.92
    body_radius = 5.0
    grid_color = "#7e7e7e"
    outer_grid_color = "#8a8a8a"
    label_color = "#767676"
    inset_center = (-296, -156, 0)
    inset_size = 360

    plotter = pv.Plotter(off_screen=True)
    plotter.set_background('white')
    plotter.add_mesh(clipped_mesh, scalars='Br', cmap='bwr', show_scalar_bar=False, clim=[-10, 10])

    if variable == 'Vr':
        plotter.add_mesh(slice_equatorial, scalars='Vr', cmap=get_mpl_cmap('tol:nightfall'), show_scalar_bar=False,
                         clim=[300, 800])
    elif variable == 'density':
        plotter.add_mesh(slice_equatorial, scalars='rho', cmap=citrus, show_scalar_bar=False, clim=[vmin, vmax])
    elif variable == 'T':
        plotter.add_mesh(slice_equatorial, scalars='T', cmap=get_mpl_cmap('cmocean:balance'), show_scalar_bar=False,
                         clim=[vmin, vmax])
    elif variable == 'Bz':
        plotter.add_mesh(slice_equatorial, scalars='Bz', cmap=get_mpl_cmap('cmocean:balance'), show_scalar_bar=False)
    elif variable == 'Bclt':
        plotter.add_mesh(slice_equatorial, scalars='Bclt', cmap=get_mpl_cmap('cmocean:balance'), show_scalar_bar=False,
                         clim=[-600, 600])

        # plotter.add_mesh(slice_equatorial, scalars='rho', cmap=Colormap('matplotlib:nipy_spectral'), show_scalar_bar=False,clim=[vmin, vmax])

    else:
        print('not valid variable')

    output_image_path = f'E:/coupling_movie/temp/visualization.png'
    plotter.view_xy()
    plotter.camera.position = (0, 0, 150)

    plotter.screenshot(output_image_path, transparent_background=True)

    plotter.close()

    if save:
        plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    else:
        plotter = pv.Plotter(window_size=[1920, 1080])

    plotter.set_background('white')
    plotter.add_text(
        f"{target_date.strftime('%Y-%m-%d %H:%M:%S')}",
        position='upper_right',
        color="#222222",
        font_size=22,
        viewport=True,
    )

    w_S = 2.6 * 10 ** (-6)
    rayon_solaire_km = 695700
    legend_entries = []
    for planet, pos in planet_positions.items():
        r = pos['r[AU]'] * 215
        AU_to_km = 149597870.7
        if r > 230:
            continue
        clt = pos['clt[rad]']
        lon = pos['lon[rad]']
        vr = pos['vr[km/s]'] / rayon_solaire_km

        x = r * np.sin(clt) * np.cos(lon)
        y = r * np.sin(clt) * np.sin(lon)
        z = r * np.cos(clt)

        r_line = np.linspace(0, 230, 200)  # Distance radiale de 0 à r
        phi_line = -w_S * (r_line - r) / vr + lon  # Angle phi en fonction de la distance radiale

        # Coordonnées x, y, z des points le long de la spirale
        x_spiral = r_line * np.sin(clt) * np.cos(phi_line)
        y_spiral = r_line * np.sin(clt) * np.sin(phi_line)
        z_spiral = r_line * np.cos(clt)

        # z=0
        color = color_dict.get(planet,
                               "white")  # Utilise "white" si la planète n'est pas dans le dictionnaire de couleurs
        plotter.add_mesh(
            pv.Sphere(radius=body_radius, center=(x, y, z)),
            color=color,
            label=planet,
            smooth_shading=False,
            lighting=False,
            ambient=1.0,
            diffuse=0.0,
            specular=0.0,
        )
        legend_entries.append([planet, color])

        points = np.column_stack((x_spiral, y_spiral, z_spiral))
        line = pv.lines_from_points(points)
        plotter.add_mesh(line, color=color, line_width=orbit_line_width, opacity=orbit_opacity)

    plotter.add_mesh(clipped_mesh, scalars='Br', cmap='bwr', show_scalar_bar=False, clim=[-10, 10])

    if variable == 'Vr':
        plotter.add_mesh(slice_equatorial, scalars='Vr', cmap=get_mpl_cmap('tol:nightfall'), show_scalar_bar=False,
                         clim=[300, 600])
        plotter.add_mesh(slice_vts, scalars='vr', cmap=get_mpl_cmap('tol:nightfall'), show_scalar_bar=False,
                         clim=[300, 600])
        plotter.add_scalar_bar(title='Vr\n[km/s]', vertical=True, title_font_size=28, label_font_size=24,
                               position_x=0.88, position_y=0.08, width=0.032, height=0.34, n_labels=5,
                               fmt="%.0f", color="#222222")

    elif variable == 'density':
        plotter.add_mesh(slice_equatorial, scalars='rho', cmap=citrus, show_scalar_bar=False, clim=[vmin, vmax])
        plotter.add_mesh(slice_vts, scalars='rho', cmap=citrus, show_scalar_bar=False, clim=[vmin, vmax])
        plotter.add_scalar_bar(title='Density * r² \n [Rs² * m-3 ]', vertical=True, title_font_size=55, label_font_size=48, position_x=0.86)
    elif variable == 'T':
        plotter.add_mesh(slice_vts, scalars='T', cmap=get_mpl_cmap('cmocean:balance'), show_scalar_bar=False,
                         clim=[vmin, vmax])
        plotter.add_mesh(slice_equatorial, scalars='T', cmap=get_mpl_cmap('cmocean:balance'), show_scalar_bar=False,
                         clim=[vmin, vmax])
        plotter.add_scalar_bar(title='T\n[K]', vertical=True, title_font_size=24, label_font_size=22,
                               position_x=0.88, position_y=0.08, width=0.032, height=0.34, n_labels=5,
                               color="#222222")
    elif variable == 'Br':
        plotter.add_mesh(slice_vts, scalars='Br', cmap=get_mpl_cmap('cmocean:balance'), show_scalar_bar=False,
                         clim=[-100, 100])
        plotter.add_mesh(slice_equatorial, scalars='Br', cmap=get_mpl_cmap('cmocean:balance'), show_scalar_bar=False,
                         clim=[-100, 100])
        #plotter.add_mesh(slice_vts_m, scalars='Br', cmap=get_mpl_cmap('cmocean:balance'), show_scalar_bar=False,
        #                 clim=[-100, 100])
        #, scalars='Br', cmap=get_mpl_cmap('cmocean:balance'), show_scalar_bar=False,
        #                 clim=[-100, 100])
        plotter.add_scalar_bar(title='Br\n[nT]', vertical=True, title_font_size=24, label_font_size=22,
                               position_x=0.88, position_y=0.08, width=0.032, height=0.34, n_labels=5,
                               color="#222222")
    elif variable == 'Bclt':
        plotter.add_mesh(slice_vts, scalars='Bclt', cmap=get_mpl_cmap('cmocean:balance'), show_scalar_bar=False,
                         clim=[-50, 50])
        plotter.add_mesh(slice_equatorial, scalars='Bclt', cmap=get_mpl_cmap('cmocean:balance'), show_scalar_bar=False,
                         clim=[-50, 50])
        #plotter.add_mesh(slice_vts_m, scalars='Bclt', cmap=get_mpl_cmap('cmocean:balance'), show_scalar_bar=False,
        #                 clim=[-50, 50])
        #plotter.add_mesh(slice_equatorial_m, scalars='Bclt', cmap=get_mpl_cmap('cmocean:balance'), show_scalar_bar=False,
        #                 clim=[-50, 50])
        plotter.add_scalar_bar(title='Bclt\n[nT]', vertical=True, title_font_size=24, label_font_size=22,
                               position_x=0.88, position_y=0.08, width=0.032, height=0.34, n_labels=5,
                               color="#222222")
    elif variable == 'Blon':
        plotter.add_mesh(slice_vts, scalars='Blon', cmap=get_mpl_cmap('cmocean:balance'), show_scalar_bar=False,
                         clim=[-50, 50])
        plotter.add_mesh(slice_equatorial, scalars='Blon', cmap=get_mpl_cmap('cmocean:balance'), show_scalar_bar=False,
                         clim=[-50, 50])
        #plotter.add_mesh(slice_vts_m, scalars='Blon', cmap=get_mpl_cmap('cmocean:balance'), show_scalar_bar=False,
        #                 clim=[-50, 50])
        #plotter.add_mesh(slice_equatorial_m, scalars='Blon', cmap=get_mpl_cmap('cmocean:balance'), show_scalar_bar=False,
        #                 clim=[-50, 50])
        plotter.add_scalar_bar(title='Blon\n[nT]', vertical=True, title_font_size=24, label_font_size=22,
                               position_x=0.88, position_y=0.08, width=0.032, height=0.34, n_labels=5,
                               color="#222222")
    else:
        print('not valid variable')

    scalar_bar_actor = list(plotter.scalar_bars.values())[-1]
    scalar_titles = {
        'Vr': 'Vr\n[km/s]',
        'density': 'Density x r²\n[Rs² m⁻³]',
        'T': 'T\n[K]',
        'Br': 'Br\n[nT]',
        'Bclt': 'Bclt\n[nT]',
        'Blon': 'Blon\n[nT]',
    }
    scalar_bar_actor.SetTitle(scalar_titles.get(variable, variable))
    scalar_bar_actor.SetPosition(0.88, 0.08)
    scalar_bar_actor.SetWidth(0.032)
    scalar_bar_actor.SetHeight(0.34)
    scalar_bar_actor.SetNumberOfLabels(5)
    scalar_bar_actor.SetUnconstrainedFontSize(True)
    scalar_bar_actor.GetTitleTextProperty().SetBold(False)
    scalar_bar_actor.GetLabelTextProperty().SetBold(False)
    if variable == 'density':
        scalar_bar_actor.SetLabelFormat("%.2e")

    theta = np.linspace(0, 2 * np.pi, 720)
    r = np.array([0, 21.5, 115, 230])
    theta_grid, r_grid = np.meshgrid(theta, r)
    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)
    z_grid = np.zeros_like(x_grid)

    for i in range(r_grid.shape[0]):
        points = np.c_[x_grid[i, :], y_grid[i, :], z_grid[i, :]]
        line = pv.lines_from_points(points)
        if i == 1:  # Index 1 pour la deuxième ligne
            # Ajouter la deuxième ligne avec une couleur différente et essayer un effet visuel pour pointillé
            plotter.add_mesh(line, color="#4b4b4b", line_width=1.8, opacity=0.72)
        else:
            plotter.add_mesh(line, color=grid_color, line_width=1.5, opacity=0.48)

    theta = np.linspace(0, 2 * np.pi, 360)
    r = np.array([0, 21.5, 115, 234])
    theta_grid, r_grid = np.meshgrid(theta, r)
    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)
    z_grid = np.zeros_like(x_grid)

    # Ajout des lignes angulaires
    angle_labels = ['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°']
    extended_radius = 231 + 20
    for i in range(0, theta_grid.shape[1], 45):
        points = np.c_[x_grid[:, i], y_grid[:, i], z_grid[:, i]]
        line = pv.lines_from_points(points)
        plotter.add_mesh(line, color=outer_grid_color, line_width=1.5, opacity=0.5)

    correctif_r = {0: -10, 45: -10, 90: -10, 135: 0, 180: 10, 225: 10, 270: 0, 315: -1}
    correctif_angle = {0: 0, 45: 0, 90: 2, 135: 3, 180: 1, 225: 0, 270: -2, 315: -1}
    for i, value in enumerate([0, 45, 90, 135, 180, 225, 270, 315]):
        new_x = (extended_radius + correctif_r[value]) * np.cos(np.radians(value + correctif_angle[value]))
        new_y = (extended_radius + correctif_r[value]) * np.sin(np.radians(value + correctif_angle[value]))

        text_position = np.array([[new_x, new_y, 0]])

        plotter.add_point_labels(text_position, [angle_labels[i]], font_size=24, point_size=1, text_color=label_color,
                                 show_points=False, shape=None)

    plane = pv.Plane(center=inset_center, direction=(0, 0, 1), i_size=inset_size, j_size=inset_size)
    texture = pv.read_texture(output_image_path)
    plotter.add_mesh(plane, texture=texture)
    plotter.add_text("COCONUT domain", position=(0.11, 0.385), font_size=18, color="#222222", viewport=True)

    plotter.camera.up = (0.0, 1.0, 0.0)
    plotter.camera.position = (-38.18563118811632, 0.8302754252406501, 970.1723378487242)
    plotter.camera.focal_point = (-38.18563118811632, 0.8302754252406501, 0.0)

    def update_text():
        cam = plotter.camera
        info = f"Position: {cam.position}\nFocal Point: {cam.focal_point}\nUp Vector: {cam.up}"
        print(info)

    def on_mouse_click(*args):
        update_text()
        plotter.render()

    # plotter.track_click_position(on_mouse_click, side="left")
    if legend_entries:
        plotter.add_legend(
            legend_entries,
            face='circle',
            bcolor=(1.0, 1.0, 1.0),
            border=True,
            loc="upper left",
            size=(0.22, 0.23),
        )
    if save:
        plotter.show(auto_close=False)
        plotter.screenshot(f'{outputdir}pyvista_{variable}_{idx:04d}.bmp')
        plotter.screenshot(f'{outputdir}pyvista_{variable}_{idx:04d}.png')
        # plotter.save_graphic(f'{outputdir}pyvista_{idx}.eps')
        plotter.close()
    else:
        plotter.show()


def find_nearest_date_index(date_list_e, target_date):
    pos = bisect_left(date_list_e, target_date)
    if pos == 0:
        return 0
    if pos == len(date_list_e):
        return len(date_list_e) - 1
    before = date_list_e[pos - 1]
    after = date_list_e[pos]

    if after - target_date < target_date - before:
        return pos
    else:
        return pos - 1


def movie(outputdir, name):
    image_files = sorted(glob(os.path.join(outputdir, f'pyvista_{name}_*.bmp')))

    # Vérifier si des fichiers ont été trouvés
    if not image_files:
        raise ValueError("Aucun fichier d'image trouvé dans le répertoire spécifié.")

    # Lire la première image pour obtenir les dimensions
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape

    # Définir le nom et le codec de la vidéo de sortie
    output_video = f'{outputdir}{name}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 5

    # Initialiser l'objet VideoWriter
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Ajouter chaque image à la vidéo
    for image_file in image_files:
        frame = cv2.imread(image_file)
        video.write(frame)

    # Libérer l'objet VideoWriter
    video.release()


if __name__ == "__main__":


    w_S = 2.66622373e-6
    #w_S = 0
    start_date = pd.to_datetime("2019-07-02T12:04:37")
    #start_date = pd.to_datetime("2024-04-09T05:04:00")
    interval_hours = 0.402 * 20 * 0.005
    #interval_hours = 0.2488 * 0.402
    date_list = [start_date + pd.to_timedelta(interval_hours * i, unit='h') for i in range(600)]

    dsvdirectory = "E:/coupling_movie/dsv/"

    folder_path = "E:/coupling_movie/vts/"
    sorted_datetimes, sorted_file = get_sorted_datetimes(folder_path)

    seconds_since_start = [(date - start_date).total_seconds() for date in sorted_datetimes]

    outputdir = "E:/coupling_movie/image/"
    
    """
    idx = 0
    file = sorted_file[idx]
    target_date = sorted_datetimes[idx]
    nearest_index = find_nearest_date_index(date_list, target_date)
    filevtu = f'E:/coupling_movie/vtu_filter/data_{nearest_index}.vtu'
    filevts = f'E:/coupling_movie/vts/{file}'
    print(f'{idx} {filevtu} {filevts}')
    print(f"Target date: {target_date} nearest date {date_list[nearest_index]}")
    time_delta = target_date - date_list[nearest_index]
    ecart = time_delta.total_seconds() / 60
    print(f'ecart {ecart:.1f} minutes since start {seconds_since_start[idx]:.1f}')

    slice2D(filevtu, filevts, dsvdirectory, target_date, outputdir, idx, ecart,
            angle=0 + np.degrees(w_S * seconds_since_start[idx]), save=False, variable='Br')


    slice2D(filevtu, filevts, dsvdirectory, target_date, outputdir, idx, ecart,
            angle=0 + np.degrees(w_S * seconds_since_start[idx]), save=False, variable='Bclt')


    slice2D(filevtu, filevts, dsvdirectory, target_date, outputdir, idx, ecart,
            angle=0 + np.degrees(w_S * seconds_since_start[idx]), save=False, variable='Blon')

    slice2D(filevtu, filevts, dsvdirectory, target_date, outputdir, idx, ecart,
            angle=0 + np.degrees(w_S * seconds_since_start[idx]), save=False, variable='Vr')

    slice2D(filevtu, filevts, dsvdirectory, target_date, outputdir, idx, ecart,
            angle=180 + np.degrees(w_S * seconds_since_start[idx]), save=False, variable='T')
    slice2D(filevtu, filevts, dsvdirectory, target_date, outputdir, idx, ecart,
            angle=180 + np.degrees(w_S * seconds_since_start[idx]), save=False, variable='density')
    slice2D(filevtu, filevts, dsvdirectory, target_date, outputdir, idx, ecart,
            angle=180 + np.degrees(w_S * seconds_since_start[idx]), save=False, variable='Vr')
    

    for idx, file in enumerate(sorted_file[:]):
        target_date = sorted_datetimes[idx]
        nearest_index = find_nearest_date_index(date_list, target_date)
        filevtu = f'E:/coupling_movie/vtu_filter/data_{nearest_index}.vtu'
        filevts = f'E:/coupling_movie/vts/{file}'
        print(f'{idx} {filevtu} {filevts}')
        print(f"Target date: {target_date} neares date {date_list[nearest_index]}")
        time_delta = target_date - date_list[nearest_index]
        ecart = time_delta.total_seconds() / 60
        print(f'ecart {ecart:.1f}')
        slice2D(filevtu, filevts, dsvdirectory, target_date, outputdir, idx, ecart, angle=180+np.degrees(w_S*seconds_since_start[idx]), save=True, variable='density')
        slice2D(filevtu, filevts, dsvdirectory, target_date, outputdir, idx, ecart, angle=180+np.degrees(w_S*seconds_since_start[idx]), save=True, variable='Vr')
        slice2D(filevtu, filevts, dsvdirectory, target_date, outputdir, idx, ecart, angle=180+np.degrees(w_S*seconds_since_start[idx]), save=True, variable='T')
        slice2D(filevtu, filevts, dsvdirectory, target_date, outputdir, idx, ecart, angle=180+np.degrees(w_S*seconds_since_start[idx]), save=True, variable='Bclt')

    """    

    outputdir = f'E:/coupling_movie/image/'
    name='density'
    movie(outputdir, name)

    outputdir = f'E:/coupling_movie/image/'
    name='Vr'
    movie(outputdir, name)

    outputdir = f'E:/coupling_movie/image/'
    name='T'
    movie(outputdir, name)

    outputdir = f'E:/coupling_movie/image/'
    name='Bclt'
    movie(outputdir, name)
    
