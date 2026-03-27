
"""Spheromak Coronal Mass Ejection Models
"""
import os
import datetime
import numpy as np
import scipy.special
import coco.core.constants as constants
from PyQt5.QtCore import QObject, pyqtSignal
from spheromak_base import LFFSpheromak
import matplotlib.pyplot as plt
from functions import read, writeouputfile, Surface_2D_onetime
import glob



class LFFSpheromakModel(QObject):
    """Linear Force-Free spheromak CME model.

    This spherical CME model consists of a linear force-free spheromak
    magnetic field configuration and a uniform temperature and density.
    As such, it is a magnetized generalization of the uniform sphere
    CME model.

    Args:
        start_time    :  Time at which the CME intersects the r=0.1 AU sphere
        lat           :  Latitude of the eruption, HEEQ deg
        lon           :  Longitude of the eruption, HEEQ deg
        radius        :  Radius of CME, RSun
        speed         :  Speed of the CME, km/s
        mass_density  :  Uniform CME mass density, kg/m^3
        temperature   :  Uniform CME temperature, K
        helicity_sign :  Sign of the helicity of the flux rope, +/- 1
        tilt          :  Tilt angle of CME, deg
        toroidal_flux :  Total toroidal magnetic flux, Wb
        solar_rotation:  Bool
    """
    progress_signal = pyqtSignal(int)

    def __init__(self,
                 start_time,
                 lat,
                 lon,
                 radius,
                 speed,
                 mass_density,
                 temperature,
                 helicity_sign,
                 tilt,
                 toroidal_flux,
                 solar_rotation):

        #
        # Store parameters
        #
        super().__init__()
        class Parameters(object): pass
        self.params = Parameters()

        self.params.interface_radius = 0.1*constants.astronomical_unit
        self.params.interface_radius_in_Rs = self.params.interface_radius/constants.solar_radius

        self.params.lat = float(lat)*np.pi/180.0
        self.params.clt = -self.params.lat + 0.5*np.pi
        self.params.lon = float(lon)*np.pi/180.0

        self.params.start_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S" )

        self.params.radius  = float(radius)*constants.solar_radius
        self.params.speed   = float(speed)*1e3
        self.params.mass_density  = float(mass_density)
        self.params.temperature   = float(temperature)
        self.params.solar_rotation = bool(solar_rotation)
        self.params.swr= 2.66622373e-6


        #
        # Set spheromak parameters
        #
        self.model = LFFSpheromak()

        self.model.parameters.helicity_sign = float(helicity_sign)
        self.model.parameters.tilt_angle = float(tilt)*np.pi/180.0
        self.model.parameters.radius = float(radius)*constants.solar_radius

        self.model.parameters.lat = self.params.lat
        self.model.parameters.lon = self.params.lon

        #
        # Force-free parameter alpha is determined so that
        #   Br(r=R) = 0 => j_1(alpha*R) = 0 => alpha*R = first zero of j_1(x)
        self.model.parameters.alpha = 4.49340945/self.model.parameters.radius

        #
        # B0 is set via the specified total toroidal flux
        #

        # Shorthand
        xi0 = self.model.parameters.alpha*self.model.parameters.radius

        # Total toroidal flux / B0
        flux_per_B0 = (2.0*self.model.parameters.helicity_sign/self.model.parameters.alpha**2)*(-np.sin(xi0) + scipy.special.sici(xi0)[0])

        self.model.parameters.B0 = float(toroidal_flux)/flux_per_B0


        self.model.initialize()

        #
        # Compute additional derived parameters
        #

        # Time at which CME has fully passed the boundary
        self.params.end_time = self.params.start_time + datetime.timedelta(seconds=2.0*self.params.radius/self.params.speed)

        # Cartesian velocity vector of CME
        self.velocity \
            = self.params.speed*np.array((np.sin(self.params.clt)*np.cos(self.params.lon),
                                          np.sin(self.params.clt)*np.sin(self.params.lon),
                                          np.cos(self.params.clt)))

        
        # Distance of center of sphere from r=0.1 AU at t = start_time
        distance_from_boundary = 0.1*constants.astronomical_unit - self.params.radius

        # Cartesian position of center of CME sphere at t = start_time
        self.pos_of_center_at_start_time \
            = distance_from_boundary*np.array((np.sin(self.params.clt)*np.cos(self.params.lon),
                                               np.sin(self.params.clt)*np.sin(self.params.lon),
                                               np.cos(self.params.clt)))


    def test_b(self, interface_radius, clts, lons, x_center, y_center, z_center ):
        print(f'The parameter are {self.params.lat} {self.params.lon} {self.params.radius} {self.params.speed} {self.model.parameters.tilt_angle}')
        self.model.parameters.x_center = x_center
        self.model.parameters.y_center = y_center
        self.model.parameters.z_center = z_center
        Br, Bt, Bp = self.model.magnetic_field((interface_radius,clts,lons))
        print(f'we get {self.model.parameters.x_center} {self.model.parameters.y_center} {self.model.parameters.z_center} {interface_radius} {clts} {lons} {Br} {Bt} {Bp}')

    def test(self):

        end_time = self.params.start_time+datetime.timedelta(hours=3)
        j=0
        current_time = self.params.start_time
        while current_time <= end_time:
            current_time = self.params.start_time+datetime.timedelta(minutes=j*5)
            j=j+1
            self.update_center_position(current_time)
            x = np.linspace(0, 21.5, 200)*695700.0e3 # 100 points from 0 to 21.5
            y = np.linspace(-21.5, 21.5, 200)*695700.0e3  # 100 points from 0 to 21.5
            z = np.linspace(-21.5, 21.5, 200) * 695700.0e3

            # Create the meshgrid
            Y, Z = np.meshgrid(y, z)
            X = np.zeros(Y.shape) + 10.5*695700.0e3
            Br_temp, Bt_temp, Bp_temp = np.zeros(X.shape)+np.nan, np.zeros(X.shape)+np.nan, np.zeros(X.shape)+np.nan
            Br_r, Bt_r, Bp_r = Br_temp.ravel(), Bt_temp.ravel(), Bp_temp.ravel()

            r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
            theta = np.arccos(Z / r)
            phi = np.arctan2(Y, X)
            print(j)
            for i in range(len(r.ravel())):
                r_i=r.ravel()[i]
                theta_i=theta.ravel()[i]
                phi_i=phi.ravel()[i]
                mask=self.model.is_inside((r_i,theta_i,phi_i))
                if mask:
                    Br_r[i], Bt_r[i], Bp_r[i]=self.model.magnetic_field((r_i, theta_i, phi_i))

            fig = plt.figure(figsize=(15,10))
            ax1 = fig.add_subplot(1, 3, 1)  # Première ligne, première colonne
            im1 = ax1.imshow(Br_r.reshape(X.shape), extent=[Y.min(), Y.max(), Z.min(), Z.max()])
            plt.colorbar(im1, ax=plt.gca(), fraction=0.05)  # Ajoute la barre de couleur pour le second subplot
            ax1.set_title('Br')

            ax2 = fig.add_subplot(1, 3, 2)  # Première ligne, première colonne
            im1 = ax2.imshow(Bt_r.reshape(X.shape), extent=[Y.min(), Y.max(), Z.min(), Z.max()])
            plt.colorbar(im1, ax=plt.gca(), fraction=0.05)  # Ajoute la barre de couleur pour le second subplot
            ax2.set_title('Bclt')

            ax3 = fig.add_subplot(1, 3, 3)  # Première ligne, première colonne
            im1 = ax3.imshow(Bp_r.reshape(X.shape), extent=[Y.min(), Y.max(), Z.min(), Z.max()])
            plt.colorbar(im1, ax=plt.gca(), fraction=0.05)  # Ajoute la barre de couleur pour le second subplot
            ax3.set_title('Blon')
            plt.savefig(f'C:/Users/luisl/Documents/Travail/GUI_V2/spheromak/magnetic_field_{j}')
            plt.close()

        return

    def center(self, t):
        """Computes the Cartesian HEEQ coordinates of the position of the
        center of the sphere at the given time

        Args:
            t (datetime) : Current date and time
        Returns:
            numpy array containing (x,y,z) coordinates
        """

        time_from_start = (t - self.params.start_time).total_seconds()

        return self.pos_of_center_at_start_time + self.velocity*time_from_start


    def update_center_position(self, t):
        """Updates CME center position in the magnetic field instance
        """

        center = self.center(t)

        self.model.parameters.x_center = center[0]
        self.model.parameters.y_center = center[1]
        self.model.parameters.z_center = center[2]

    def createboundary(self, solar_wind_file, time_step,output_dir,savefig):


        base_dir = os.path.dirname(solar_wind_file)
        pattern = os.path.join(base_dir, 'solar_wind_boundary_*.dat')
        files = sorted(glob.glob(pattern), key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))

        if len(files) == 1:
            time, radius, colatitude_points, longitude_points, clt, lon, vr, vp, vt, n, t, br, bp, bt = read(solar_wind_file)
            time_obj_s = datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")
            total_iterations = int((self.params.end_time - time_obj_s).total_seconds() // (time_step * 60))

            for i in range(total_iterations):
                print(i)
                time_obj = time_obj_s + datetime.timedelta(minutes=i*time_step)
                progress = int((i / total_iterations) * 100)
                self.progress_signal.emit(progress)

                if time_obj>=self.params.start_time:

                    lonmesh, cltmesh =np.meshgrid(lon,clt,indexing='ij')

                    clt_r=cltmesh.ravel()
                    lon_r=lonmesh.ravel()

                    self.update_center_position(time_obj)

                    center = self.center(time_obj)
                    r = int(radius)

                    dx = r * np.sin(clt_r) * np.cos(lon_r) - center[0]
                    dy = r * np.sin(clt_r) * np.sin(lon_r) - center[1]
                    dz = r * np.cos(clt_r) - center[2]

                    dsqr = dx ** 2 + dy ** 2 + dz ** 2

                    mask = dsqr <= (self.params.radius ** 2)

                    self._idx  = np.where(mask)[0]

                    if len(self._idx) != 0:

                        clts=clt_r[self._idx]
                        lons=lon_r[self._idx]

                        vr_temp = self.velocity[0] * np.sin(clts) * np.cos(lons) \
                             + self.velocity[1] * np.sin(clts) * np.sin(lons) \
                             + self.velocity[2] * np.cos(clts)

                        vp_temp = - self.velocity[0] * np.sin(lons) \
                             + self.velocity[1] * np.cos(lons)

                        vt_temp = self.velocity[0] * np.cos(clts) * np.cos(lons) \
                             + self.velocity[1] * np.cos(clts) * np.sin(lons) \
                             - self.velocity[2] * np.sin(clts)


                        vr_n, vp_n, vt_n, n_n, t_n, br_n, bp_n, bt_n=np.copy(vr), np.copy(vp), np.copy(vt), np.copy(n), np.copy(t), np.copy(br), np.copy(bp), np.copy(bt)
                        vr_n[self._idx]=vr_temp
                        vp_n[self._idx]=vp_temp
                        vt_n[self._idx]=vt_temp
                        n_n[self._idx]=self.params.mass_density
                        t_n[self._idx]=self.params.temperature

                        for id in self._idx:
                            Br_temp, Bt_temp, Bp_temp = self.model.magnetic_field((r,clt_r[id],lon_r[id]))
                            br_n[id]=Br_temp
                            bp_n[id]=Bp_temp
                            bt_n[id]=Bt_temp

                        writeouputfile(output_dir, i, time_obj, radius,
                                            colatitude_points, longitude_points, clt, lon, vr_n, vp_n, vt_n, n_n, t_n, br_n, bp_n, bt_n)
                    else:
                        writeouputfile(output_dir, i, time_obj, radius,
                                            colatitude_points, longitude_points, clt, lon, vr, vp, vt, n, t, br, bp, bt)
                else:
                    writeouputfile(output_dir,i,time_obj,radius,colatitude_points,longitude_points, clt, lon, vr, vp, vt, n, t, br, bp, bt)
                if savefig:
                    path_components = output_dir.split('/')
                    path_components = [component for component in path_components if component != 'dat']
                    image_dir = os.path.join(*path_components, 'image')
                    if not os.path.exists(image_dir):
                        os.makedirs(image_dir)
                    inputfile = f'{output_dir}/solar_wind_boundary_{str(int(time_obj.timestamp()))}.dat'
                    outputfile = f'{image_dir}/surface_{i:04}.png'
                    Surface_2D_onetime(inputfile, outputfile,time_obj)
        else:
            for i, file in enumerate(files):
                time, radius, colatitude_points, longitude_points, clt, lon, vr, vp, vt, n, t, br, bp, bt = read(file)
                time_obj = datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")

                if time_obj < self.params.start_time or time_obj > self.params.end_time:
                    continue  # Ne rien faire si c'est en dehors de l'intervalle [start_time, end_time]

                print(i, time_obj)
                self.progress_signal.emit(int((i / len(files)) * 100))

                lonmesh, cltmesh = np.meshgrid(lon, clt, indexing='ij')
                clt_r = cltmesh.ravel()
                lon_r = lonmesh.ravel()

                self.update_center_position(time_obj)
                center = self.center(time_obj)
                r = int(radius)

                dx = r * np.sin(clt_r) * np.cos(lon_r) - center[0]
                dy = r * np.sin(clt_r) * np.sin(lon_r) - center[1]
                dz = r * np.cos(clt_r) - center[2]

                dsqr = dx ** 2 + dy ** 2 + dz ** 2
                mask = dsqr <= (self.params.radius ** 2)

                self._idx = np.where(mask)[0]

                if len(self._idx) != 0:
                    clts = clt_r[self._idx]
                    lons = lon_r[self._idx]

                    vr_temp = self.velocity[0] * np.sin(clts) * np.cos(lons) \
                              + self.velocity[1] * np.sin(clts) * np.sin(lons) \
                              + self.velocity[2] * np.cos(clts)

                    vp_temp = - self.velocity[0] * np.sin(lons) \
                              + self.velocity[1] * np.cos(lons)

                    vt_temp = self.velocity[0] * np.cos(clts) * np.cos(lons) \
                              + self.velocity[1] * np.cos(clts) * np.sin(lons) \
                              - self.velocity[2] * np.sin(clts)

                    vr_n, vp_n, vt_n, n_n, t_n, br_n, bp_n, bt_n = np.copy(vr), np.copy(vp), np.copy(vt), np.copy(
                        n), np.copy(t), np.copy(br), np.copy(bp), np.copy(bt)
                    vr_n[self._idx] = vr_temp
                    vp_n[self._idx] = vp_temp
                    vt_n[self._idx] = vt_temp
                    n_n[self._idx] = self.params.mass_density
                    t_n[self._idx] = self.params.temperature

                    for idx in self._idx:
                        Br_temp, Bt_temp, Bp_temp = self.model.magnetic_field((r, clt_r[idx], lon_r[idx]))
                        br_n[idx] = Br_temp
                        bp_n[idx] = Bp_temp
                        bt_n[idx] = Bt_temp

                    writeouputfile(output_dir, i, time_obj, radius,
                                   colatitude_points, longitude_points, clt, lon, vr_n, vp_n, vt_n, n_n, t_n, br_n,
                                   bp_n, bt_n)

                    if savefig:
                        path_components = output_dir.split('/')
                        path_components = [component for component in path_components if component != 'dat']
                        image_dir = os.path.join(*path_components, 'image')
                        if not os.path.exists(image_dir):
                            os.makedirs(image_dir)
                        inputfile = file
                        outputfile = f'{image_dir}/surface_{i:04}.png'
                        Surface_2D_onetime(inputfile, outputfile, time_obj)
        return

    def mask(self, clt, lon, t):
        """Computes a mask of the grid points on the boundary at r = 0.1 AU
        where the CME is inserted.
        """

        # Construct grid of coordinates
        cltmesh, lonmesh = np.meshgrid(clt, lon, indexing="ij")

        # Update center position
        self.update_center_position(t)

        # Get Cartesian coordinates of center of sphere
        center = self.center(t)

        #
        # Compute squared distance to center of sphere
        #
        r  = 0.1*constants.astronomical_unit

        dx = r*np.sin(cltmesh)*np.cos(lonmesh)-center[0]
        dy = r*np.sin(cltmesh)*np.sin(lonmesh)-center[1]
        dz = r*np.cos(cltmesh)-center[2]

        dsqr = dx**2 + dy**2 + dz**2

        # If distance < radius => inside sphere
        mask = dsqr <= (self.params.radius**2)

        return mask


    def indices_to_insert(self, clt, lon, t):
        """Indices of grid points on the boundary at r = 0.1 AU where the
        CME is inserted.
        """

        idx = ()

        if t >= self.params.start_time and t <= self.params.end_time:

            mask  = self.mask(clt, lon, t)
            shape = (len(clt), len(lon))
            idx   = np.ravel_multi_index(np.where(mask > 0.0), shape)

        return idx


    def get_coordinates_on_sphere(self, clt, lon, idx):
        """Get (colat, lon) coordinates on the given flattened indices
        """

        cltmesh, lonmesh = np.meshgrid(clt, lon, indexing="ij")

        clts = cltmesh.ravel()[idx] # Use ravel instead?
        lons = lonmesh.ravel()[idx]

        return clts, lons


    def mass_density(self, clt, lon, t):
        """Computes CME mass density
        """
        return self.indices_to_insert(clt, lon, t), self.params.mass_density


    def temperature(self, clt, lon, t):
        """Computes CME temperature
        """
        return self.indices_to_insert(clt, lon, t), self.params.temperature


    def vr(self, clt, lon, t):
        """Computes CME radial speed
        """

        vr  = 0.0
        idx = self.indices_to_insert(clt, lon, t)

        print(np.shape(idx))

        # Compute on the indices where the CME is to be inserted
        if len(idx) > 0:

            clts, lons = self.get_coordinates_on_sphere(clt, lon, idx)

            vr =   self.velocity[0]*np.sin(clts)*np.cos(lons) \
                 + self.velocity[1]*np.sin(clts)*np.sin(lons) \
                 + self.velocity[2]*np.cos(clts)
        return idx, vr


    def vt(self, clt, lon, t):
        """Computes CME colatitudinal speed
        """

        vt  = 0.0
        idx = self.indices_to_insert(clt, lon, t)

        # Compute on the indices where the CME is to be inserted
        if len(idx) > 0:

            clts, lons = self.get_coordinates_on_sphere(clt, lon, idx)

            vt =   self.velocity[0]*np.cos(clts)*np.cos(lons) \
                 + self.velocity[1]*np.cos(clts)*np.sin(lons) \
                 - self.velocity[2]*np.sin(clts)

        return idx, vt


    def vp(self, clt, lon, t):
        """Computes CME longitudinal speed
        """

        vp  = 0.0
        idx = self.indices_to_insert(clt, lon, t)

        # Compute on the indices where the CME is to be inserted
        if len(idx) > 0:

            clts, lons = self.get_coordinates_on_sphere(clt, lon, idx)

            vp = - self.velocity[0]*np.sin(lons) \
                 + self.velocity[1]*np.cos(lons)

        return idx, vp


    def magnetic_field(self, p, t):

        self.update_center_position(t)

        return self.model.magnetic_field(p)


    def Br(self, clt, lon, t):
        """Computes CME radial magnetic field component
        """

        Br  = 0.0
        idx = self.indices_to_insert(clt, lon, t)

        # Compute on the indices where the CME is to be inserted
        if len(idx) > 0:

            clts, lons = self.get_coordinates_on_sphere(clt, lon, idx)

            # Compute Br at the given points
            Br   = np.zeros(len(idx))

            for i in range(len(idx)):
                Br[i], Bt, Bp \
                    = self.model.magnetic_field((self.params.interface_radius,
                                                 clts[i],
                                                 lons[i]))

        return idx, Br


    def Bt(self, clt, lon, t):
        """Computes CME colatitudinal magnetic field component
        """

        Bt  = 0.0
        idx = self.indices_to_insert(clt, lon, t)

        # Compute on the indices where the CME is to be inserted
        if len(idx) > 0:

            clts, lons = self.get_coordinates_on_sphere(clt, lon, idx)

            # Compute Br at the given points
            Bt   = np.zeros(len(idx))

            for i in range(len(idx)):
                Br, Bt[i], Bp \
                    = self.model.magnetic_field((self.params.interface_radius,
                                                 clts[i],
                                                 lons[i]))

        return idx, Bt


    def Bp(self, clt, lon, t):
        """Computes CME longitudinal magnetic field component
        """

        Bp  = 0.0
        idx = self.indices_to_insert(clt, lon, t)

        # Compute on the indices where the CME is to be inserted
        if len(idx) > 0:

            clts, lons = self.get_coordinates_on_sphere(clt, lon, idx)

            # Compute Br at the given points
            Bp   = np.zeros(len(idx))

            for i in range(len(idx)):
                Br, Bt, Bp[i] \
                    = self.model.magnetic_field((self.params.interface_radius,
                                                 clts[i],
                                                 lons[i]))

        return idx, Bp


    def __str__(self):

        message = "LFF Spheromak CME at {}".format(self.params.start_time)

        return message
