import numpy as np
from scipy.special import spherical_jn
from numpy import sin, cos

class LFFSpheromakParameters:
    def __init__(self):
        self.helicity_sign = 1.0  # Par défaut
        self.radius = 1.0
        self.alpha = 1.0
        self.x_center = 0.0
        self.y_center = 0.0
        self.z_center = 0.0
        self.lat = 0.0
        self.lon = 0.0
        self.tilt_angle = 0.0
        self.B0 = 1.0


class LFFSpheromakMagneticField:
    def __init__(self, param):
        self.param = param

    def magnetic_field(self, p):
        r, th = p[0], p[1]
        B = np.array([self.dAdth(p) / r, -self.dAdr(p), self.Q(p)]) * (1.0 / (r * sin(th)))
        return B

    def poloidal_potential(self, p):
        return self.A(p)

    def toroidal_potential(self, p):
        return self.Q(p)

    def A(self, p):
        r, th = p[0], p[1]
        return (self.param.B0 / self.param.alpha) * r * spherical_jn(1, self.param.alpha * r) * sin(th) * sin(th)

    def Q(self, p):
        return self.Q_scalar(self.A(p))

    def Q_scalar(self, A):
        return self.param.helicity_sign * self.param.alpha * A

    def dAdr(self, p):
        r, th = p[0], p[1]
        x = self.param.alpha * r
        return (self.param.B0 / self.param.alpha) * (spherical_jn(1, x) + x * self.dj1dx(x)) * sin(th) * sin(th)

    def dAdth(self, p):
        r, th = p[0], p[1]
        return (self.param.B0 / self.param.alpha) * r * spherical_jn(1, self.param.alpha * r) * 2.0 * sin(th) * cos(th)

    def j1(self, x):
        return (-x * cos(x) + sin(x)) / (x * x)

    def dj1dx(self, x):
        j0 = sin(x) / x
        return j0 - 2.0 * spherical_jn(1, x) / x


class LFFSpheromak:
    def __init__(self):
        self.parameters = LFFSpheromakParameters()
        self.internal = LFFSpheromakMagneticField(self.parameters)

    def initialize(self):
        self.internal.param = self.parameters

    def is_inside(self, p):
        p_local_internal_sph = self.transform_to_local_spherical(p)
        return p_local_internal_sph[0] <= self.parameters.radius

    def B(self, p):
        # Convertir la position en coordonnées sphériques locales
        p_local_internal_sph = self.transform_to_local_spherical(p)

        # Vérifier si le point est à l'intérieur du rayon
        if p_local_internal_sph[0] <= self.parameters.radius:
            # Champ magnétique interne en coordonnées sphériques locales
            bint_sph = self.internal.magnetic_field(p_local_internal_sph)
            # Convertir en coordonnées globales sphériques
            return self.transform_to_global_spherical(bint_sph, p_local_internal_sph, p)

        return np.zeros(3)

    def magnetic_field(self, p):
        return self.B(p)

    def transform_to_local_spherical(self, p):
        # Conversion de sphérique à cartésien
        p_cartesian = self.spherical_to_cartesian(p)

        # Appliquer la translation puis la rotation
        p_local_cartesian = self.apply_transformations(p_cartesian)

        # Retour à des coordonnées sphériques
        return self.cartesian_to_spherical(p_local_cartesian)

    def spherical_to_cartesian(self, p):
        r, theta, phi = p
        x = r * sin(theta) * cos(phi)
        y = r * sin(theta) * sin(phi)
        z = r * cos(theta)
        return np.array([x, y, z])

    def cartesian_to_spherical(self, p):
        x, y, z = p
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return np.array([r, theta, phi])

    def apply_transformations(self, p):
        # Appliquer la translation
        translation = np.array([-self.parameters.x_center, -self.parameters.y_center, -self.parameters.z_center])
        p_translated = p + translation

        # Appliquer les rotations
        rotation_matrix = self.get_rotation_matrix()
        return np.dot(rotation_matrix, p_translated)

    def get_rotation_matrix(self):
        # Création des matrices de rotation pour les angles
        tilt = self.parameters.tilt_angle
        lat = self.parameters.lat
        lon = self.parameters.lon

        rotate_x = np.array([[1, 0, 0],
                             [0, cos(tilt), -sin(tilt)],
                             [0, sin(tilt), cos(tilt)]])

        rotate_y = np.array([[cos(lat), 0, sin(lat)],
                             [0, 1, 0],
                             [-sin(lat), 0, cos(lat)]])

        rotate_z = np.array([[cos(lon), -sin(lon), 0],
                             [sin(lon), cos(lon), 0],
                             [0, 0, 1]])

        # Appliquer dans l'ordre z, y, x
        return rotate_x@rotate_y@rotate_z

    def spherical_to_cartesian_vector(self, v_sph, p_sph):
        r, t, p = p_sph
        Br, Bt, Bp = v_sph

        Bx = Br * np.sin(t) * np.cos(p) + Bt * np.cos(t) * np.cos(p) - Bp * np.sin(p)
        By = Br * np.sin(t) * np.sin(p) + Bt * np.cos(t) * np.sin(p) + Bp * np.cos(p)
        Bz = Br * np.cos(t) - Bt * np.sin(t)

        return np.array((Bx, By, Bz))
    def cartesian_to_spherical_vector(self, v_cart, p_sph):
        r, t, p = p_sph
        Bx, By, Bz = v_cart

        Br = Bx * np.sin(t) * np.cos(p) + By * np.sin(t) * np.sin(p) + Bz * np.cos(t)
        Bt = Bx * np.cos(t) * np.cos(p) + By * np.cos(t) * np.sin(p) - Bz * np.sin(t)
        Bp = -Bx * np.sin(p) + By * np.cos(p)

        return np.array((Br, Bt, Bp))

    def transform_to_global_spherical(self, bint_sph, p_sph,p):
        # Transformer le vecteur du champ magnétique sphérique en cartésien (en tant que vecteur, pas point)
        bint_cart = self.spherical_to_cartesian_vector(bint_sph, p_sph)

        # Appliquer l'inverse des rotations pour revenir au système global
        b_global_cart = np.dot(self.get_rotation_matrix().T, bint_cart)

        # Convertir en sphérique dans le système global
        return self.cartesian_to_spherical_vector(b_global_cart, p)
