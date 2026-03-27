"""Local linear force-free spheromak model without external GUI dependencies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import sici, spherical_jn


def _direction_from_lat_lon(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Return the Cartesian unit vector associated with latitude/longitude.

    Args:
        lat_deg: Latitude in degrees.
        lon_deg: Longitude in degrees.

    Returns:
        Unit vector in Cartesian coordinates.
    """
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    return np.array(
        [
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class LocalSpheromakParameters:
    """Parameters of the local linear force-free spheromak model.

    Args:
        radius_m: Spheromak radius in meters.
        lat_deg: Latitude of the structure in degrees.
        lon_deg: Longitude of the structure in degrees.
        tilt_deg: Tilt angle in degrees.
        helicity_sign: Helicity sign, typically `+1` or `-1`.
        toroidal_flux_wb: Total toroidal magnetic flux in Weber.
        center_m: Cartesian center in meters.
    """

    radius_m: float
    lat_deg: float
    lon_deg: float
    tilt_deg: float
    helicity_sign: float
    toroidal_flux_wb: float
    center_m: np.ndarray


class LocalLFFSpheromak:
    """Evaluate a linear force-free spheromak in Cartesian coordinates."""

    def __init__(self, parameters: LocalSpheromakParameters) -> None:
        """Initialize the spheromak model.

        Args:
            parameters: Physical and geometrical parameters of the model.
        """
        self.parameters = parameters
        self._alpha = 4.49340945 / parameters.radius_m
        xi0 = self._alpha * parameters.radius_m
        flux_per_b0 = (
            2.0
            * parameters.helicity_sign
            / self._alpha**2
            * (-np.sin(xi0) + sici(xi0)[0])
        )
        self._b0 = parameters.toroidal_flux_wb / flux_per_b0
        self._rotation = self._build_rotation_matrix(
            lat_deg=parameters.lat_deg,
            lon_deg=parameters.lon_deg,
            tilt_deg=parameters.tilt_deg,
        )

    @staticmethod
    def _build_rotation_matrix(lat_deg: float, lon_deg: float, tilt_deg: float) -> np.ndarray:
        """Build the rotation matrix used by the original spheromak script.

        Args:
            lat_deg: Latitude in degrees.
            lon_deg: Longitude in degrees.
            tilt_deg: Tilt angle in degrees.

        Returns:
            Rotation matrix of shape `(3, 3)`.
        """
        lat = np.deg2rad(lat_deg)
        lon = np.deg2rad(lon_deg)
        tilt = np.deg2rad(tilt_deg)

        rotate_x = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(tilt), -np.sin(tilt)],
                [0.0, np.sin(tilt), np.cos(tilt)],
            ]
        )
        rotate_y = np.array(
            [
                [np.cos(lat), 0.0, np.sin(lat)],
                [0.0, 1.0, 0.0],
                [-np.sin(lat), 0.0, np.cos(lat)],
            ]
        )
        rotate_z = np.array(
            [
                [np.cos(lon), -np.sin(lon), 0.0],
                [np.sin(lon), np.cos(lon), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return rotate_x @ rotate_y @ rotate_z

    def set_center(self, center_m: np.ndarray) -> None:
        """Update the center of the spheromak.

        Args:
            center_m: New center in meters.
        """
        self.parameters = LocalSpheromakParameters(
            radius_m=self.parameters.radius_m,
            lat_deg=self.parameters.lat_deg,
            lon_deg=self.parameters.lon_deg,
            tilt_deg=self.parameters.tilt_deg,
            helicity_sign=self.parameters.helicity_sign,
            toroidal_flux_wb=self.parameters.toroidal_flux_wb,
            center_m=np.asarray(center_m, dtype=np.float64),
        )

    def bulk_velocity_vector(self, speed_m_s: float) -> np.ndarray:
        """Return the Cartesian bulk propagation velocity.

        Args:
            speed_m_s: Speed magnitude in meters per second.

        Returns:
            Cartesian velocity vector in meters per second.
        """
        return speed_m_s * _direction_from_lat_lon(
            lat_deg=self.parameters.lat_deg,
            lon_deg=self.parameters.lon_deg,
        )

    def magnetic_field_cartesian(self, points_m: np.ndarray) -> np.ndarray:
        """Evaluate the magnetic field in global Cartesian coordinates.

        Args:
            points_m: Cartesian coordinates of shape `(N, 3)` in meters.

        Returns:
            Magnetic field of shape `(N, 3)` in Tesla.
        """
        points_m = np.asarray(points_m, dtype=np.float64)
        if points_m.ndim != 2 or points_m.shape[1] != 3:
            raise ValueError("points_m must have shape (N, 3).")

        translated = points_m - self.parameters.center_m
        local_points = translated @ self._rotation.T

        radius = np.linalg.norm(local_points, axis=1)
        inside = radius <= self.parameters.radius_m
        if not np.any(inside):
            return np.zeros_like(points_m)

        field_local = np.zeros_like(points_m)
        points_inside = local_points[inside]

        x_local = points_inside[:, 0]
        y_local = points_inside[:, 1]
        z_local = points_inside[:, 2]
        r_local = np.linalg.norm(points_inside, axis=1)
        r_local_safe = np.maximum(r_local, 1.0e-14)
        theta_local = np.arccos(np.clip(z_local / r_local_safe, -1.0, 1.0))
        phi_local = np.arctan2(y_local, x_local)

        sin_theta = np.sin(theta_local)
        sin_theta_safe = np.where(np.abs(sin_theta) < 1.0e-14, 1.0e-14, sin_theta)

        xi = self._alpha * r_local_safe
        xi_safe = np.where(np.abs(xi) < 1.0e-14, 1.0e-14, xi)
        j1 = spherical_jn(1, xi)
        j0 = np.sin(xi) / xi_safe
        dj1dx = j0 - 2.0 * j1 / xi_safe

        a_field = (
            (self._b0 / self._alpha)
            * r_local_safe
            * j1
            * sin_theta
            * sin_theta
        )
        dadr = (
            (self._b0 / self._alpha)
            * (j1 + xi * dj1dx)
            * sin_theta
            * sin_theta
        )
        dadtheta = (
            (self._b0 / self._alpha)
            * r_local_safe
            * j1
            * 2.0
            * sin_theta
            * np.cos(theta_local)
        )
        q_field = self.parameters.helicity_sign * self._alpha * a_field

        br = (dadtheta / r_local_safe) / (r_local_safe * sin_theta_safe)
        btheta = -dadr / (r_local_safe * sin_theta_safe)
        bphi = q_field / (r_local_safe * sin_theta_safe)

        bx_local = (
            br * np.sin(theta_local) * np.cos(phi_local)
            + btheta * np.cos(theta_local) * np.cos(phi_local)
            - bphi * np.sin(phi_local)
        )
        by_local = (
            br * np.sin(theta_local) * np.sin(phi_local)
            + btheta * np.cos(theta_local) * np.sin(phi_local)
            + bphi * np.cos(phi_local)
        )
        bz_local = br * np.cos(theta_local) - btheta * np.sin(theta_local)

        field_local[inside, 0] = bx_local
        field_local[inside, 1] = by_local
        field_local[inside, 2] = bz_local

        return field_local @ self._rotation
