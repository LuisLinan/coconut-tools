"""Module for interpolating gridded data.
"""

import numpy as np

import numba
from numba.experimental import jitclass

spec = [("edges",numba.u1),
        ("dr", numba.float64),
        ("dt", numba.float64),
        ("dp", numba.float64),
        ("add_to_lon", numba.float64),
        ("r", numba.float64[:]),
        ("t", numba.float64[:]),
        ("p", numba.float64[:]),
        ("data_Br", numba.float64[:, :, :]),
        ("data_Bt", numba.float64[:, :, :]),
        ("data_Bp", numba.float64[:, :, :])
        ]

@jitclass(spec)
class MagneticFieldInterpolator(object):
    """Class for interpolating staggered magnetic field data.

    The interpolator requires as input the coordinates of the grid as well as the
    gridded magnetic field data. The coordinates represent the edges of the
    grid cells.

    The magnetic field components are defined on staggered locations:

        Br[i, j, k] \equiv Br_{i-1/2, j, k} \equiv Br(r_{i-1/2}, y_j, z_k)
        Bt[i, j, k] \equiv Bt_{i, j-1/2, k}
        Bp[i, j, k] \equiv Bp_{i, j, k-1/2}

    where Br, Bt, Bp are the radial, co-latitudinal and longitudinal components,
    respectively.

    NOTE: Currently restricted to uniform grids!
    """

    def __init__(self):
        pass

    def set_grid_data(self, r, clt, lon, Br, Bclt, Blon):
        """Initializes using the given grid data
        """
        self.set_grid_coordinates(r, clt, lon)
        self.set_field_data(Br, Bclt, Blon)

    def set_grid_coordinates(self, r, clt, lon):
        """Set coordinates that define the grid

        Args:
            r  : edge coordinates r_{i-1/2} of the grid in the r-direction
        """

        # Grid edge coordinates
        self.r = np.copy(r)
        self.t = np.copy(clt)
        self.p = np.copy(lon)

        # Constant grid spacings
        self.dr = self.r[1] - self.r[0]
        self.dt = self.t[1] - self.t[0]
        self.dp = self.p[1] - self.p[0]

    def set_field_data(self, Br, Bt, Bp):
        if 0 in (len(self.r), len(self.t), len(self.p)):
            raise ValueError("Coordinates must be set before setting field data")

        self.data_Br = np.zeros((Br.shape[0], Br.shape[1], Br.shape[2]))
        self.data_Bt = np.zeros((Bt.shape[0], Bt.shape[1], Bt.shape[2]))
        self.data_Bp = np.zeros((Bp.shape[0], Bp.shape[1], Bp.shape[2]))

        self.data_Br[:, :, :] = Br[:, :, :]
        self.data_Bt[:, :, :] = Bt[:, :, :]
        self.data_Bp[:, :, :] = Bp[:, :, :]

    def get_cell_idx(self, r, t, p):
        """Find the cell index that the given point belongs to
        """

        # NOTE: no bounds checking
        i = int((r - self.r[0])/self.dr)
        j = int((t - self.t[0])/self.dt)
        k = int((p - self.p[0])/self.dp)

        return i, j, k%len(self.p)

    def Br(self, r, t, p):
        """Radial magnetic field component at the given coordinate
        """

        i, j, k = self.get_cell_idx(r, t, p)

        # Coordinates in local unit cube, (r_{i-1/2}, t_j, p_k)
        ru = (r-self.r[i])/self.dr
        tu = (t-self.t[j]-0.5*self.dt)/self.dt
        pu = (p-self.p[k]-0.5*self.dp)/self.dp

        # Trilinear interpolation
        return self.trilinear(self.data_Br, ru, tu, pu, i, j, k)

    def Bt(self, r, t, p):
        """Co-latitudinal magnetic field component at the given coordinate
        """

        i, j, k = self.get_cell_idx(r, t, p)

        # Coordinates in local unit cube, (r_{i}, t_{j-1/2}, p_k)
        ru = (r-self.r[i]-0.5*self.dr)/self.dr
        tu = (t-self.t[j])/self.dt
        pu = (p-self.p[k]-0.5*self.dp)/self.dp

        # Trilinear interpolation
        return self.trilinear(self.data_Bt, ru, tu, pu, i, j, k)

    def Bp(self, r, t, p):
        """Longitudinal magnetic field component at the given coordinate
        """

        i, j, k = self.get_cell_idx(r, t, p)

        # Coordinates in local unit cube, (r_{i}, t_j, p_{k-1/2})
        ru = (r-self.r[i]-0.5*self.dr)/self.dr
        tu = (t-self.t[j]-0.5*self.dt)/self.dt
        pu = (p-self.p[k])/self.dp

        # Trilinear interpolation
        return self.trilinear(self.data_Bp, ru, tu, pu, i, j, k)

    def trilinear(self, data, ru, tu, pu, i, j, k):
        """Trilinear interpolation
        """
        l = (k+1)%len(self.p)
        return \
              data[i  , j  , k  ]*(1.0-ru)*(1.0-tu)*(1.0-pu) \
            + data[i+1, j  , k  ]*(    ru)*(1.0-tu)*(1.0-pu) \
            + data[i  , j+1, k  ]*(1.0-ru)*(    tu)*(1.0-pu) \
            + data[i+1, j+1, k  ]*(    ru)*(    tu)*(1.0-pu) \
            + data[i  , j  , l  ]*(1.0-ru)*(1.0-tu)*(    pu) \
            + data[i+1, j  , l  ]*(    ru)*(1.0-tu)*(    pu) \
            + data[i  , j+1, l  ]*(1.0-ru)*(    tu)*(    pu) \
            + data[i+1, j+1, l  ]*(    ru)*(    tu)*(    pu)

    def mesh_spacing_at(self, r, t, p):
        """Returns the mesh spacing at the given coordinates
        """
        return np.array((self.dr, self.dt, self.dp))

    def displacement_vector(self, r, t, p):
        """Returns the coordinate displacement vector in the direction of the field.
        """

        # Field vector at the given point
        Br = self.Br(r, t, p)
        Bt = self.Bt(r, t, p)
        Bp = self.Bp(r, t, p)

        Babs = np.sqrt(Br*Br + Bt*Bt + Bp*Bp)

        #
        # Compute unit vectors in direction of field
        #
        br = Br/Babs
        bt = Bt/Babs
        bp = Bp/Babs

        # Displacement vector dl = (dr, r dclt, r sin(clt) dlon)
        return np.array((br, bt/r, bp/(r*np.sin(t))))

    def cell_length_at(self, r, t, p):
        """Returns the physical lengths of the cell at the given coordinates
        """

        i, j, k = self.get_cell_idx(r, t, p)

        R = 0.5*(self.r[i] + self.r[i+1])
        clt = 0.5*(self.t[j] + self.t[j+1])

        return np.array((self.dr, R*self.dt, R*np.sin(clt)*self.dp))

    def Br_at(self, coordinates):
        """Compute Br for a set of points
        """

        values = np.zeros(len(coordinates))

        for idx in range(len(coordinates)):
            values[idx] = self.Br(coordinates[idx][0], coordinates[idx][1], coordinates[idx][2])

        return values

    def Bt_at(self, coordinates):
        """Compute Bt for a set of points
        """

        values = np.zeros(len(coordinates))

        for idx in range(len(coordinates)):
            values[idx] = self.Bt(coordinates[idx][0], coordinates[idx][1], coordinates[idx][2])

        return values

    def Bp_at(self, coordinates):
        """Compute Bp for a set of points
        """

        values = np.zeros(len(coordinates))

        for idx in range(len(coordinates)):
            values[idx] = self.Bp(coordinates[idx][0], coordinates[idx][1], coordinates[idx][2])

        return values


spec = [("dr", numba.float64),
        ("dt", numba.float64),
        ("dp", numba.float64),
        ("add_to_lon", numba.float64),
        ("r", numba.float64[:]),
        ("t", numba.float64[:]),
        ("p", numba.float64[:]),
        ("data", numba.float64[:, :, :])
        ]

@jitclass(spec)
class CellCenteredScalarInterpolator(object):
    """Class for interpolating cell-centered scalar field data.

    The interpolator requires as input the coordinates of the grid as well as the
    gridded field data. The coordinates represent the edges of the
    grid cells.

    NOTE: Currently restricted to uniform grids!
    """

    def __init__(self):
        pass

    def set_grid_data(self, r, clt, lon, q):
        """Initializes using the given grid data
        """

        self.set_grid_coordinates(r, clt, lon)

        self.set_field_data(q)

    def set_grid_coordinates(self, r, clt, lon):
        """Set coordinates that define the grid

        Args:
            r  : center coordinates r_{i} of the grid in the r-direction
        """

        # Constant grid spacings
        self.dr = r[1]   - r[0]
        self.dt = clt[1] - clt[0]
        self.dp = lon[1] - lon[0]

        # Grid edge coordinates
        self.r = np.copy(r)   
        self.t = np.copy(clt) 
        self.p = np.copy(lon)


    def set_field_data(self, q):
        """Set field data.

        Creates a copy of the data of the given scalar.
        """
        if 0 in (len(self.r), len(self.t), len(self.p)):
            raise ValueError("Coordinates must be set before setting field data")

        if q.shape != (len(self.r), len(self.t), len(self.p)):
            raise ValueError("Shape of given field incompatible with grid coordinates")

        self.data = np.zeros(q.shape)

        self.data[:, :, :] = q[:, :, :]

    def get_left_idx(self, r, t, p):
        """Find the index i in each dimension for which xc_i <= x < xc_{i+1}
        """

        # NOTE: no bounds checking
        i = int((r - self.r[0])/self.dr)
        j = int((t - self.t[0])/self.dt)
        k = int((p - self.p[0])/self.dp)

        return i, j, k
    def in_domain(self,r,t,p):
        return (self.r[0]<=r<self.r[-1]) and (self.t[0]<=t<self.t[-1]) and (self.p[0]<=p<self.p[-1])

    def wrap_p(self,p):
        return p  - 2 * np.pi * np.floor( (p - self.p[0])/(2*np.pi))

    def value(self, r, t, p):
        """Field value at the given coordinate
        """
        p = self.wrap_p(p)
        if not self.in_domain(r,t,p): return np.nan

        i, j, k = self.get_left_idx(r, t, p)

        # Coordinates in local unit cube
        ru = (r-self.r[i])/self.dr
        tu = (t-self.t[j])/self.dt
        pu = (p-self.p[k])/self.dp

        # Trilinear interpolation
        return self.trilinear(self.data, ru, tu, pu, i, j, k)

    def trilinear(self, data, ru, tu, pu, i, j, k):
        """Trilinear interpolation
        """
        l = (k+1)%len(self.p)
        return \
              data[i  , j  , k  ]*(1.0-ru)*(1.0-tu)*(1.0-pu) \
            + data[i+1, j  , k  ]*(    ru)*(1.0-tu)*(1.0-pu) \
            + data[i  , j+1, k  ]*(1.0-ru)*(    tu)*(1.0-pu) \
            + data[i+1, j+1, k  ]*(    ru)*(    tu)*(1.0-pu) \
            + data[i  , j  , l  ]*(1.0-ru)*(1.0-tu)*(    pu) \
            + data[i+1, j  , l  ]*(    ru)*(1.0-tu)*(    pu) \
            + data[i  , j+1, l  ]*(1.0-ru)*(    tu)*(    pu) \
            + data[i+1, j+1, l  ]*(    ru)*(    tu)*(    pu)

    def mesh_spacing_at(self, r, t, p):
        """Returns the mesh spacing at the given coordinates
        """
        return np.array((self.dr, self.dt, self.dp))

    def cell_length_at(self, r, t, p):
        """Returns the physical lengths of the cell at the given coordinates
        """

        i, j, k = self.get_left_idx(r, t, p)
        R   = 0.5*(self.r[i] + self.r[i+1])
        clt = 0.5*(self.t[j] + self.t[j+1])

        return np.array((self.dr, R*self.dt, R*np.sin(clt)*self.dp))

    def values_at(self, coordinates):
        """Compute the scalar for a set of points
        """

        values = np.zeros(len(coordinates))

        for idx in range(len(coordinates)):
            values[idx] = self.value(coordinates[idx][0], coordinates[idx][1], coordinates[idx][2])

        return values
