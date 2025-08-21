"""
Utility functions for numerical derivatives and vector field rotation.

This module provides helper tools commonly used in MHD or plasma simulations, including:
- `nufd1`: construction of first-order derivative operators on non-uniform grids using finite differences.
- `ComputeRot`: computation of the curl (∇×F) of a vector field in Cartesian or spherical geometry.

These functions were developed by Antoine Strugarek (CEA) and are intended for post-processing
or analysis of vector fields, such as magnetic fields in flux rope models.

Functions:
    - `nufd1(x)`: returns a sparse matrix (scipy COO format) representing the first-order derivative operator.
    - `ComputeRot(x1, x2, x3, var1, var2, var3, geom='cartesian')`: computes the curl of a 3D vector field
      in either Cartesian or spherical coordinates, with support for 2.5D geometries.
"""


import numpy as np
from scipy.sparse import coo_matrix
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def nufd1(x):
    """
    Build a finite-difference matrix for first-order derivatives on a non-uniform grid.

    This function returns a second-order accurate sparse matrix that approximates the
    first derivative operator ∂/∂x on a given 1D grid, even if the grid spacing is non-uniform.

    Args:
        x (array-like): 1D array of coordinates (assumed to be ordered).

    Returns:
        scipy.sparse.coo_matrix: Sparse matrix of shape (n, n) representing the derivative operator.

    Notes:
        This function was developed by Antoine Strugarek (CEA).
    """

    n = len(x)
    if (n == 1):
        return 1.
    h = x[1:]-x[:n-1]
    a0 = -(2*h[0]+h[1])/(h[0]*(h[0]+h[1]))
    ak = -h[1:]/(h[:n-2]*(h[:n-2]+h[1:]))
    an = h[-1]/(h[-2]*(h[-1]+h[-2]))
    b0 = (h[0]+h[1])/(h[0]*h[1])
    bk = (h[1:] - h[:n-2])/(h[:n-2]*h[1:])
    bn = -(h[-1]+h[-2])/(h[-1]*h[-2])
    c0 = -h[0]/(h[1]*(h[0]+h[1]))
    ck = h[:n-2]/(h[1:]*(h[:n-2]+h[1:]))
    cn = (2*h[-1]+h[-2])/(h[-1]*(h[-2]+h[-1]))
    val  = np.hstack((a0,ak,an,b0,bk,bn,c0,ck,cn))
    row = np.tile(np.arange(n),3)
    dex = np.hstack((0,np.arange(n-2),n-3))
    col = np.hstack((dex,dex+1,dex+2))
    D = coo_matrix((val,(row,col)),shape=(n,n))

    return D


def ComputeRot(x1,x2,x3,var1,var2,var3,geom='cartesian'):
    """
    Compute the curl (∇×F) of a vector field in Cartesian or spherical coordinates.

    Args:
        x1 (array-like): Grid points in the first dimension (r or x).
        x2 (array-like): Grid points in the second dimension (θ or y).
        x3 (array-like): Grid points in the third dimension (φ or z).
        var1 (ndarray): First component of the vector field (F₁).
        var2 (ndarray): Second component of the vector field (F₂).
        var3 (ndarray): Third component of the vector field (F₃).
        geom (str): Geometry type. 'cartesian' or 'spherical'.

    Returns:
        list of ndarray: Components [∇×F]_1, [∇×F]_2, [∇×F]_3 of the curl of the vector field.

    Notes:
        - The implementation supports 3D and 2.5D geometries.
        - For spherical geometry, `x1`, `x2`, and `x3` must represent [r, θ, φ] respectively.
        - This function was developed by Antoine Strugarek (CEA).
    """
    Rot1=np.zeros_like(var1); Rot2=np.zeros_like(var1); Rot3=np.zeros_like(var1);
    if geom == 'cartesian':
        deriv_x = nufd1(x1) ; deriv_y = nufd1(x2) ; deriv_z = nufd1(x3)
        if len(x3) != 1:
            for ix,xx in enumerate(x1):
                for iz,zz in enumerate(x3):
                    Rot1[ix,:,iz] +=  (deriv_y*var3[ix,:,iz])
                    Rot3[ix,:,iz] += -(deriv_y*var1[ix,:,iz])
                for iy,yy in enumerate(x2):
                    Rot1[ix,iy,:] += -(deriv_z*var2[ix,iy,:])
                    Rot2[ix,iy,:] +=  (deriv_z*var1[ix,iy,:])
            for iy,yy in enumerate(x2):
                for iz,zz in enumerate(x3):
                    Rot2[:,iy,iz] += -(deriv_x*var3[:,iy,iz])
                    Rot3[:,iy,iz] +=  (deriv_x*var2[:,iy,iz])
        else:
            logger.warning("Rotational not coded yet in cartesian in 2.5D")
    elif geom == 'spherical':
        deriv_r = nufd1(x1) ; deriv_t = nufd1(x2) ; deriv_p = nufd1(x3)
        if len(x3) == 1:
            for ix,xx in enumerate(x1):
                Rot1[ix,:] +=  (deriv_t*(np.sin(x2)*var3[ix,:]))/(xx*np.sin(x2))
                Rot3[ix,:] += -(deriv_t*var1[ix,:])/xx
            for iy,yy in enumerate(x2):
                Rot2[:,iy] += -(deriv_r*(x1*var3[:,iy]))/x1
                Rot3[:,iy] +=  (deriv_r*(x1*var2[:,iy]))/x1
        else:
            for ix,xx in enumerate(x1):
                for iz,zz in enumerate(x3):
                    Rot1[ix,:,iz] +=  (deriv_t*(np.sin(x2)*var3[ix,:,iz]))/(xx*np.sin(x2))
                    Rot3[ix,:,iz] += -(deriv_t*var1[ix,:,iz])/xx
                for iy,yy in enumerate(x2):
                    Rot1[ix,iy,:] += -(deriv_p*var2[ix,iy,:])/(xx*np.sin(yy))
                    Rot2[ix,iy,:] +=  (deriv_p*var1[ix,iy,:])/(xx*np.sin(yy))
            for iy,yy in enumerate(x2):
                for iz,zz in enumerate(x3):
                    Rot2[:,iy,iz] += -(deriv_r*(x1*var3[:,iy,iz]))/x1
                    Rot3[:,iy,iz] +=  (deriv_r*(x1*var2[:,iy,iz]))/x1
    else:
        logger.warning('Rotation for '+geom+' not coded yet')

    return [Rot1,Rot2,Rot3]
