"""
Module to construct a Regularized Biot–Savart Law (RBSL) flux rope for
COCONUT and EUHFORIA simulations in spherical coordinates.

This implementation follows the formulation from Titov et al. (2018) and
builds a discrete flux rope loop plus its mirror, from which the magnetic
field is derived via Biot–Savart integrals.

Typical usage:
    Br, Bth, Bph = RBSL_setup(...)

Author:
    Jinhan Guo (KU Leuven & Nanjing University)
    Adapted and integrated by Luis Linan
"""

import numpy as np
import math
from .utils import ComputeRot

def x_Sph2Car(r0,theta0,phi0):
    return(r0 * np.sin(theta0) * np.cos(phi0))

def y_Sph2Car(r0,theta0,phi0):
    return(r0 * np.sin(theta0) * np.sin(phi0))

def z_Sph2Car(r0,theta0,phi0):
    return(r0 * np.cos(theta0))

def get_lon_lat(x_in,y_in,z_in):

    Ebm=math.sqrt(x_in**2+y_in**2+z_in**2)
    x=x_in/Ebm
    y=y_in/Ebm
    z=z_in/Ebm

    if x == 0 and y == 0:
        lon = 0.0
        if z >= 0.0:
            lat0 = 0.5 * math.pi
        else:
            lat0 = -0.5 * math.pi
    elif z == 0:
        lat0 = 0.0
        if y == 0:
            lon0 = 0.0
        elif x == 0:
            if y >= 0.0:
                lon0 = 0.5 * math.pi
            else:
                lon0 = -0.5 * math.pi
        else:
            lon0 = math.atan(y / x)
    else:
        lat0 = math.asin(z)
        if y == 0:
            lon0 = 0.0
        elif x == 0:
            if y >= 0.0:
                lon0 = 0.5 * math.pi
            else:
                long0 = -0.5 * math.pi
        else:
            lon0 = math.atan(y / x)

    return(lon0, lat0)

def cartesian_to_spherical(x, y, z):
    rad = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arcsin(z / rad)
    lon = np.arctan2(y, x)
    return(lon, lat)

def flux_rope_path(cen_lon_fr,cen_lat_fr,xc,xh,angle_fr,hh_fr,ll_fr,nfr,lon_fr,lat_fr,r_fr):
    x_axis = np.ones(shape=(nfr), dtype='float')
    y_axis = np.ones(shape=(nfr), dtype='float')
    s = np.ones(shape=(nfr), dtype='float')
    ff = np.ones(shape=(nfr), dtype='float')

    for ixp in range(0, nfr):
        s[ixp] = float(ixp) / float(nfr)

    s[0] = 0.0
    s[nfr-1] = 1.0

    for ixp in range(0,nfr):
        if s[ixp] <= xc:
            ff[ixp]=(angle_fr*s[ixp]*(2.0*xc-s[ixp]))/(xc**2)
        else:
            ff[ixp]=(angle_fr*(s[ixp]-2.0*xc+1.0)*(1.0-s[ixp]))/((1.0-xc)**2)


    for ixp in range(0,nfr):
        x_axis[ixp]=(s[ixp]-xc)*np.cos(ff[ixp])+xc
        y_axis[ixp]=(s[ixp]-xc)*np.sin(ff[ixp])



    for ixp in range(0,nfr):
      if x_axis[ixp] <= xh:
         r_fr[ixp] = (x_axis[ixp]*(2.0*xh-x_axis[ixp])*hh_fr)/(xh*xh)
      else:
         r_fr[ixp] = hh_fr*(x_axis[ixp]-2.0*xh+1.0)*(1.0-x_axis[ixp])/((1.0-xh)**2)


    for ixp in range(0, nfr):
        lon_fr[ixp] = x_axis[ixp] * ll_fr - ll_fr/2.0 + cen_lon_fr
        lat_fr[ixp] = y_axis[ixp] * ll_fr + cen_lat_fr


def get_mirror_path(nfr,lon_fr,lat_fr,r_fr):
    # get the flux-rope center
    lon_fp1 = lon_fr[0]
    lon_fp2 = lon_fr[nfr-1]
    lat_fp1 = lat_fr[0]
    lat_fp2 = lat_fr[nfr-1]

    x_fp1 = math.cos(lon_fp1) * math.cos(lat_fp1)
    y_fp1 = math.cos(lat_fp1) * math.sin(lon_fp1)
    z_fp1 = math.sin(lat_fp1)

    x_fp2 = math.cos(lon_fp2) * math.cos(lat_fp2)
    y_fp2 = math.cos(lat_fp2) * math.sin(lon_fp2)
    z_fp2 = math.sin(lat_fp2)

    x_cen = 0.5* (x_fp1 + x_fp2)
    y_cen = 0.5* (y_fp1 + y_fp2)
    z_cen = 0.5* (z_fp1 + z_fp2)
    r_cen = math.sqrt(x_cen**2+y_cen**2+z_cen**2)
    lonc,latc=cartesian_to_spherical(x_cen,y_cen,z_cen)

    lona = lon_fr-lonc
    lata = lat_fr-latc
    ra=r_fr
    xa=ra*np.cos(lata)*np.cos(lona)
    ya=ra*np.cos(lata)*np.sin(lona)
    za=ra*np.sin(lata)
    lonp=lonc-lonc
    latp=latc-latc
    xp=r_cen*np.cos(latp)*np.cos(lonp)
    xm=2.0*xp-xa
    ym=ya
    zm=za


    lon_mir = np.ones(shape=[nfr], dtype='float')
    lat_mir = np.ones(shape=[nfr], dtype='float')

    for inp in range(0,nfr):
        xtmp = xm[inp]
        ytmp = ym[inp]
        ztmp = zm[inp]
        lon_mir[inp],lat_mir[inp]=cartesian_to_spherical(xtmp, ytmp, ztmp)

    lon_mir=lon_mir+lonc
    lat_mir=lat_mir+latc
    r_mir = np.sqrt(xm ** 2 + ym ** 2 + zm ** 2)

    return(lon_mir,lat_mir,r_mir)

def RBSL_flux_rope(a,F_flx,nc,nr,nth,nph,x_car,y_car,z_car,fr_x_ca,fr_y_ca,fr_z_ca):

    re_pi = 1.0 / math.pi
    I_cur = 5.0*np.sqrt(2.0)*F_flx/(3.0*4.0*math.pi*a)
    fsqrt6= 1.0/np.sqrt(6.0)

    AIx = np.zeros(shape=[nr,nth,nph],dtype='float')
    AIy = np.zeros(shape=[nr,nth,nph],dtype='float')
    AIz = np.zeros(shape=[nr,nth,nph],dtype='float')

    AFx = np.zeros(shape=[nr,nth,nph],dtype='float')
    AFy = np.zeros(shape=[nr,nth,nph],dtype='float')
    AFz = np.zeros(shape=[nr,nth,nph],dtype='float')

    Atotal_x = np.zeros(shape=[nr,nth,nph],dtype='float')
    Atotal_y = np.zeros(shape=[nr,nth,nph],dtype='float')
    Atotal_z = np.zeros(shape=[nr,nth,nph],dtype='float')
    r_mag = np.zeros(shape=[nr, nth, nph], dtype='float')


    for ixp in range(0,nc):
        r_vec1 = (x_car - fr_x_ca[ixp])/a
        r_vec2 = (y_car - fr_y_ca[ixp])/a
        r_vec3 = (z_car - fr_z_ca[ixp])/a
        r_mag = np.sqrt(r_vec1**2+r_vec2**2+r_vec3**2)
        f52r = (5.0 - 2.0 * r_mag ** 2)
        if ixp == 0:
            Rpl1 = 0.5*(fr_x_ca[ixp+1]-fr_x_ca[nc-1])
            Rpl2 = 0.5*(fr_y_ca[ixp+1]-fr_y_ca[nc-1])
            Rpl3 = 0.5*(fr_z_ca[ixp+1]-fr_z_ca[nc-1])
        elif ixp == nc-1:
            Rpl1 = 0.5*(fr_x_ca[0]-fr_x_ca[ixp-2])
            Rpl2 = 0.5*(fr_y_ca[0]-fr_y_ca[ixp-2])
            Rpl3 = 0.5*(fr_z_ca[0]-fr_z_ca[ixp-2])
        else:
            Rpl1 = 0.5*(fr_x_ca[ixp+1]-fr_x_ca[ixp-1])
            Rpl2 = 0.5*(fr_y_ca[ixp+1]-fr_y_ca[ixp-1])
            Rpl3 = 0.5*(fr_z_ca[ixp+1]-fr_z_ca[ixp-1])

        Rcr1 = Rpl2*r_vec3 - Rpl3*r_vec2
        Rcr2 = Rpl3*r_vec1 - Rpl1*r_vec3
        Rcr3 = Rpl1*r_vec2 - Rpl2*r_vec1
        sqrt1r = np.where(r_mag < 1.0, np.sqrt(1.0-r_mag**2), 1.0)
        KIr = np.where(r_mag < 1.0, 2.0*re_pi*(np.arcsin(r_mag)/r_mag + f52r*sqrt1r/3.0), 1.0/r_mag)
        KFr = np.where(r_mag < 1.0, 2.0 * re_pi/(r_mag**2) * (np.arcsin(r_mag) / r_mag - sqrt1r) + \
                                    2.0 * re_pi * sqrt1r + f52r * 0.5 * fsqrt6 * (1.0 - \
                                    2.0 * re_pi * np.arcsin((1.0+2.0 * r_mag ** 2) / f52r)), 1.0/r_mag**3)


        AIx = AIx+KIr*Rpl1
        AIy = AIy+KIr*Rpl2
        AIz = AIz+KIr*Rpl3
        AFx = AFx+KFr*Rcr1
        AFy = AFy+KFr*Rcr2
        AFz = AFz+KFr*Rcr3


    AIx = AIx*I_cur/a
    AIy = AIy*I_cur/a
    AIz = AIz*I_cur/a
    AFx = AFx*F_flx*0.25*re_pi/a**2
    AFy = AFy*F_flx*0.25*re_pi/a**2
    AFz = AFz*F_flx*0.25*re_pi/a**2

    Atotal_x = AIx + AFx
    Atotal_y = AIy + AFy
    Atotal_z = AIz + AFz
    return (Atotal_x, Atotal_y, Atotal_z)

def to_spheric(A,rr,tt,pph):
    ''' Transform each components of the vector potential in spherical system
    '''

    A_out = dict({})
    A_out['r'] = np.sin(tt)*np.cos(pph) * A['x'] \
            + np.sin(tt) * np.sin(pph) * A['y'] \
            + np.cos(tt) * A['z']
    A_out['theta'] = np.cos(tt)*np.cos(pph) * A['x'] \
            + np.cos(tt) * np.sin(pph) * A['y'] \
            -np.sin(tt) * A['z']
    A_out['phi'] = -np.sin(pph) * A['x'] \
            + np.cos(pph) * A['y']

    return A_out

def cart_to_sph(Ax_in, Ay_in, Az_in, xx2, xx3):

        sinth = np.sin(xx2)
        costh = np.cos(xx2)
        sinph = np.sin(xx3)
        cosph = np.cos(xx3)
        vsph = [0] * 3

        vsph[0] = sinth * cosph * Ax_in + sinth * sinph * Ay_in + costh * Az_in
        vsph[1] = costh * cosph * Ax_in + costh * sinph * Ay_in - sinth * Az_in
        vsph[2] = -sinph * Ax_in + cosph * Ay_in

        return (vsph[0], vsph[1], vsph[2])

def compute_Is(B_p,R,a):
    ''' Compute the shafranov intensity according to Titov et al. 2014
    (Equation 14)

    Inputs
    ======
    B_p : ambiant magnetic field 
    R : major radius of the torus
    a : minor radius of the torus

    Output
    ======
    Is : Shafranov intensity
    '''

    # eq. 7 of Titov 2014
    Is = - (4 * np.pi * R * B_p) / (np.log(8 * R/a) - 3/2 + 1/2)

    return Is

def RBSL_setup(nfr,nb_r,nb_th,nb_phi,x1, x2, x3, X1,X2,X3,grid_x,grid_y,grid_z,cen_lon_fr,cen_lat_fr,xc,xh,angle_fr,hh_fr,ll_fr,a,F_flx):

    # Step 1: Get the flux-rope path
    lon_fr = np.ones(shape=[nfr], dtype='float')
    lat_fr = np.ones(shape=[nfr], dtype='float')
    r_fr = np.ones(shape=[nfr], dtype='float')
    flux_rope_path(cen_lon_fr, cen_lat_fr, xc, xh, angle_fr, hh_fr, ll_fr, nfr, lon_fr, lat_fr, r_fr)
    r_fr = r_fr + 1.0


    # Step 2: Get the mirror path
    lon_mir, lat_mir, r_mir = get_mirror_path(nfr, lon_fr, lat_fr, r_fr)


    # Step 3: Get a circle
    fr_r_sp = np.ones(shape=[2 * nfr], dtype='float')
    fr_theta_sp = np.ones(shape=[2 * nfr], dtype='float')
    fr_phi_sp = np.ones(shape=[2 * nfr], dtype='float')
    for i in range(0, nfr):
        fr_r_sp[i] = r_fr[i]
        fr_theta_sp[i] = 0.5 * math.pi - lat_fr[i]
        fr_phi_sp[i] = lon_fr[i]
        fr_r_sp[2 * nfr - i - 1] = r_mir[i]
        fr_theta_sp[2 * nfr - i - 1] = 0.5 * math.pi - lat_mir[i]
        fr_phi_sp[2 * nfr - i - 1] = lon_mir[i]

    nc = 2 * nfr
    fr_x_ca = fr_r_sp * np.sin(fr_theta_sp) * np.cos(fr_phi_sp)
    fr_y_ca = fr_r_sp * np.sin(fr_theta_sp) * np.sin(fr_phi_sp)
    fr_z_ca = fr_r_sp * np.cos(fr_theta_sp)

    # Step 4: Calculate the flux-rope magnetic field derived from the RBSL method
    Ax, Ay, Az = RBSL_flux_rope(a, F_flx, nc, nb_r, nb_th, nb_phi, grid_x, grid_y, grid_z, fr_x_ca, fr_y_ca, fr_z_ca)
    A_tot = dict({})
    A_tot['x']=Ax
    A_tot['y']=Ay
    A_tot['z']=Az
    A_sph = to_spheric(A_tot, X1, X2, X3)
    Ar=A_sph['r']
    Atheta=A_sph['theta']
    Aphi=A_sph['phi']


    Br_fr, Bth_fr, Bph_fr = ComputeRot(x1, x2, x3, A_sph['r'],A_sph['theta'],A_sph['phi'], geom='spherical')

    return(Br_fr,Bth_fr,Bph_fr)
