"""Utility module.

For the IROS competition

"""

from math import sqrt, sin, cos
import numpy as np

def cubic_interp(inv_t, x0, xf, dx0, dxf):
    dx = xf - x0
    return (
        x0,
        dx0,
        (3*dx*inv_t - 2*dx0 - dxf)*inv_t,
        (-2*dx*inv_t + (dxf + dx0))*inv_t*inv_t)

def quintic_interp(inv_t, x0, xf, dx0, dxf, d2x0, d2xf):
    dx = xf - x0
    return (
        x0,
        dx0,
        d2x0*0.5,
        ((10*dx*inv_t - (4*dxf+6*dx0))*inv_t - 0.5*(3*d2x0-d2xf))*inv_t,
        ((-15*dx*inv_t + (7*dxf+8*dx0))*inv_t + 0.5*(3*d2x0-2*d2xf))*inv_t*inv_t,
        ((6*dx*inv_t - 3*(dxf+dx0))*inv_t - 0.5*(d2x0-d2xf))*inv_t*inv_t*inv_t)


def f(coeffs, idx, t):
    val = 0
    for coeff in coeffs[:,idx]:
        val *= t
        val += coeff
    return val

def df(coeffs, idx, t):
    coeffs_no = len(coeffs[:,idx])
    val = 0
    for i, coeff in enumerate(coeffs[:-1,idx]):
        val *= t
        factor = (coeffs_no - i - 1)
        val += coeff * factor
    return val

def d2f(coeffs, idx, t):
    coeffs_no = len(coeffs[:,idx])
    val = 0
    for i, coeff in enumerate(coeffs[:-2,idx]):
        val *= t
        factor = (coeffs_no - i - 1)
        val += coeff * factor * (factor - 1)
    return val

def d3f(coeffs, idx, t):
    coeffs_no = len(coeffs[:,idx])
    val = 0
    for i, coeff in enumerate(coeffs[:-3,idx]):
        val *= t
        factor = (coeffs_no - i - 1)
        val += coeff * factor * (factor - 1) * (factor - 2)
    return val

def df_idx(length, ts, x_coeffs, y_coeffs, z_coeffs):
    def df_(t):
        for idx in range(length-1):
            if (ts[idx+1] > t):
                break
        t -= ts[idx]
        dx = df(x_coeffs, idx, t)
        dy = df(y_coeffs, idx, t)
        dz = df(z_coeffs, idx, t)
        return sqrt(dx*dx+dy*dy+dz*dz)
    return df_

# Fill s, v, a, t arrays fro S-curve using j and T
def fill(T, t, s, v, a, j):
    for i in range(7):
        dt = T[i+1]
        t[i+1] = t[i] + dt
        s[i+1] = ((j[i]/6*dt + a[i]/2)*dt + v[i])*dt+ s[i]
        v[i+1] =  (j[i]/2*dt + a[i]  )*dt + v[i]
        a[i+1] =   j[i]  *dt + a[i]

# RPY angles to 3x3 rotation matrix
def rpy2rot(roll, pitch, yaw):
    sr = sin(roll)
    cr = cos(roll)
    sp = sin(pitch)
    cp = cos(pitch)
    sy = sin(yaw)
    cy = cos(yaw)
    m = np.matrix(((cy*cp, -sy*cr+cy*sp*sr,  sy*sr+cy*cr*sp),
                (sy*cp,  cy*cr+sy*sp*sr, -cy*sr+sp*sy*cr),
                (-sp   ,  cp*sr         ,  cp*cr)))
    return m