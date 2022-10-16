"""Utility module.

For the IROS competition

"""

from math import sqrt, sin, cos, atan2, pi
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

def is_intersect(waypoints, obstacle, low, high):
    [x,y] = obstacle[0:2]
    x_min, x_max = x + low, x + high
    y_min, y_max = y + low, y + high
    for idx, waypoint in enumerate(waypoints):
        [x, y] = waypoint[0:2]
        # print((x,y), obstacle, (x_min, y_min), (x_max, y_max))
        if x_min <= x and x <= x_max and y_min <= y and y <= y_max:
            return True, idx
    return False, None

# Distance is distance away from obstacle that vector between spline_waypoint and obstacle is projected
# returns new waypoint [x,y,z,yaw]
def project_point(spline_waypoint, obstacle, distance):
    [x1,y1] = spline_waypoint[0:2]
    [x2,y2] = obstacle[0:2]
    vec = np.array([x1-x2, y1-y2])
    norm_vec = vec * (1/np.linalg.norm(vec))
    new_waypoint = norm_vec * distance + np.array(obstacle[0:2])
    z = spline_waypoint[2]
    yaw = pi/2. + atan2(new_waypoint[1], new_waypoint[0])
    new_waypoint = [new_waypoint[0], new_waypoint[1], z, yaw]
    print("new_waypoint: ", new_waypoint)
    return new_waypoint

def update_waypoints_avoid_obstacles(spline_waypoints, waypoints, obstacles, initial_info):
    is_collision = False

    ### Determine range around obstacle that is dangerous (low,high)
    obstacle_distrib_dict = initial_info['gates_and_obs_randomization']
    tolerance = 0.1 # magic number to say how near we can be to the obs. includes drone size as well
    low = -initial_info['obstacle_dimensions']['radius'] - tolerance
    high = initial_info['obstacle_dimensions']['radius'] + tolerance
    if 'obstacles' in obstacle_distrib_dict:
        obstacle_distrib_dict = obstacle_distrib_dict['obstacles']
        low -= obstacle_distrib_dict['low']
        high += obstacle_distrib_dict['high']
    
    # Check if obstacle position uncertainty intersects with waypoint path 
    for obstacle in obstacles:
        collision, idx = is_intersect(spline_waypoints, obstacle, low, high)
        if collision:
            is_collision = True
            print("Collision!")
            dist = max(-low, high) + 0.5
            new_waypoint = project_point(spline_waypoints[idx], obstacle, dist)
            # TODO: Fix indexing!!
            waypoints = np.insert(waypoints, 1, new_waypoint, axis=0)

    return is_collision, waypoints
