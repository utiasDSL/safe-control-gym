"""Utility module.

For the IROS competition

"""

from math import sqrt, sin, cos, atan2, pi
import numpy as np

def cubic_interp(inv_t, x0, xf, dx0, dxf):
    dx = xf - x0
    return np.flip((
        x0,
        dx0,
        (3*dx*inv_t - 2*dx0 - dxf)*inv_t,
        (-2*dx*inv_t + (dxf + dx0))*inv_t*inv_t))

def quintic_interp(inv_t, x0, xf, dx0, dxf, d2x0, d2xf):
    dx = xf - x0
    return np.flip((
        x0,
        dx0,
        d2x0*0.5,
        ((10*dx*inv_t - (4*dxf+6*dx0))*inv_t - 0.5*(3*d2x0-d2xf))*inv_t,
        ((-15*dx*inv_t + (7*dxf+8*dx0))*inv_t + 0.5*(3*d2x0-2*d2xf))*inv_t*inv_t,
        ((6*dx*inv_t - 3*(dxf+dx0))*inv_t - 0.5*(d2x0-d2xf))*inv_t*inv_t*inv_t))


def f(coeffs, t):
    return np.polyval(coeffs, t)

def df(coeffs, t):
    return np.polyval(np.polyder(coeffs), t)

def d2f(coeffs, t):
    return np.polyval(np.polyder(coeffs,2), t)

def d3f(coeffs, t):
    return np.polyval(np.polyder(coeffs,3), t)

def df_idx(length, ts, x_coeffs, y_coeffs, z_coeffs):
    def df_(t):
        for idx in range(length-1):
            if (ts[idx+1] > t):
                break
        t -= ts[idx]
        dx = df(x_coeffs[:, idx], t)
        dy = df(y_coeffs[:, idx], t)
        dz = df(z_coeffs[:, idx], t)
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

# Get nearest path segment (in x y plane)
def get_nearest_path_segment(new_waypoint, waypoints):
    if(len(waypoints) < 2):
        return 0.0
    min_dist_to_path = np.Inf
    min_idx = 0
    for idx in range(len(waypoints)-1):
        vec = np.array(waypoints[idx+1] - waypoints[idx])
        main_vec = new_waypoint - waypoints[idx]
        vec[2] = main_vec[2] = 0.0
        dist_to_path = np.dot(vec, main_vec) * (1/np.linalg.norm(vec))
        if(dist_to_path < min_dist_to_path):
            min_dist_to_path = dist_to_path
            min_idx = idx
    print(idx)
    return min_idx+1

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
            insertion_idx = get_nearest_path_segment(new_waypoint, waypoints)
            waypoints = np.insert(waypoints, insertion_idx, new_waypoint, axis=0)

    return is_collision, waypoints

def check_intersect_poly(x_coeff, y_coeff, dt, x, y, r):
    # Function describing distance between spline and point (x,y)
    x_poly = np.copy(x_coeff)
    x_poly[-1] -= x
    y_poly = np.copy(y_coeff)
    y_poly[-1] -= y
    LHS = np.polyadd(np.polymul(x_poly, x_poly), np.polymul(y_poly, y_poly))

    # Find stationary points
    dLHS = np.polyder(LHS)
    dLHS_roots = np.real_if_close(np.roots(dLHS))

    # Find stationarty points that lie within radius of obstacle
    tolerance = 0.1 # parameter to say how near we can be to the obs. includes drone size as well
    r_2 = (r + tolerance)**2
    selected = []
    for root in dLHS_roots:
        if root.imag:
            continue
        root = root.real
        if root < 0 or root > dt:
            continue
        LHS_curr = np.polyval(LHS, root)
        if LHS_curr > r_2:
            continue
        dist = sqrt(LHS_curr)
        selected.append((root, dist))

    if not selected:
        # no collision
        return

    # Collect all collisions
    projected_points = []
    for t, dist in selected:
        spline_x = f(x_coeff, t)
        spline_y = f(y_coeff, t)
        spline_dx = df(x_coeff, t)
        spline_dy = df(y_coeff, t)
        spline_d2x = d2f(x_coeff, t)
        spline_d2y = d2f(y_coeff, t)
        scale = (r + 2*tolerance) / dist
        # print("scale", scale)
        projected_x = x + (spline_x - x) * scale
        projected_y = y + (spline_y - y) * scale
        projected_points.append((t, (projected_x, spline_dx, spline_d2x),(projected_y, spline_dy, spline_d2y)))
    return projected_points