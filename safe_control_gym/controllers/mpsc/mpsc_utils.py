"""Utility functions for Model Predictive Safety Certification.

"""
import cvxpy as cp
import numpy as np
import pytope as pt

from itertools import product
from functools import partial
from matplotlib.patches import Ellipse

from safe_control_gym.envs.constraints import BoundedConstraint, LinearConstraint


def compute_RPI_set(Acl,
                    w,
                    tau
                    ):
    """Compute a Robust positively invariant set (RPI).

    This follows the method in sec IV A from [1] to compute the robust positively invartiant set through a convex
    optimzation problem (LMI?) (equation 8a and 8b).

    [1] K.P. Wabsersich and M.N. Zeilinger "Linear model predictive safety certification for learning-based control"
        2019. https://arxiv.org/pdf/1803.08552.pdf

    Args:
        Acl (np.array): Closed loop gain matrix A+BK (nx by nx)
        w (np.array): Collection of dynamics error residuals with dim (nx by n_samples)
        tau (float): Coefficient from the s-procedure. Somewhere near 0.9 seems to work.

    Returns:
        P (np.array): P from eqn 8 that defines the ellipsoidal RPI set.

    """
    n_samples = w.shape[1]
    P = cp.Variable(Acl.shape, symmetric=True)
    small = 1e-8
    constraints = []
    constraints += [P >> small]
    for i in range(n_samples):
        w_i = w[:, i, None]
        con_11 = Acl.T @ P @ Acl - tau*P
        con_12 = Acl.T @ P @ w_i
        con_21 = w_i.T @ P @ Acl
        con_22 = w_i.T @ P @ w_i + tau - 1
        constraints += [cp.bmat([[con_11, con_12],
                                 [con_21, con_22]]) << 0]
    prob = cp.Problem(cp.Minimize(-cp.log_det(P)), constraints)
    try:
        results =  prob.solve(solver='MOSEK', verbose=True)
    except cp.SolverError:
        print("[ERROR] RPI computation requires the MOSEK solver.")
        exit()
    return P.value

def ellipse_bounding_box(P):
    """Finds the bounding box of an ellipse defined by x^T P x <= 1.

    Args:
        P (np.array): n by n array defining the ellipse.

    Returns:
        vertices (np.array): An vertical of the vertices (number of verts by dim of space).

    """
    c = np.eye(P.shape[0])
    extremes = []
    for i in range(P.shape[0]):
        extremes.append((np.sqrt(c[:,i, None].T @ np.linalg.inv(P) @ c[:,i, None])[0,0],
                        -np.sqrt(c[:,i, None].T @ np.linalg.inv(P) @ c[:,i, None])[0,0]))
    vertices = list(product(*extremes))
    return np.vstack(vertices)

def get_ellipse_eig_decomp(P):
    """Gets the egienvalue decomposition of an ellipse defined by P.

    Args:
        P (np.array): n by n array defining the ellipse.

    Returns:
        minor_eval (float): Minor axis eigenvalue.
        major_eval (np.array): major axis eigenvalue.
        minor_evec (float): Minor axis eigenvector.
        major_evec (np.array): Major axis eigenvector.

    """
    evals, evecs = np.linalg.eig(P)
    major_axis_ind = np.argmin(evals)
    minor_axis_ind = 0 if major_axis_ind == 1 else 1
    major_eval = evals[major_axis_ind]
    minor_eval = evals[minor_axis_ind]
    major_evec = evecs[:, major_axis_ind]
    minor_evec = evecs[:, minor_axis_ind]
    return minor_eval, major_eval, minor_evec, major_evec

def get_ellipse_angle_rep(P,
                          rads=True
                          ):
    """Gets the angle representation of the ellipse which is required for plotting.

    Args:
        P (np.array): n by n array defining the ellipse.
        rads (bool) : Optional argument to return the angle in rads (True) or degrees (False)

    Returns:
        minor_axis_length (float): Length of the minor axis.
        major_axis_length (float): Length of the major axis.
        alpha (float): Angle between the horizontal axis and the major axis.

    """
    minor_eval, major_eval, minor_evec, major_evec = get_ellipse_eig_decomp(P)
    alpha = np.arctan2(major_evec[1], major_evec[0])
    major_axis_length = 1/np.sqrt(major_eval)
    minor_axis_length = 1/np.sqrt(minor_eval)
    if rads:
        return minor_axis_length, major_axis_length, alpha
    else:
        return minor_axis_length, major_axis_length, alpha*180/np.pi

def add_2d_ellipse(position,
                   cov,
                   ax,
                   legend=None
                   ):
    """Add a 2D Ellipse patch to an axis.

    Args:
        position (np.array): Position of the centre of the ellipse.
        cov (np.array): Covariance matrix (ellipse to be plotted).
        ax (matplotlib.axes): Axis on which to the ellipse should be added.
        legend (str): Optional addition of ellipse legend.

    """
    minor_axis_length, major_axis_length, alpha = get_ellipse_angle_rep(cov, rads=False)
    if legend:
        ellipse = Ellipse(position,
                          2*major_axis_length,
                          2*minor_axis_length,
                          angle=alpha,
                          alpha=0.5,
                          label=legend)
    else:
        ellipse = Ellipse(position,
                          2*major_axis_length,
                          2*minor_axis_length,
                          angle=alpha,
                          alpha=0.5)
    ax.add_artist(ellipse)

def pontryagin_difference_AABB(verts1,
                               verts2
                               ):
    """Computre verts1 (-) verts2.

    Args:
        verts1, verts2 (np.array): Array of vertices ( n vertices by space dimension) stacked vertically
        so that every vertix is a horizonal array:
            vertsi = [v1,
                      v2,
                      ...,
                      vn]
    Returns:
        verts (np.array): Array of pontyagin difference.
        const_func (partial onstraint): Constraint function initialized with parameters from the difference.

    """
    if verts1.shape[1] > 1:
        # If greater than 2 dimensions.
        poly1 = pt.Polytope(verts1)
        poly2 = pt.Polytope(verts2)
        diff = poly1 - poly2
        A = diff.A
        b = diff.b
        verts = diff.V
        const_func = partial(LinearConstraint, A=A, b=b)
        return verts, const_func
    else:
        # If 1D data. Only handles closed compact sets.
        vert2_range = np.ptp(verts2)
        vert_min = np.min(verts1) + vert2_range/2
        vert_max = np.max(verts1) - vert2_range/2
        const_func = partial(BoundedConstraint, lower_bounds=vert_min, upper_bounds=vert_max)
        if vert_max > vert_min:
            return np.vstack((vert_min, vert_max)), const_func
        else:
            print("Warning: Tightend set is the Zero set.")
            return np.array([[0,0]]).T, const_func
