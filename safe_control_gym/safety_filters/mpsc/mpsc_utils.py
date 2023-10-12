'''Utility functions for Model Predictive Safety Certification.'''

from enum import Enum
from functools import partial
from itertools import product

import cvxpy as cp
import numpy as np
import pytope as pt

from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.envs.constraints import BoundedConstraint, LinearConstraint


class Cost_Function(str, Enum):
    '''MPSC Cost functions enumeration class.'''

    ONE_STEP_COST = 'one_step_cost'         # Default MPSC cost function.


def compute_RPI_set(Acl,
                    w,
                    tau
                    ):
    '''Compute a Robust positively invariant set (RPI).

    This follows the method in sec IV A from [1] to compute the robust positively invartiant set through a convex
    optimzation problem (LMI?) (equation 8a and 8b).

    [1] K.P. Wabsersich and M.N. Zeilinger 'Linear model predictive safety certification for learning-based control'
        2019. https://arxiv.org/pdf/1803.08552.pdf

    Args:
        Acl (ndarray): Closed loop gain matrix A+BK (nx by nx)
        w (ndarray): Collection of dynamics error residuals with dim (nx by n_samples)
        tau (float): Coefficient from the s-procedure. Somewhere near 0.9 seems to work.

    Returns:
        P (ndarray): P from eqn 8 that defines the ellipsoidal RPI set.
    '''

    n_samples = w.shape[1]
    P = cp.Variable(Acl.shape, PSD=True)
    constraints = []
    for i in range(n_samples):
        w_i = w[:, i, None]
        con_11 = Acl.T @ P @ Acl - tau * P
        con_12 = Acl.T @ P @ w_i
        con_21 = w_i.T @ P @ Acl
        con_22 = w_i.T @ P @ w_i + tau - 1
        constraints += [cp.bmat([[con_11, con_12],
                                 [con_21, con_22]]) << 0]
    prob = cp.Problem(cp.Minimize(-cp.log_det(P)), constraints)
    try:
        prob.solve(solver=cp.MOSEK, verbose=True)
        if 'optimal' not in prob.status:
            raise cp.SolverError
    except cp.SolverError:
        print('[ERROR] RPI Computation failed. Ensure you have the MOSEK solver. Otherwise, error unknown.')
        exit()
    return P.value


def ellipse_bounding_box(P):
    '''Finds the bounding box of an ellipse defined by x^T P x <= 1.

    Args:
        P (ndarray): n by n array defining the ellipse.

    Returns:
        vertices (ndarray): A vertical of the vertices (number of verts by dim of space).
    '''

    c = np.eye(P.shape[0])
    extremes = []
    for i in range(P.shape[0]):
        extremes.append((np.sqrt(c[:, i, None].T @ np.linalg.inv(P) @ c[:, i, None])[0, 0],
                         -np.sqrt(c[:, i, None].T @ np.linalg.inv(P) @ c[:, i, None])[0, 0]))
    vertices = list(product(*extremes))
    return np.vstack(vertices)


def pontryagin_difference_AABB(verts1,
                               verts2
                               ):
    '''Computre verts1 (-) verts2.

    Args:
        verts1, verts2 (ndarray): Array of vertices ( n vertices by space dimension) stacked vertically
        so that every vertix is a horizonal array:
            vertsi = [v1,
                      v2,
                      ...,
                      vn]
    Returns:
        verts (ndarray): Array of pontyagin difference.
        const_func (partial constraint): Constraint function initialized with parameters from the difference.
    '''

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
        vert_min = np.min(verts1) + vert2_range / 2
        vert_max = np.max(verts1) - vert2_range / 2
        const_func = partial(BoundedConstraint, lower_bounds=vert_min, upper_bounds=vert_max)
        if vert_max > vert_min:
            return np.vstack((vert_min, vert_max)), const_func
        else:
            print('Warning: Tightend set is the Zero set.')
            return np.array([[0, 0]]).T, const_func


def get_trajectory_on_horizon(env, iteration, horizon):
    '''Gets the trajectory segment for the next horizon steps.

    Args:
        env (BenchmarkEnv): Environment for the task.
        iteration (int): The current iteration, used for trajectory tracking.
        horizon (int): The MPC horizon.

    Returns:
        clipped_X_GOAL (ndarray): The trajectory for the next horizon steps.
    '''

    if env.TASK == Task.TRAJ_TRACKING:
        wp_idx = [
            min(iteration + i, env.X_GOAL.shape[0] - 1)
            for i in range(horizon)
        ]
        clipped_X_GOAL = env.X_GOAL[wp_idx]
    else:
        clipped_X_GOAL = env.X_GOAL

    return clipped_X_GOAL
