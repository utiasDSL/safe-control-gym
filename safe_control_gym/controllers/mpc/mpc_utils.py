"""General MPC utility functions.

"""
import numpy as np
import scipy
from pytope import Polytope
import pytope


def get_cost_weight_matrix(weights,
                           dim
                           ):
    """Gets weight matrix from input args.

    """
    if len(weights) == dim:
        W = np.diag(weights)
    elif len(weights) == 1:
        W = np.diag(weights * dim)
    else:
        raise Exception("Wrong dimension for cost weights.")
    return W


def discretize_linear_system(A,
                             B,
                             dt,
                             exact=False
                             ):
    """Discretize a linear system.
    
    dx/dt = A x + B u
    --> xd[k+1] = Ad xd[k] + Bd ud[k] where xd[k] = x(k*dt)

    Args:
        A: np.array, system transition matrix.
        B: np.array, input matrix.
        dt: scalar, step time interval.
        exact: bool, if to use exact discretization.

    Returns:
        Discretized matrices Ad, Bd.

    """
    state_dim, input_dim = A.shape[1], B.shape[1]
    if exact:
        M = np.zeros((state_dim + input_dim, state_dim + input_dim))
        M[:state_dim, :state_dim] = A
        M[:state_dim, state_dim:] = B
        Md = scipy.linalg.expm(M * dt)
        Ad = Md[:state_dim, :state_dim]
        Bd = Md[:state_dim, state_dim:]
    else:
        I = np.eye(state_dim)
        Ad = I + A * dt
        Bd = B * dt
    return Ad, Bd

def compute_min_RPI(A, wmax, eps=1e-5, s_max=50):
    """Compute the minimal robust positively invariant set (mRPI)
    
    From: Rakovic et al. (2005, IEEE Tran on Automatic Control)

    Args:
        A: np.array, system transition matrix
        w: np.array, defines the disturbance bounds |w| < wmax
        eps: float, precision criteria for stopping (eq 14 : Rakovic, 2005)
        s_max: int, max number of iterations
    """
    if abs(np.linalg.det(np.eye(A.shape[0]) + A) - 1) < 1.0e-8:
        raise NotImplementedError("mRPI for nilpotent A not yet implemented")

    W = Polytope(lb=-wmax, ub=wmax)
    return pytope.eps_MRPI(A, W, eps, s_max)
