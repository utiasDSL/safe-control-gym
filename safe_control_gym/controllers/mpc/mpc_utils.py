"""General MPC utility functions.

"""
import numpy as np
import scipy
import pytope
from pytope import Polytope
import matplotlib.pyplot as plt


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

def compute_min_RPI_v0(A, wmax, eps=1e-5, s_max=50):
    """Compute the minimal robust positively invariant set (mRPI)
    
    From: Rakovic et al. (2005, IEEE Tran on Automatic Control)

    Note: doesn't work for high-dim state spaces (n > 2)

    Args:
        A: np.array, closed-loop system transition matrix
        wmax: np.array, defines the disturbance bounds |w| < wmax
        eps: float, precision criteria for stopping (eq 14 : Rakovic, 2005)
        s_max: int, max number of iterations
    """
    if abs(np.linalg.det(np.eye(A.shape[0]) + A) - 1) < 1.0e-8:
        raise NotImplementedError("mRPI for nilpotent A not yet implemented")
    W = Polytope(lb=-wmax, ub=wmax)
    return pytope.eps_MRPI(A, W, eps, s_max)

def compute_min_RPI(A, wmax, vol_converge=1e-3, s_max=500, debug=False):
    """Compute the minimal robust positively invariant set (mRPI)
    
    Args:
        A: np.array, closed-loop system transition matrix
        wmax: np.array, defines the disturbance bounds |w| < wmax
        vol_converge: float, algo converges when (vol - volprev) / vol < vol_converge
        s_max: int, max number of iterations (Minkowski additions)
    """
    print('computing min RPI...')
    if abs(np.linalg.det(np.eye(A.shape[0]) + A) - 1) < 1.0e-8:
        raise NotImplementedError("mRPI for nilpotent A not yet implemented")
    Z = Polytope(lb=-wmax, ub=wmax)
    W = Polytope(lb=-wmax, ub=wmax)
    vols = []
    Ai = np.copy(A)
    Ai = np.identity(A.shape[0])
    volprev = 0
    vol_converged = False
    for i in range(s_max):
        Z += Ai * W
        # approximate Z with hypercube
        Z = Polytope(lb=np.min(Z.V, axis=0), ub=np.max(Z.V, axis=0))
        sides = np.max(Z.V, axis=0) - np.min(Z.V, axis=0)
        vol = 1
        for side in sides:
            vol *= side
        vols.append(vol)
        if debug:
            print('i: {} vol: {} Z_sides: {}'.format(i, vol, sides))
        if (vol - volprev) / (vol + 1e-15) < vol_converge:
            print('mRPI converged!')
            vol_converged = True
            break
        Ai = A @ Ai
        volprev = vol
    if debug:
        plt.plot(vols)
        plt.show()
    if not vol_converged:
        raise RuntimeError("volume of mRPI did not converge!")
    print('mRPI: {}'.format(np.max(Z.V, axis=0)))
    return Z
