"""Module descripion.

"""
import numpy as np
import scipy


def get_cost_weight_matrix(weights, dim):
    """Gets weight matrix from input args.


    """
    if len(weights) == dim:
        W = np.diag(weights)
    elif len(weights) == 1:
        W = np.diag(weights * dim)
    else:
        raise Exception("Wrong dimension for cost weights.")
    return W


def discretize_linear_system(A, B, dt, exact=False):
    """ discretization of a linear system 
    
    dx/dt = A x + B u 
    --> xd[k+1] = Ad xd[k] + Bd ud[k] where xd[k] = x(k*dt)

    Args:
        A: np.array, system transition matrix  
        B: np.array, input matrix 
        dt: scalar, step time interval 
        exact: bool, if to use exact discretization 

    Returns:
        discretized matrices Ad, Bd 
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
