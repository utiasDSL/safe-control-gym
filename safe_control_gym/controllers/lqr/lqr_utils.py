'''Linear Quadratic Regulator (LQR) utilities.'''

import numpy as np
import scipy.linalg


def compute_lqr_gain(model, x_0, u_0, Q, R, discrete_dynamics=True):
    '''Computes the LQR gain from the model, linearization points, and Q and R matrices.

    Args:
        model (SymbolicModel): The SymbolicModel of the system.
        x_0 (ndarray): The linearization point of the state X.
        u_0 (ndarray): The linearization point of the input U.
        Q (ndarray): The state cost matrix Q.
        R (ndarray): The input cost matrix R.
        discrete_dynamics (bool): If to use discrete or continuous dynamics.

    Returns:
        gain (ndarray): The LQR gain for the system.
    '''

    # Linearization.
    df = model.df_func(x_0, u_0)
    A, B = df[0].toarray(), df[1].toarray()

    # Compute controller gain.
    if discrete_dynamics:
        # x[k+1] = A x[k] + B u[k]
        A, B = discretize_linear_system(A, B, model.dt)
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        btp = np.dot(B.T, P)
        gain = np.dot(np.linalg.inv(R + np.dot(btp, B)),
                      np.dot(btp, A))
    else:
        # dx/dt = A x + B u
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        gain = np.dot(np.linalg.inv(R), np.dot(B.T, P))

    return gain


def discretize_linear_system(A, B, dt, exact=False):
    '''Discretization of a linear system

    dx/dt = A x + B u
    --> xd[k+1] = Ad xd[k] + Bd ud[k] where xd[k] = x(k*dt)

    Args:
        A (ndarray): System transition matrix.
        B (ndarray): Input matrix.
        dt (scalar): Step time interval.
        exact (bool): If to use exact discretization.

    Returns:
        Ad (ndarray): The discrete linear state matrix A.
        Bd (ndarray): The discrete linear input matrix B.
    '''

    state_dim, input_dim = A.shape[1], B.shape[1]

    if exact:
        M = np.zeros((state_dim + input_dim, state_dim + input_dim))
        M[:state_dim, :state_dim] = A
        M[:state_dim, state_dim:] = B

        Md = scipy.linalg.expm(M * dt)
        Ad = Md[:state_dim, :state_dim]
        Bd = Md[:state_dim, state_dim:]
    else:
        Identity = np.eye(state_dim)
        Ad = Identity + A * dt
        Bd = B * dt

    return Ad, Bd


def get_cost_weight_matrix(weights, dim):
    '''Gets weight matrix from input args.

    Args:
        weights (list): A 1D list of weights.
        dim (int): The dimension of the desired cost weight matrix.

    Returns:
        W (ndarray): The cost weight matrix.
    '''

    if len(weights) == dim:
        W = np.diag(weights)
    elif len(weights) == 1:
        W = np.diag(weights * dim)
    else:
        raise Exception('Wrong dimension for cost weights.')
    return W
