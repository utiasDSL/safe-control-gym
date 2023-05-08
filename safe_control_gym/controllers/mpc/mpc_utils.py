'''General MPC utility functions.'''

import numpy as np
import casadi as cs
import scipy

from safe_control_gym.envs.constraints import ConstraintList


def get_cost_weight_matrix(weights,
                           dim
                           ):
    '''Gets weight matrix from input args.'''
    if len(weights) == dim:
        W = np.diag(weights)
    elif len(weights) == 1:
        W = np.diag(weights * dim)
    else:
        raise Exception('Wrong dimension for cost weights.')
    return W


def discretize_linear_system(A,
                             B,
                             dt,
                             exact=False
                             ):
    '''Discretize a linear system.

    dx/dt = A x + B u
    --> xd[k+1] = Ad xd[k] + Bd ud[k] where xd[k] = x(k*dt)

    Args:
        A: np.array, system transition matrix.
        B: np.array, input matrix.
        dt: scalar, step time interval.
        exact: bool, if to use exact discretization.

    Returns:
        Discretized matrices Ad, Bd.
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
        I = np.eye(state_dim)
        Ad = I + A * dt
        Bd = B * dt
    return Ad, Bd


def compute_discrete_lqr_gain_from_cont_linear_system(dfdx, dfdu, Q_lqr, R_lqr, dt):
    '''Computes the LQR gain used for propograting GP uncertainty from the prior model dynamics.

    Args:
        dfdx (np.array): CT A matrix
        dfdu (np.array): CT B matrix
        Q, R (np.array): Gain matrices
        dt (float): Time discretization

    Retrun:
        lqr_gain (np.array): LQR optimal gain, such that (A+BK) is hurwitz
    '''
    # Determine the LQR gain K to propogate the input uncertainty (doing this at each timestep will increase complexity).
    A, B = discretize_linear_system(dfdx, dfdu, dt)
    P = scipy.linalg.solve_discrete_are(A, B, Q_lqr, R_lqr)
    btp = np.dot(B.T, P)
    lqr_gain = -np.dot(np.linalg.inv(R_lqr + np.dot(btp, B)), np.dot(btp, A))

    return lqr_gain, A, B


def rk_discrete(f, n, m, dt):
    '''Runge Kutta discretization for the function.

    Args:
        f (casadi function): Function to discretize.
        n (int): state dimensions.
        m (int): input dimension.
        dt (float): discretization time.

    Return:
        x_next (casadi function?):
    '''
    X = cs.SX.sym('X', n)
    U = cs.SX.sym('U', m)
    # Runge-Kutta 4 integration
    k1 = f(X, U)
    k2 = f(X + dt / 2 * k1, U)
    k3 = f(X + dt / 2 * k2, U)
    k4 = f(X + dt * k3, U)
    x_next = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    rk_dyn = cs.Function('rk_f', [X, U], [x_next], ['x0', 'p'], ['xf'])

    return rk_dyn


def compute_state_rmse(state_error):
    '''Compute root-mean-square error.'''
    mse = np.mean(state_error ** 2, axis=0)
    state_rmse = np.sqrt(mse)
    state_rmse_scalar = np.sqrt(np.sum(mse))

    return state_rmse, state_rmse_scalar


def reset_constraints(constraints):
    '''Setup the constraints list.

    Args:
        constraints (list): List of constraints controller is subject too.
    '''

    constraints_list = ConstraintList(constraints)
    state_constraints_sym = constraints_list.get_state_constraint_symbolic_models()
    input_constraints_sym = constraints_list.get_input_constraint_symbolic_models()
    if len(constraints_list.input_state_constraints) > 0:
        raise NotImplementedError('[Error] Cannot handle combined state input constraints yet.')
    return constraints_list, state_constraints_sym, input_constraints_sym
