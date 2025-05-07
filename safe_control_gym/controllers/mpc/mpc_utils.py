'''General MPC utility functions.'''

import casadi as cs
import numpy as np
import scipy
import scipy.linalg

from safe_control_gym.controllers.lqr.lqr_utils import discretize_linear_system
from safe_control_gym.envs.constraints import ConstraintList


def get_cost_weight_matrix(weights,
                           dim
                           ):
    '''Get weight matrix from input arguments.

    Args:
        weights (list): List of weights.
        dim (int): Dimension of the matrix.

    Returns:
        W (np.array): Weight matrix.
    '''
    if len(weights) == dim:
        W = np.diag(weights)
    elif len(weights) == 1:
        W = np.diag(weights * dim)
    else:
        raise Exception('Wrong dimension for cost weights.')
    return W


def compute_discrete_lqr_gain_from_cont_linear_system(dfdx,
                                                      dfdu,
                                                      Q_lqr,
                                                      R_lqr,
                                                      dt
                                                      ):
    '''Compute the LQR gain used for propagating GP uncertainty from the prior model dynamics.

    Args:
        dfdx (np.array): Continuous-time A matrix.
        dfdu (np.array): Continuous-time B matrix.
        Q_lqr (np.array): State cost matrix.
        R_lqr (np.array): Input cost matrix.
        dt (float): Time discretization.

    Returns:
        lqr_gain (np.array): LQR optimal gain, such that (A+BK) is Hurwitz.
        A (np.array): Discretized A matrix.
        B (np.array): Discretized B matrix.
        P (np.array): Solution to the discrete-time Riccati equation.
    '''
    # Determine the LQR gain K to propogate the input uncertainty (doing this at each timestep will increase complexity).
    A, B = discretize_linear_system(dfdx, dfdu, dt)
    P = scipy.linalg.solve_discrete_are(A, B, Q_lqr, R_lqr)
    btp = np.dot(B.T, P)
    lqr_gain = -np.dot(np.linalg.inv(R_lqr + np.dot(btp, B)), np.dot(btp, A))

    return lqr_gain, A, B, P


def rk_discrete(f, n, m, dt):
    '''Runge-Kutta discretization for the function.

    Args:
        f (casadi.Function): Function to discretize.
        n (int): State dimension.
        m (int): Input dimension.
        dt (float): Discretization time.

    Returns:
        rk_dyn (casadi.Function): Discretized function.
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
    '''Compute root-mean-square error.

    Args:
        state_error (np.array): State error array.

    Returns:
        state_rmse (np.array): Root-mean-square error of the state.
        state_rmse_scalar (float): Total RMSE across all states.
    '''
    mse = np.mean(state_error ** 2, axis=0)
    state_rmse = np.sqrt(mse)
    state_rmse_scalar = np.sqrt(np.sum(mse))

    return state_rmse, state_rmse_scalar


def reset_constraints(constraints):
    '''Set up the constraints list.

    Args:
        constraints (list): List of constraints the controller is subject to.

    Returns:
        constraints_list (ConstraintList): List of constraints.
        state_constraints_sym (list): Symbolic state constraints.
        input_constraints_sym (list): Symbolic input constraints.
    '''

    constraints_list = ConstraintList(constraints)
    state_constraints_sym = constraints_list.get_state_constraint_symbolic_models()
    input_constraints_sym = constraints_list.get_input_constraint_symbolic_models()
    if len(constraints_list.input_state_constraints) > 0:
        raise NotImplementedError('[Error] Cannot handle combined state input constraints yet.')
    return constraints_list, state_constraints_sym, input_constraints_sym


def set_acados_constraint_bound(constraint,
                                bound_type,
                                bound_value=None,
                                ):
    '''Set the acados constraint bound.

    Args:
        constraint (casadi.MX or casadi.SX): Constraint expression.
        bound_type (str): Type of bound ('lb' or 'ub').
        bound_value (float, optional): Value of the bound.

    Returns:
        bound (np.array): Constraint bound value.

    Note:
        All constraints in safe-control-gym are defined as g(x, u) <= constraint_tol.
        However, acados requires the constraints to be defined as lb <= g(x, u) <= ub.
        Thus, a large negative number (-1e8) is used as the lower bound.
        See: https://github.com/acados/acados/issues/650
    '''
    if bound_value is None:
        if bound_type == 'lb':
            bound_value = -1e8
        elif bound_type == 'ub':
            bound_value = 1e-6

    return bound_value * np.ones(constraint.shape)
