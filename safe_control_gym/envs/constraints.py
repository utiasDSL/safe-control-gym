"""Constraints module.

A base class for constraints and a class of a list of constraints.

Reference implementations:
    * github.com/google-research/realworldrl_suite/blob/master/docs/README.md
    * github.com/google-research/realworldrl_suite/blob/be7a51cffa7f5f9cb77a387c16bad209e0f851f8/realworldrl_suite/environments/cartpole.py#L38

TODO:
    * Allow for states/inputs to be randomly sampled within constraints limits
    * Constraint List have combined representation, and vertex representation
"""
import numpy as np
from enum import Enum
import casadi as cs
from safe_control_gym.envs.benchmark_env import BenchmarkEnv


class ConstraintInputType(str, Enum):
    """Allowable constraint type specifiers."""

    STATE = 'STATE'  # Constraints who are a function of the state X.
    INPUT = 'INPUT'  # Constraints who are a function of the input U.
    INPUT_AND_STATE = 'INPUT_AND_STATE'  # Constraints who are a function of the input U and state X.


class Constraint:
    """Implements a (state-wise/trajectory-wise/stateful) constraint.

    A constraint can contain multiple scalar-valued constraint functions.
    Each should be represented as g(x) <= 0.

    Attributes:
        dim (int): The number of inequalities representing the constraint.
    """

    dim = 0  # todo: why is this a class attribute?

    def __init__(self, env: BenchmarkEnv, constraint_input_type: ConstraintInputType, active_dims=None, tolerance=None, **kwargs):
        """Defines params (e.g. bounds) and state.

        Args:
            env (safe_control_gym.envs.bechmark_env.BenchmarkEnv): The environment the constraint is for.
            constraint_input_type (ConstraintInputType): Specifies the input type to the constraint as a constraint
            that acts on the state, input, or both
            active_dims (list of ints): Filters the constraint to only act only select certian dimensions
            tolerance (list or np.array): The distance from the constraint at which is_almost_active returns True

        """
        self.constraint_input_type = ConstraintInputType(constraint_input_type)
        if self.constraint_input_type == ConstraintInputType.STATE:
            self.dim = env.observation_space.shape[0]
        elif self.constraint_input_type == ConstraintInputType.INPUT:
            self.dim = env.action_space.shape[0]
        else:
            self.dim = env.observation_space.shape[0] + env.action_space.shape[0]

        # Only want to select specific dimensions. Implemented via a filter matrix
        if active_dims is not None:
            if isinstance(active_dims, int):
                active_dims = [active_dims]
            self.constraint_filter = np.eye(self.dim)[active_dims]
            self.dim = len(active_dims)
        else:
            self.constraint_filter = np.eye(self.dim)

        if tolerance is not None:
            self.tolerance = np.array(tolerance, ndmin=1)
            assert self.tolerance.shape[0] == len(active_dims)
        else:
            self.tolerance = np.array(np.zeros(self.dim), ndmin=1)

        self.num_input_constraints = None

    def reset(self):
        """Clears up the constraint state (if any)."""
        pass

    def get_symbolic_model(self, env):
        """Gets the symbolic form of the constraint function.

        Args:
            env: The environment to constrain.

        Returns:
            obj: The symbolic form of the constraint.
        """
        return NotImplementedError

    def get_value(self, env):
        """Gets the constraint function value.

        Args:
            env: The environment to constrain.

        Returns:
            float: The evaulation of the constraint.

        """
        env_value = self.get_env_constraint_var(env)
        return np.squeeze(self.sym_func(np.array(env_value, ndmin=1)))

    def is_violated(self, env):
        """Checks if constraint is violated.

        Args:
            env: The environment to constrain.

        Returns:
            bool: Whether the constraint was violeted.
        """
        c_value = self.get_value(env)
        flag = np.any(np.greater(c_value, 0.))
        return bool(flag)

    def is_almost_active(self, env):
        """Checks if constraint is nearly violated.

        This is checked by using a slack variable (from init args).
        This can be used for reward shaping/constraint penalty in RL methods.
        """

        if not hasattr(self, "tolerance") or self.tolerance is None:
            return False
        c_value = self.get_value(env)
        flag = np.any(np.greater(c_value + self.tolerance, 0.))
        return bool(flag)

    def get_env_constraint_var(self, env: BenchmarkEnv):
        if self.constraint_input_type == ConstraintInputType.STATE:
            return env.state
        elif self.constraint_input_type == ConstraintInputType.INPUT:
            return env.current_raw_input_action
        elif self.constraint_input_type == ConstraintInputType.INPUT_AND_STATE:
            return (env.state, env.current_raw_input_action)
        else:
            raise NotImplementedError("Constraint input type not implemented.")


# TODO: Add these constraints to benchmark constraints
class QuadraticContstraint(Constraint):
    """Constraint class for constraints of the form x.T @ P @ x <= b."""

    def __init__(self, env: BenchmarkEnv, P: np.ndarray, b: float, constraint_input_type: ConstraintInputType, active_dims=None, tolerance=None):
        """Initializes the class.

        An example of how to specify in YAML, one would add:

        constraints:
          quadratic_constraint:
            P:
              - [1.0, 2.0]
              - [3.0, 4.0]
            b: 1.0
          constraint_input_type: STATE


        Args:
            env (safe_control_gym.envs.bechmark_env.BenchmarkEnv): The environment the constraint is for.
            P (np.array): The square matrix representing the quadratic.
            b (float): The scalar limit for the quadatic constraint.
            constraint_input_type (ConstraintInputType): Specifies the input type to the constraint as a constraint
            that acts on the state, input, or both
            active_dims (list of ints): Filters the constraint to only act only select certian dimensions
            tolerance (list or np.array): The distance from the constraint at which is_almost_active returns True

        """
        super().__init__(env, constraint_input_type, active_dims=active_dims, tolerance=tolerance)
        assert P.shape == (self.dim, self.dim), ValueError("P has the wrong dimension!")
        self.P = P
        assert isinstance(b, float), ValueError("b is not a scalar!")
        self.b = b
        self.num_constraints = 1  # always scalar

        self.sym_func = lambda x: x.T @ self.constraint_filter.T @ self.P @ self.constraint_filter @ x - self.b

    def get_symbolic_model(self):
        """Gets the symbolic form of the constraint function.

        Returns:
            lambda: The symbolic form of the constraint.

        """
        return self.sym_func


class LinearConstraint(Constraint):
    """Constraint class for constraints of the form A @ x <= b"""

    def __init__(self, env: BenchmarkEnv, A: np.ndarray, b: np.ndarray, constraint_input_type: ConstraintInputType, active_dims=None, tolerance=None):
        """Initialize the class.

        An example of how to specify in YAML, one would add:

        constraints:
          linear_constraint:
            A:
              - [1.0, 2.0]
              - [3.0, 4.0]
            b:
              - 1.0
              - 1.0
          constraint_input_type: STATE

        Args:
            env (BenchmarkEnv): The environment to constraint.
            A (np.array or list): A matrix of the constraint (self.num_constraints by self.dim).
            b (np.array or list): b matrix of the constraint (1D array self.num_constraints)
            constraint_input_type (ConstraintInputType): Type of constraint.
            active_dims (list or int): List specifying which dimensions the constraint is active for.
            tolerance (float): The distance at which is_almost_active(env) triggers.

        """
        super().__init__(env, constraint_input_type, active_dims=active_dims, tolerance=tolerance)
        A = np.array(A, ndmin=1)
        b = np.array(b, ndmin=1)
        assert A.shape[1] == self.dim, ValueError("A has the wrong dimension!")
        self.A = A
        assert b.shape[0] == A.shape[0], ValueError("Dimension 0 of b does not match A!")
        self.b = b
        self.num_constraints = A.shape[0]

        self.sym_func = lambda x: self.A @ self.constraint_filter @ x - self.b

    def get_symbolic_model(self):
        """Gets the symbolic form of the constraint function.

        Returns:
            lambda: The symbolic form of the constraint.

        """
        return self.sym_func


class BoundedConstraint(LinearConstraint):
    """ Class for bounded constraints lb <= x <= ub as polytopic constraints -Ix + b <= 0 and Ix - b <= 0 """

    def __init__(self,
                 env: BenchmarkEnv,
                 lower_bounds: np.ndarray,
                 upper_bounds: np.ndarray,
                 constraint_input_type: ConstraintInputType,
                 active_dims=None,
                 tolerance=None):
        """Initialize the constraint.

        An example of how to specify in YAML, one would add:

        constraints:
          bounded_constraint:
            lower_bound:
                - -0.2
                - 0.4
            upper_bound:
              - 0.2
              - 0.8
          constraint_input_type: INPUT


        Args:
            env (BenchmarkEnv): The environment to constraint.
            lower_bounds (np.array or list): Lower bound of constraint.
            upper_bounds (np.array or list): Uppbound of constraint.
            constraint_input_type (ConstraintInputType): Type of constraint.
            active_dims (list or int): List specifying which dimensions the constraint is active for.
            tolerance (float): The distance at which is_almost_active(env) triggers.

        """
        self.lower_bounds = np.array(lower_bounds, ndmin=1)
        self.upper_bounds = np.array(upper_bounds, ndmin=1)
        dim = self.lower_bounds.shape[0]
        A = np.vstack((-np.eye(dim), np.eye(dim)))
        b = np.hstack((-self.lower_bounds, self.upper_bounds))
        super().__init__(env, A, b, constraint_input_type, active_dims=active_dims, tolerance=tolerance)
        self.num_input_constraints = 2 * dim


def get_symbolic_constraint_models(constraint_list):
    """Create list of symbolic models from list of constraints."""
    symbolic_models = [con.get_symbolic_model() for con in constraint_list]
    return symbolic_models


class ConstraintList:
    """Collates a (ordered) list of constraints."""

    def __init__(self, constraints):
        """Initialize the constraint list.

        Args:
            constraints: The list of constraints.

        """
        self.constraints = constraints  # Todo: this could probably be names better since we often see constraints.constraints
        self.num_constraints = sum([con.num_constraints for con in self.constraints])
        self.state_constraints = [con for con in self.constraints if con.constraint_input_type == ConstraintInputType.STATE]
        self.num_state_constraints = sum([con.num_constraints for con in self.state_constraints])
        self.input_constraints = [con for con in self.constraints if con.constraint_input_type == ConstraintInputType.INPUT]
        self.num_input_constraints = sum([con.num_constraints for con in self.input_constraints])
        self.input_state_constraints = [con for con in self.constraints if con.constraint_input_type == ConstraintInputType.INPUT_AND_STATE]
        self.num_input_state_constraints = sum([con.num_constraints for con in self.input_state_constraints])

    def __len__(self):
        """Gets the constraint list length.

        Returns:
            int: The number of constraints in the list.

        """
        return len(self.constraints)

    def get_all_symbolic_models(self):
        """Return all the symbolic models the constraints."""
        return get_symbolic_constraint_models(self.constraints)

    def get_state_constraint_symbolic_models(self):
        """Return only the constraints that act on the state."""
        return get_symbolic_constraint_models(self.state_constraints)

    def get_input_constraint_symbolic_models(self):
        """Return only the constraints that act on the input."""
        return get_symbolic_constraint_models(self.input_constraints)

    def get_input_and_state_constraint_symbolic_models(self):
        """Return only the constraints that act on both state and inputs simultaneously."""
        return get_symbolic_constraint_models(self.input_state_constraints)

    def get_stacked_symbolic_model(self, env):
        # Todo: Can this be removed?
        """Gets the symbolic form of all constraints.

        Args:
            env: The environment to constrain.

        Returns:
            obj: The symbolic form of the constraint.
        """
        symbolic_models = [con.get_symbolic_model() for con in self.constraints]

        X = env.symbolic.x_sym
        U = env.symbolic.u_sym
        stack_c_sym = cs.vertcat(*[func(X, U) for func in symbolic_models])
        sym_func = cs.Function("constraints", [X, U], [stack_c_sym])
        return sym_func

    def get_values(self, env, only_state=False):
        """Gets all constraint function values.

        Args:
            env: The environment to constrain.

        Returns:
            ndarray: An array with the evaluation of each constraint.

        """
        if only_state:
            con_values = np.concatenate([con.get_value(env) for con in self.state_constraints])
        else:
            con_values = np.concatenate([con.get_value(env) for con in self.constraints])
        return con_values

    def get_violations(self, env, only_state=False):
        """Gets all constraint violations.

        Args:
            env: The environment to constrain.

        Returns:
            list: A list of booleans saying whether each constraint was violated.

        """
        if only_state:
            flags = [con.is_violated(env) for con in self.state_constraints]
        else:
            flags = [con.is_violated(env) for con in self.constraints]
        return flags

    def is_violated(self, env):
        """Checks if any of the constraints is violated.

        Args:
            env: The environment to constrain.

        Returns:
            bool: A boolean flag if any constraint is violeted. 

        """
        flag = any([con.is_violated(env) for con in self.constraints])
        return flag

    def is_almost_active(self, env):
        """Checks if constraint is nearly violated.

        This is checked by using a slack variable (from init args).
        This can be used for reward shaping/constraint penalty in RL methods.
        """
        flag = any([con.is_almost_active(env) for con in self.constraints])
        return flag


GENERAL_CONSTRAINTS = {
    'linear_constraint': LinearConstraint,
    'quadratic_constraint': QuadraticContstraint,
    'bounded_constraint': BoundedConstraint,
}


def create_ConstraintList_from_dict(constraint_dict, available_constraints, env):
    """Create a constraint list from a dict (or YAML output).

    Args:
        constraint_dict (dict): Dict specifying the constraint parameters
        available_constraints (dict): Dict of the constraints that are available
        env (BenchmarkEnv): Env for which the constraints will be applied
    """
    constraint_list = []
    for name, cfg in constraint_dict.items():
        assert name in available_constraints, "[ERROR]. constraint not in list of available constraints"
        con_class = available_constraints[name]
        constraint_list.append(con_class(env, **cfg))

    return ConstraintList(constraint_list)
