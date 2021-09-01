"""Constraints module.

Classes for constraints and lists of constraints.

"""
import numpy as np
import casadi as cs

from enum import Enum

from safe_control_gym.envs.benchmark_env import BenchmarkEnv


class ConstrainedVariableType(str, Enum):
    """Allowable constraint type specifiers.

    """

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

    def __init__(self,
                 env: BenchmarkEnv,
                 constrained_variable: ConstrainedVariableType,
                 strict: bool=False,
                 active_dims=None,
                 tolerance=None,
                 **kwargs
                 ):
        """Defines params (e.g. bounds) and state.

        Args:
            env (safe_control_gym.envs.bechmark_env.BenchmarkEnv): The environment the constraint is for.
            constrained_variable (ConstrainedVariableType): Specifies the input type to the constraint as a constraint
                                                         that acts on the state, input, or both.
            strict (optional, bool): Whether the constraint is violated also when equal to its threshold.
            active_dims (list of ints): Filters the constraint to only act only select certian dimensions.
            tolerance (list or np.array): The distance from the constraint at which is_almost_active returns True.

        """
        self.constrained_variable = ConstrainedVariableType(constrained_variable)
        if self.constrained_variable == ConstrainedVariableType.STATE:
            self.dim = env.observation_space.shape[0]
        elif self.constrained_variable == ConstrainedVariableType.INPUT:
            self.dim = env.action_space.shape[0]
        elif self.constrained_variable == ConstrainedVariableType.INPUT_AND_STATE:
            self.dim = env.observation_space.shape[0] + env.action_space.shape[0]
        else:
            raise NotImplementedError('[ERROR] invalid constrained_variable (use STATE, INPUT or INPUT_AND_STATE).')
        # Save the strictness attribute
        self.strict = strict
        # Only want to select specific dimensions, implemented via a filter matrix.
        if active_dims is not None:
            if isinstance(active_dims, int):
                active_dims = [active_dims]
            assert isinstance(active_dims, (list, np.ndarray)), '[ERROR] active_dims is not a list/array.'
            assert (len(active_dims) <= self.dim), '[ERROR] more active_dim than constrainable self.dim'
            assert all(isinstance(n, int) for n in active_dims), '[ERROR] non-integer active_dim.'
            assert all((n < self.dim) for n in active_dims), '[ERROR] active_dim not stricly smaller than self.dim.'
            assert (len(active_dims) == len(set(active_dims))), '[ERROR] duplicates in active_dim'
            self.constraint_filter = np.eye(self.dim)[active_dims]
            self.dim = len(active_dims)
        else:
            self.constraint_filter = np.eye(self.dim)
        if tolerance is not None:
            self.tolerance = np.array(tolerance, ndmin=1)
            # assert self.tolerance.shape[0] == len([self.num_constraints]), '[ERROR] self.tolerance is not of size active_dims'
        else:
            self.tolerance = None
        self.num_input_constraints = None

    def reset(self):
        """Clears up the constraint state (if any).

        """
        pass

    def get_symbolic_model(self,
                           env
                           ):
        """Gets the symbolic form of the constraint function.

        Args:
            env: The environment to constrain.

        Returns:
            obj: The symbolic form of the constraint.

        """
        return NotImplementedError

    def get_value(self,
                  env
                  ):
        """Gets the constraint function value.

        Args:
            env: The environment to constrain.

        Returns:
            ndarray: The evaulation of the constraint.

        """
        env_value = self.get_env_constraint_var(env)
        return np.atleast_1d(np.squeeze(self.sym_func(np.array(env_value, ndmin=1))))

    def is_violated(self,
                    env
                    ):
        """Checks if constraint is violated.

        Args:
            env: The environment to constrain.

        Returns:
            bool: Whether the constraint was violeted.

        """
        c_value = self.get_value(env)
        if self.strict:
            flag = np.any(np.greater_equal(c_value, 0.))
        else:
            flag = np.any(np.greater(c_value, 0.))
        return bool(flag)

    def is_almost_active(self,
                         env
                         ):
        """Checks if constraint is nearly violated.

        This is checked by using a slack variable (from init args).
        This can be used for reward shaping/constraint penalty in RL methods.

        """
        if not hasattr(self, "tolerance") or self.tolerance is None:
            return False
        c_value = self.get_value(env)
        flag = np.any(np.greater(c_value + self.tolerance, 0.))
        return bool(flag)

    def get_env_constraint_var(self,
                               env: BenchmarkEnv
                               ):
        if self.constrained_variable == ConstrainedVariableType.STATE:
            return env.state
        elif self.constrained_variable == ConstrainedVariableType.INPUT:
            return env.current_raw_input_action
        elif self.constrained_variable == ConstrainedVariableType.INPUT_AND_STATE:
            return (env.state, env.current_raw_input_action)
        else:
            raise NotImplementedError("Constraint input type not implemented.")

    def check_tolerance_shape(self):
        if self.tolerance is not None:
            assert len(self.tolerance) == self.num_constraints, '[ERROR] the tolerance dimension does not match the number of constraints.'


class QuadraticContstraint(Constraint):
    """Constraint class for constraints of the form x.T @ P @ x <= b.

    """

    def __init__(self,
                 env: BenchmarkEnv,
                 P: np.ndarray,
                 b: float,
                 constrained_variable: ConstrainedVariableType,
                 strict: bool=False,
                 active_dims=None,
                 tolerance=None
                 ):
        """Initializes the class.

        Args:
            env (safe_control_gym.envs.bechmark_env.BenchmarkEnv): The environment the constraint is for.
            P (np.array): The square matrix representing the quadratic.
            b (float): The scalar limit for the quadatic constraint.
            constrained_variable (ConstrainedVariableType): Specifies the input type to the constraint as a constraint
                                                        that acts on the state, input, or both.
            strict (optional, bool): Whether the constraint is violated also when equal to its threshold.
            active_dims (list of ints): Filters the constraint to only act only select certian dimensions.
            tolerance (list or np.array): The distance from the constraint at which is_almost_active returns True.

        """
        super().__init__(env, constrained_variable, strict=strict, active_dims=active_dims, tolerance=tolerance)
        P = np.array(P, ndmin=1)
        assert P.shape == (self.dim, self.dim), '[ERROR] P has the wrong dimension!'
        self.P = P
        assert isinstance(b, float), '[ERROR] b is not a scalar!'
        self.b = b
        self.num_constraints = 1  # Always scalar.
        self.sym_func = lambda x: x.T @ self.constraint_filter.T @ self.P @ self.constraint_filter @ x - self.b
        self.check_tolerance_shape()

    def get_symbolic_model(self):
        """Gets the symbolic form of the constraint function.

        Returns:
            lambda: The symbolic form of the constraint.

        """
        return self.sym_func


class LinearConstraint(Constraint):
    """Constraint class for constraints of the form A @ x <= b.

    """

    def __init__(self,
                 env: BenchmarkEnv,
                 A: np.ndarray,
                 b: np.ndarray,
                 constrained_variable: ConstrainedVariableType,
                 strict: bool=False,
                 active_dims=None,
                 tolerance=None
                 ):
        """Initialize the class.

        Args:
            env (BenchmarkEnv): The environment to constraint.
            A (np.array or list): A matrix of the constraint (self.num_constraints by self.dim).
            b (np.array or list): b matrix of the constraint (1D array self.num_constraints)
                                  constrained_variable (ConstrainedVariableType): Type of constraint.
            strict (optional, bool): Whether the constraint is violated also when equal to its threshold.
            active_dims (list or int): List specifying which dimensions the constraint is active for.
            tolerance (float): The distance at which is_almost_active(env) triggers.

        """
        super().__init__(env, constrained_variable, strict=strict, active_dims=active_dims, tolerance=tolerance)
        A = np.array(A, ndmin=1)
        b = np.array(b, ndmin=1)
        assert A.shape[1] == self.dim, '[ERROR] A has the wrong dimension!'
        self.A = A
        assert b.shape[0] == A.shape[0], '[ERROR] Dimension 0 of b does not match A!'
        self.b = b
        self.num_constraints = A.shape[0]
        self.sym_func = lambda x: self.A @ self.constraint_filter @ x - self.b
        self.check_tolerance_shape()

    def get_symbolic_model(self):
        """Gets the symbolic form of the constraint function.

        Returns:
            lambda: The symbolic form of the constraint.

        """
        return self.sym_func


class BoundedConstraint(LinearConstraint):
    """ Class for bounded constraints lb <= x <= ub as polytopic constraints -Ix + b <= 0 and Ix - b <= 0.

    """

    def __init__(self,
                 env: BenchmarkEnv,
                 lower_bounds: np.ndarray,
                 upper_bounds: np.ndarray,
                 constrained_variable: ConstrainedVariableType,
                 strict: bool=False,
                 active_dims=None,
                 tolerance=None):
        """Initialize the constraint.

        Args:
            env (BenchmarkEnv): The environment to constraint.
            lower_bounds (np.array or list): Lower bound of constraint.
            upper_bounds (np.array or list): Uppbound of constraint.
            constrained_variable (ConstrainedVariableType): Type of constraint.
            strict (optional, bool): Whether the constraint is violated also when equal to its threshold.
            active_dims (list or int): List specifying which dimensions the constraint is active for.
            tolerance (float): The distance at which is_almost_active(env) triggers.

        """
        self.lower_bounds = np.array(lower_bounds, ndmin=1)
        self.upper_bounds = np.array(upper_bounds, ndmin=1)
        dim = self.lower_bounds.shape[0]
        A = np.vstack((-np.eye(dim), np.eye(dim)))
        b = np.hstack((-self.lower_bounds, self.upper_bounds))
        super().__init__(env, A, b, constrained_variable, strict=strict, active_dims=active_dims, tolerance=tolerance)
        self.num_input_constraints = 2 * dim
        self.check_tolerance_shape()


class DefaultConstraint(BoundedConstraint):
    """Use the environments default observation_space for constraints.

    """

    def __init__(self,
                 env,
                 constrained_variable: ConstrainedVariableType,
                 lower_bounds=None,
                 upper_bounds=None,
                 strict: bool=False,
                 tolerance=None
                 ):
        """"Initialize the class.

        Args:
            env (BenchmarkEnv): Environment for the constraint.
            lower_bounds (list, np.array): 1D array or list of the lower bounds. Length must match
                the environemt observation space dimension. If none, the env defaults are used
            upper_bounds (list, np.array): 1D array or list of the lower bounds. Length must match
                the environemt observation space dimension. If None, the env defaults are used.
            strict (optional, bool): Whether the constraint is violated also when equal to its threshold.
            tolerance (float): The distance at which is_almost_active(env) triggers.

        """
        if constrained_variable==ConstrainedVariableType.STATE:
            default_constraint_space = env.observation_space
        elif constrained_variable==ConstrainedVariableType.INPUT:
            default_constraint_space = env.action_space
        else:
            raise NotImplementedError('[ERROR] DefaultConstraint can only be of type STATE or INPUT')
        if upper_bounds is None:
            upper_bounds = default_constraint_space.high
        else:
            if isinstance(upper_bounds, float):
                upper_bounds = [upper_bounds]
            assert len(upper_bounds) == default_constraint_space.shape[0],\
                ValueError("[ERROR]: Upper bound must have length equal to state dimension.")
        if lower_bounds is None:
            if isinstance(lower_bounds, float):
                lower_bounds = [lower_bounds]
            lower_bounds = default_constraint_space.low
        else:
            assert len(upper_bounds) == default_constraint_space.shape[0],\
                ValueError("[ERROR]: Lower bound must have length equal to state dimension.")
        super().__init__(env,
                         lower_bounds=lower_bounds,
                         upper_bounds=upper_bounds,
                         constrained_variable=constrained_variable,
                         strict=strict,
                         active_dims=None,
                         tolerance=tolerance)


class SymmetricStateConstraint(BoundedConstraint):
    """Symmetric state bound constraint.

    Note: speficially intended for Cartpole and Safe Exploration (Dalal 2018).

    """

    def __init__(self,
                 env,
                 bound,
                 constrained_variable,
                 strict: bool=False,
                 active_dims=None,
                 tolerance=None,
                 **kwrags
                 ):
        """

        """
        assert bound is not None
        self.bound = np.array(bound, ndmin=1)
        super().__init__(env,
                         lower_bounds=-bound,
                         upper_bounds=bound,
                         constrained_variable=constrained_variable,
                         strict=strict,
                         active_dims=active_dims,
                         tolerance=tolerance,
                         **kwrags)
        assert (env.NAME == 'cartpole'), '[ERROR] SymmetricStateConstraint is meant for CartPole environments'
        self.num_constraints = self.bound.shape[0]

    def get_value(self, env):
        c_value = np.abs(self.constraint_filter @ env.state) - self.bound
        return c_value


def get_symbolic_constraint_models(constraint_list):
    """Create list of symbolic models from list of constraints.

    """
    symbolic_models = [con.get_symbolic_model() for con in constraint_list]
    return symbolic_models


class ConstraintList:
    """Collates a (ordered) list of constraints.

    """

    def __init__(self,
                 constraints
                 ):
        """Initialize the constraint list.

        Args:
            constraints: The list of constraints.

        """
        self.constraints = constraints
        self.num_constraints = sum([con.num_constraints for con in self.constraints])
        self.state_constraints = [con for con in self.constraints if con.constrained_variable == ConstrainedVariableType.STATE]
        self.num_state_constraints = sum([con.num_constraints for con in self.state_constraints])
        self.input_constraints = [con for con in self.constraints if con.constrained_variable == ConstrainedVariableType.INPUT]
        self.num_input_constraints = sum([con.num_constraints for con in self.input_constraints])
        self.input_state_constraints = [con for con in self.constraints if con.constrained_variable == ConstrainedVariableType.INPUT_AND_STATE]
        self.num_input_state_constraints = sum([con.num_constraints for con in self.input_state_constraints])

    def __len__(self):
        """Gets the constraint list length.

        Returns:
            int: The number of constraints in the list.

        """
        return len(self.constraints)

    def get_all_symbolic_models(self):
        """Return all the symbolic models the constraints.

        """
        return get_symbolic_constraint_models(self.constraints)

    def get_state_constraint_symbolic_models(self):
        """Return only the constraints that act on the state.

        """
        return get_symbolic_constraint_models(self.state_constraints)

    def get_input_constraint_symbolic_models(self):
        """Return only the constraints that act on the input.

        """
        return get_symbolic_constraint_models(self.input_constraints)

    def get_input_and_state_constraint_symbolic_models(self):
        """Return only the constraints that act on both state and inputs simultaneously.

        """
        return get_symbolic_constraint_models(self.input_state_constraints)

    def get_stacked_symbolic_model(self, env):
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

    def get_values(self,
                   env,
                   only_state=False
                   ):
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

    def get_violations(self,
                       env,
                       only_state=False
                       ):
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

    def is_violated(self,
                    env
                    ):
        """Checks if any of the constraints is violated.

        Args:
            env: The environment to constrain.

        Returns:
            bool: A boolean flag if any constraint is violeted. 

        """
        flag = any([con.is_violated(env) for con in self.constraints])
        return flag

    def is_almost_active(self,
                         env
                         ):
        """Checks if constraint is nearly violated.

        This is checked by using a slack variable (from init args) and can be used
        for reward shaping/constraint penalty in RL methods.

        """
        flag = any([con.is_almost_active(env) for con in self.constraints])
        return flag


 # Move to benchmark_env.
GENERAL_CONSTRAINTS = {
    'linear_constraint': LinearConstraint,
    'quadratic_constraint': QuadraticContstraint,
    'bounded_constraint': BoundedConstraint,
    'default_constraint': DefaultConstraint
}


 # Move to benchmark_env.
def create_ConstraintList_from_list(constraint_yaml_list, available_constraints, env):
    """Create a constraint list from a dict (or YAML output).

    Args:
        constraint_yaml_list (list): List of dicts defining the problem constraints.
        available_constraints (dict): Dict of the constraints that are available
        env (BenchmarkEnv): Env for which the constraints will be applied
    """
    constraint_list = []
    for constraint in constraint_yaml_list:
        assert isinstance(constraint, dict), "[ERROR]: Each constraint must be specified as a dict."
        assert "constraint_form" in constraint.keys(),\
            "[ERROR]: Each constraint must have a key 'constraint_form'"
        con_form = constraint.constraint_form
        assert con_form in available_constraints, "[ERROR]. constraint not in list of available constraints"
        con_class = available_constraints[con_form]
        cfg = { key: constraint[key] for key in constraint if key != "constraint_form"}
        constraint_list.append(con_class(env, **cfg))

    return ConstraintList(constraint_list)
