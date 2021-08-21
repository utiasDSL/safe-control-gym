"""Cartpole constraints.

"""
import casadi as cs
import numpy as np

from safe_control_gym.envs.constraints import BoundedConstraint, ConstraintInputType, Constraint


class CartPoleStateConstraint(BoundedConstraint):
    """Constrain the cart's state to the observation space bounds.

    """

    def __init__(self,
                 env,
                 low=None,
                 high=None,
                 **kwargs):
        """Initialize the constraint.

        Args:
            env (BenchmarkEnv): The environment to constrain.
            low (list): to overwrite the environment minimums.
            high (list): To overwrite the environment maximums.

        """
        if high is None:
            self.high = np.array([env.x_threshold * 2,  # Limit set to 2x: i.e. a failing observation is still within bounds.
                                  np.finfo(np.float32).max,
                                  env.theta_threshold_radians * 2,  # Limit set to 2x: i.e. a failing observation is still within bounds.
                                  np.finfo(np.float32).max])
        else:
            assert len(high) == env.observation_space.shape[0]
            self.high = high
        if low is None:
            self.low = -1 * np.array([env.x_threshold * 2,  # Limit set to 2x: i.e. a failing observation is still within bounds.
                                      np.finfo(np.float32).max,
                                      env.theta_threshold_radians * 2,  # Limit set to 2x: i.e. a failing observation is still within bounds.
                                      np.finfo(np.float32).max])
        else:
            assert len(low) == env.observation_space.shape[0]
            self.low = low
        super().__init__(env, self.low, self.high, ConstraintInputType.STATE, **kwargs)

    def get_value(self, env):
        """Gets the constraint function value.

        Args:
            env: The environment to constrain.

        Returns:
            ndarray: The evaluation of the constraint.

        """
        return np.squeeze(self.sym_func(env.state))

    def is_violated(self, env):
        """Checks if constraint is violated.

        Args:
            env: The environment to constrain.

        Returns:
            bool: Whether the constraint was violated.

        """
        c_value = self.get_value(env)
        flag = np.any(np.greater(c_value, 0.))
        return flag


class CartPoleInputConstraint(BoundedConstraint):
    """Constrain the cart's commanded input to the action space bounds.

    """

    def __init__(self,
                 env,
                 low=None,
                 high=None,
                 **kwargs):
        """Initialize the constraint.

        Args:
            env: The environment to constrain.
            low (float): to overwrite the environment minimums
            high (float): To overwrite the environment maximums
            active.

        """
        if high is None:
            self.high = env.action_threshold
        else:
            assert isinstance(high, float)
            self.high = high
        if low is None:
            self.low = -1 * env.action_threshold
        else:
            assert isinstance(low, float)
            self.low = low
        super().__init__(env, self.low, self.high, ConstraintInputType.INPUT, **kwargs)

    def get_value(self, env):
        """Gets the constraint function value.

        Args:
            env: The environment to constrain.

        Returns:
            ndarray: The evaluation of the constraint.

        """
        return self.sym_func(np.array(env.current_raw_input_action, ndmin=1))

    def is_violated(self, env):
        """Checks if constraint is violated.

        Args:
            env: The environment to constrain.

        Returns:
            bool: Whether the constraint was violated.

        """
        c_value = self.get_value(env)
        flag = np.any(np.greater(c_value, 0.))
        return flag


class CartPoleSymmetricStateConstraint(BoundedConstraint):
    """Symmetric state bound constraint.

     """

    def __init__(self,
                 env,
                 bound,
                 constraint_input_type,
                 active_dims=None,
                 tolerance=None,
                 **kwrags
                 ):
        assert bound is not None
        self.bound = np.array(bound, ndmin=1)
        super().__init__(env,
                         lower_bounds=-bound,
                         upper_bounds=bound,
                         constraint_input_type=constraint_input_type,
                         active_dims=active_dims,
                         tolerance=tolerance,
                         **kwrags)
        self.num_constraints = self.bound.shape[0]

    def get_value(self, env):
        c_value = np.abs(self.constraint_filter @ env.state) - self.bound
        return c_value


class CartPoleBoundConstraint(Constraint):
    """Implements bound constraint for state or input as c(x, u) <= 0.

    Concrete implementation is based on lb <= Ax <= ub, but transformed to
    stack of lb - Ax <= 0 and Ax - ub <= 0, or equivalently,
    A'x + b' <= 0, where A' = [-A; A], b' = [lb; -ub]

    The same applies to u (input/action).
    """

    def __init__(self,
                 env,
                 var_name="state",
                 index=None,
                 low=None,
                 high=None,
                 slack=None
                 ):
        super().__init__()
        assert var_name == "state" or var_name == "input", ("Must specify state or input to constrain.")
        assert low is not None or high is not None, ("Must specify either lower or upper bound.")
        # Which variable (state, input) to constrain.
        self.var_name = var_name
        if var_name == "state":
            var_dim = env.observation_space.shape[0]
        else:
            var_dim = env.action_space.shape[0]
        # Which fields of the variable to bound.
        self.index = index
        # Construct A matrix.
        weight = np.eye(var_dim)
        if isinstance(index, int):
            weight = weight[[index]]
        elif isinstance(index, list):
            weight = weight[index]
        self.low = np.array(low, ndmin=1)
        self.high = np.array(high, ndmin=1)
        # Construct A', b'.
        dim = 0
        full_weight = []
        full_threshold = []
        if self.low is not None:
            dim += len(self.low)
            full_weight.append(-1 * weight)
            full_threshold.append(self.low)
        if self.high is not None:
            dim += len(self.high)
            full_weight.append(weight)
            full_threshold.append(-1 * self.high)
        self.dim = dim
        self.full_weight = np.concatenate(full_weight)
        self.full_threshold = np.concatenate(full_threshold)
        self.slack = np.array(slack, ndmin=1)

    def get_symbolic_model(self, env):
        """

        """
        X = env.symbolic.x_sym
        U = env.symbolic.u_sym
        if self.var_name == "state":
            sym_func = cs.Function("state_bound", [X, U],
                                   [self.full_weight @ X + self.full_threshold]
                                   )
        else:
            sym_func = cs.Function("input_bound", [X, U],
                                   [self.full_weight @ U + self.full_threshold]
                                   )
        return sym_func

    def get_value(self, env):
        """

        """
        if self.var_name == "state":
            var = env.state
        else:
            var = env.current_raw_input_action
            if var is None:
                var = np.zeros(env.action_space.shape[0])
        c_value = self.full_weight @ var + self.full_threshold
        return c_value
