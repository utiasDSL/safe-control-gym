"""Quadrotor constraints.

"""
import casadi as cs
import numpy as np

from safe_control_gym.envs.constraints import Constraint, ConstraintInputType, BoundedConstraint
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType


class QuadrotorStateConstraint(BoundedConstraint):
    """Constrain the quadrotor's state to the observation space bounds.

    """
    def __init__(self,
                 env,
                 low=None,
                 high=None,
                 **kwargs):
        """Initialize the constraint.

       Args:
            env (BenchmarkEnv): The environment to constrain.
            low (list): to overwrite the environment minimums
            high (list): To overwrite the environment maximums

        """
        if env.QUAD_TYPE == QuadType.ONE_D:
            # x = {z, z_dot}
            self.low = np.array([0,
                                 -np.finfo(np.float32).max])
            self.high = np.array([env.z_threshold * 2,
                                  np.finfo(np.float32).max])
        elif env.QUAD_TYPE == QuadType.TWO_D:
            # x = {x, x_dot, z, z_dot, theta, theta_dot}
            self.low = np.array([-env.x_threshold * 2,
                                 -np.finfo(np.float32).max,
                                 env.GROUND_PLANE_Z * 2,
                                 -np.finfo(np.float32).max,
                                 -env.theta_threshold_radians * 2,
                                 -np.finfo(np.float32).max])
            self.high = np.array([env.x_threshold * 2,
                                  np.finfo(np.float32).max,
                                  env.z_threshold * 2,
                                  np.finfo(np.float32).max,
                                  env.theta_threshold_radians * 2,
                                  np.finfo(np.float32).max])

        # Overwrite with provided high and low
        if high is not None:
            assert len(high) == env.observation_space.shape[0]
            self.high = high
        if low is not None:
            assert len(low) == env.observation_space.shape[0]
            self.low = low

        super().__init__(env, self.low, self.high, ConstraintInputType.STATE,  **kwargs)

    def get_value(self, env):
        """Gets the constraint function value.

        Args:
            env: The environment to constrain.

        Returns:
            ndarray: The evaluation of the constraint.

        """

        return np.squeeze(self.sym_func(env.state))



class QuadrotorDiagConstraint(Constraint):
    """Constrain the quadrotor's commanded input to the action space bounds.

    """

    def __init__(self, env):
        """Initialize the constraint.

        Args:
            env: The environment to constrain.

        """
        dim = env.observation_space.shape[0]
        if env.QUAD_TYPE == QuadType.TWO_D:
            self.h = np.zeros((1, dim))
            self.h[0, 0] = -1
            self.h[0, 2] = 1
            self.b = 1.1
        else:
            raise NotImplementedError

        self.sym_func = lambda x: self.h @ x - self.b

    def get_symbolic_model(self, env):
        """Gets the symbolic form of the constraint function.

        Args:
            env: The environment to constrain.

        Returns:
            lambda: The symbolic form of the constraint.

        """
        return self.sym_func

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

class QuadrotorInputConstraint(BoundedConstraint):
    """Constrain the quadrotor's commanded input to the action space bounds.

    """

    def __init__(self,
                 env,
                 low=None,
                 high=None,
                 **kwargs):
        """Initialize the constraint.

       Args:
            env (BenchmarkEnv): The environment to constrain.
            low (list): to overwrite the environment minimums
            high (list): To overwrite the environment maximums

        """
        if high is None:
            self.high = env.MAX_THRUST * np.ones(int(env.QUAD_TYPE))
        else:
            assert len(high) == env.action_space.shape[0]
            self.high = high

        if low is None:
            self.low = np.zeros(int(env.QUAD_TYPE))
        else:

            assert len(low) == env.action_space.shape[0]
            self.low = low

        super().__init__(env, self.low, self.high, ConstraintInputType.INPUT, **kwargs)

    def get_value(self, env):
        """Gets the constraint function value.

        Args:
            env: The environment to constrain.

        Returns:
            ndarray: The evaluation of the constraint.

        """
        return np.squeeze(self.sym_func(env.current_raw_input_action))
