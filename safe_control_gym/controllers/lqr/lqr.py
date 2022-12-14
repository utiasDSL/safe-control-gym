'''Linear Quadratic Regulator (LQR) '''

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.lqr.lqr_utils import get_cost_weight_matrix, compute_lqr_gain
from safe_control_gym.envs.benchmark_env import Task


class LQR(BaseController):
    '''Linear quadratic regulator. '''

    def __init__(
            self,
            env_func,
            # Model args.
            q_lqr: list = None,
            r_lqr: list = None,
            discrete_dynamics: bool = True,
            **kwargs):
        '''Creates task and controller.

        Args:
            env_func (Callable): Function to instantiate task/environment.
            q_lqr (list): Diagonals of state cost weight.
            r_lqr (list): Diagonals of input/action cost weight.
            discrete_dynamics (bool): If to use discrete or continuous dynamics.
        '''

        super().__init__(env_func, **kwargs)

        self.env = env_func()
        # Controller params.
        self.model = self.get_prior(self.env)
        self.discrete_dynamics = discrete_dynamics
        self.Q = get_cost_weight_matrix(q_lqr, self.model.nx)
        self.R = get_cost_weight_matrix(r_lqr, self.model.nu)
        self.env.set_cost_function_param(self.Q, self.R)

        self.gain = compute_lqr_gain(self.model, self.model.X_EQ, self.model.U_EQ,
                                     self.Q, self.R, self.discrete_dynamics)

    def reset(self):
        '''Prepares for evaluation. '''
        self.env.reset()

    def close(self):
        '''Cleans up resources. '''
        self.env.close()

    def select_action(self, obs, info=None):
        '''Determine the action to take at the current timestep.

        Args:
            obs (ndarray): The observation at this timestep.
            info (dict): The info at this timestep.

        Returns:
            action (ndarray): The action chosen by the controller.
        '''

        step = self.extract_step(info)

        if self.env.TASK == Task.STABILIZATION:
            return -self.gain @ (obs - self.env.X_GOAL) + self.model.U_EQ
        elif self.env.TASK == Task.TRAJ_TRACKING:
            return -self.gain @ (obs - self.env.X_GOAL[step]) + self.model.U_EQ
