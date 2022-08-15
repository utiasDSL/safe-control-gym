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
            env_func (Callable): function to instantiate task/environment.
            q_lqr (list): diagonals of state cost weight.
            r_lqr (list): diagonals of input/action cost weight.
            discrete_dynamics (bool): if to use discrete or continuous dynamics.
        '''

        super().__init__(env_func, **kwargs)

        self.env = env_func()

        # Controller params.
        self.model = self.env.symbolic
        self.discrete_dynamics = discrete_dynamics
        self.Q = get_cost_weight_matrix(q_lqr, self.model.nx)
        self.R = get_cost_weight_matrix(r_lqr, self.model.nu)
        self.env.set_cost_function_param(self.Q, self.R)

        if self.env.TASK == Task.STABILIZATION:
            self.gain = compute_lqr_gain(self.model, self.env.X_GOAL, self.env.U_GOAL,
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
            obs (ndarray): the observation at this timestep.
            info (list): the info at this timestep.

        Returns:
            action (ndarray): the action chosen by the controller.
        '''

        step = self.extract_step(info)

        if self.env.TASK == Task.STABILIZATION:
            return -self.gain @ (obs - self.env.X_GOAL) + self.env.U_GOAL
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.gain = compute_lqr_gain(self.model, self.env.X_GOAL[step],
                                         self.env.U_GOAL, self.Q, self.R,
                                         self.discrete_dynamics)
            return -self.gain @ (obs - self.env.X_GOAL[step]) + self.env.U_GOAL

    def run(self, env=None, max_steps=500):
        '''Runs evaluation with current policy.

        Args:
            env (gym.Env): environment for the task.
            max_steps (int): maximum number of steps.
        '''

        if env is None:
            env = self.env

        # Reseed for batch-wise consistency.
        obs, info = env.reset()

        for step in range(max_steps):
            # Select action.
            action = self.select_action(obs=obs, info=info)
            # Step forward.
            obs, _, done, info = env.step(action)

            if done:
                print(f'SUCCESS: Reached goal on step {step}. Terminating...')
                break
