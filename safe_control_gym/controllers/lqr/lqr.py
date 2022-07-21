"""Linear Quadratic Regulator (LQR)

Example:
    run lqr on cartpole balance:

        python3 experiments/main.py --func test --tag lqr_pendulum --algo lqr --task cartpole

    run lqr on quadrotor stabilization:

        python3 experiments/main.py --func test --tag lqr_quad --algo lqr --task quadrotor --q_lqr 0.1

"""
import numpy as np
from munch import munchify

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.lqr.lqr_utils import get_cost_weight_matrix, compute_lqr_gain 
from safe_control_gym.envs.benchmark_env import Task


class LQR(BaseController):
    """Linear quadratic regulator.

    Attributes: 
        env (gym.Env): environment for the task.
        Q, R (np.array): cost weight matrix. 
        X_GOAL, U_GOAL (np.array): equilibrium state & input.
        gain (np.array): input gain matrix.

    """

    def __init__(
            self,
            env_func,
            # Model args.
            q_lqr=[1],
            r_lqr=[1],
            discrete_dynamics=True,
            **kwargs):
        """Creates task and controller.

        Args:
            env_func (Callable): function to instantiate task/environment.
            q_lqr (list): diagonals of state cost weight.
            r_lqr (list): diagonals of input/action cost weight.
            discrete_dynamics (bool): if to use discrete or continuous dynamics.
        """

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

        self.reset_results_dict()

    def reset(self):
        """Prepares for evaluation.

        """
        self.env.reset()
        self.reset_results_dict()

    def reset_results_dict(self):
        """
        Reset dictionary of experiment results
        """
        self.results_dict = { 'obs': [],
                              'reward': [],
                              'done': [],
                              'info': [],
                              'action': [],
        }

        if self.safety_filter:
            self.results_dict['corrections'] = []

    def close_results_dict(self):
        """Cleanup the results dict and munchify it.

        """
        self.results_dict['obs'] = np.vstack(self.results_dict['obs'])
        self.results_dict['reward'] = np.vstack(self.results_dict['reward'])
        self.results_dict['done'] = np.vstack(self.results_dict['done'])
        self.results_dict['info'] = np.vstack(self.results_dict['info'])
        self.results_dict['action'] = np.vstack(self.results_dict['action'])

        if self.safety_filter:
            self.results_dict['corrections'].append(0.0)
            self.results_dict['corrections'] = np.hstack(self.results_dict['corrections'])

        self.results_dict = munchify(self.results_dict)

    def close(self):
        """Cleans up resources."""
        self.env.close()

    def select_action(self, obs, step=0):
        """Calculates control input u = -K x.

        Args:
            obs (np.array): step-wise observation/input.
            step (int): the current iteration for trajectory tracking purposes

        Returns:
           np.array: step-wise control input/action.
        """

        if self.env.TASK == Task.STABILIZATION:
            return -self.gain @ (obs - self.env.X_GOAL) + self.env.U_GOAL
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.gain = compute_lqr_gain(self.model, self.env.X_GOAL[step],
                                         self.env.U_GOAL, self.Q, self.R,
                                         self.discrete_dynamics)
            return -self.gain @ (obs - self.env.X_GOAL[step]) + self.env.U_GOAL
        else:
            print("Incorrect task specified.")

    def run(self, env=None, max_steps=500):
        """Runs evaluation with current policy.

        Args:
            env (gym.Env): environment for the task.
            max_steps (int): maximum number of steps

        Returns:
            dict: evaluation results
            
        """
        if env is None:
            env = self.env

        # Reseed for batch-wise consistency.
        obs, _ = env.reset()
        self.results_dict['obs'].append(obs)

        for step in range(max_steps):
            # Select action.
            action = self.select_action(obs=obs, step=step)
            if self.safety_filter: 
                new_action, success = self.safety_filter.certify_action(current_state=obs, uncertified_action=action, iteration=step)
                if success:
                    action_diff = np.linalg.norm(new_action - action)
                    self.results_dict['corrections'].append(action_diff)
                    action = new_action
                else:
                    self.results_dict['corrections'].append(0.0)

            # Step forward.
            obs, reward, done, info = env.step(action)
            self.results_dict['obs'].append(obs)
            self.results_dict['reward'].append(reward)
            self.results_dict['done'].append(done)
            self.results_dict['info'].append(info)
            self.results_dict['action'].append(action)

            if done:
                print(f'SUCCESS: Reached goal on step {step}. Terminating...')
                break

        self.close_results_dict()

        return self.results_dict
