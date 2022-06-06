"""Linear Quadratic Regulator (LQR)

Example:
    run lqr on cartpole balance:

        python3 experiments/main.py --func test --tag lqr_pendulum --algo lqr --task cartpole

    run lqr on quadrotor stabilization:

        python3 experiments/main.py --func test --tag lqr_quad --algo lqr --task quadrotor --q_lqr 0.1

"""
from copy import deepcopy
import os
import numpy as np
import scipy.linalg
from termcolor import colored

from safe_control_gym.envs.env_wrappers.record_episode_statistics import RecordEpisodeStatistics
from safe_control_gym.utils.logging import ExperimentLogger
from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.lqr.lqr_utils import *
from safe_control_gym.envs.benchmark_env import Task


class LQR(BaseController):
    """Linear quadratic regulator.

    Attributes: 
        env (gym.Env): environment for the task.
        Q, R (np.array): cost weight matrix. 
        x_0, u_0 (np.array): equilibrium state & input.
        gain (np.array): input gain matrix.

    """

    def __init__(
            self,
            env_func,
            # Model args.
            q_lqr=[1],
            r_lqr=[1],
            discrete_dynamics=1,
            # Runner args.
            deque_size=10,
            eval_batch_size=1,
            # Shared/base args.
            output_dir="results/temp",
            verbose=False,
            model_step_chk=False,
            save_data=False,
            data_dir=None,
            plot_traj=False,
            plot_dir=None,
            save_plot=False,
            **kwargs):
        """Creates task and controller.

        Args:
            env_func (Callable): function to instantiate task/environment.
            q_lqr (list): diagonals of state cost weight.
            r_lqr (list): diagonals of input/action cost weight.
            discrete_dynamics (int): if to use discrete or continuous dynamics.
            deque_size (int): number of episodes to average over per evaluation statistic.
            eval_batch_size (int): number of episodes to run for evaluation.
            output_dir (str): output directory to write logs and results.

        """
        # All params/args.
        for k, v in locals().items():
            if k != "self" and k != "kwargs" and "__" not in k:
                self.__dict__[k] = v

        self.env_func = env_func
        self.deque_size = deque_size
        self.env = env_func()
        self.env = RecordEpisodeStatistics(self.env, deque_size)

        # Controller params.
        self.model = self.env.symbolic
        self.Q = get_cost_weight_matrix(self.q_lqr, self.model.nx)
        self.R = get_cost_weight_matrix(self.r_lqr, self.model.nu)
        self.env.set_cost_function_param(self.Q, self.R)

        # Linearize at operating point (equilibrium for stabilization).
        self.x_0, self.u_0 = self.env.X_GOAL, self.env.U_GOAL
        self.discrete_dynamics = discrete_dynamics

        if self.env.TASK == Task.STABILIZATION:
            self.gain = compute_lqr_gain(self.model, self.x_0, self.u_0,
                                         self.Q, self.R, self.discrete_dynamics)

        # Model step for debugging
        # self.env.reset()
        self.stepsize = self.model.dt

        # Check model step flag.
        self.model_step_chk = model_step_chk

        # Plot trajectory.
        self.plot_traj = plot_traj

        # Save plot.
        self.save_plot = save_plot

        # Plot output directory.
        self.plot_dir = plot_dir
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # Save data.
        self.save_data = save_data

        # Data output directory.
        self.data_dir = data_dir
        if self.data_dir:
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
        else:
            self.save_data = False

        # Logging.
        self.logger = ExperimentLogger(output_dir)
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

    def model_step(self):
        model = self.env.symbolic
        self.model_input = self.select_action(self.model_state)
        self.state_dot = model.fc_func(self.model_state, self.model_input)
        self.model_state = self.model_state + self.stepsize * self.state_dot

    def close(self):
        """Cleans up resources."""
        self.env.close()
        self.logger.close()

    def select_action(self, x):
        """Calculates control input u = -K x.

        Args:
            x (np.array): step-wise observation/input.

        Returns:
           np.array: step-wise control input/actino.

        """
        if self.env.TASK == Task.STABILIZATION:
            return -self.gain @ (x - self.x_0) + self.u_0
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.gain = compute_lqr_gain(self.model, self.x_0[self.k],
                                         self.u_0, self.Q, self.R,
                                         self.discrete_dynamics)
            return -self.gain @ (x - self.x_0[self.k]) + self.u_0
        else:
            print(colored("Incorrect task specified.", "red"))

    def compute_lqr_gain(self, x_0, u_0):
        # Linearization.
        df = self.model.df_func(x_0, u_0)
        A, B = df[0].toarray(), df[1].toarray()

        # Compute controller gain.
        if self.discrete_dynamics:
            # x[k+1] = A x[k] + B u[k]
            A, B = discretize_linear_system(A, B, self.model.dt)
            P = scipy.linalg.solve_discrete_are(A, B, self.Q, self.R)
            btp = np.dot(B.T, P)
            gain = np.dot(np.linalg.inv(self.R + np.dot(btp, B)),
                               np.dot(btp, A))
        else:
            # dx/dt = A x + B u
            P = scipy.linalg.solve_continuous_are(A, B, self.Q, self.R)
            gain = np.dot(np.linalg.inv(self.R), np.dot(B.T, P))

        return gain

    def run(self, render=False, logging=False, verbose=False):
        """Runs evaluation with current policy.

        Args:
            render (bool): if to render during the runs.
            logging (bool): if to log using logger during the runs.

        Returns:
            dict: evaluation results
            
        """
        ep_returns, ep_lengths, ep_results = [], [], []
        frames = []
        self.ep_counter = 0
        self.k = 0

        # Reseed for batch-wise consistency.
        obs, _ = self.env.reset()
        self.results_dict['obs'].append(obs)

        ep_seed = 1 #self.env.SEED

        while len(ep_returns) < self.eval_batch_size:
            # Current goal.
            if self.env.TASK == Task.STABILIZATION:
                current_goal = self.x_0
            elif self.env.TASK == Task.TRAJ_TRACKING:
                current_goal = self.x_0[self.k]

            # Select action.
            action = self.select_action(self.env.state)

            # Save initial condition.
            if self.k == 0:
                x_init = self.env.state
                if self.model_step_chk:
                    self.model_state = self.env.state

                # Initialize state and input stack.
                state_stack = self.env.state
                input_stack = action
                goal_stack = current_goal

                # Print initial state.
                print(colored("initial state (%d): " % ep_seed + get_arr_str(self.env.state), "green"))

            else:
                # Save state and input.
                state_stack = np.vstack((state_stack, self.env.state))
                input_stack = np.vstack((input_stack, action))
                goal_stack = np.vstack((goal_stack, current_goal))

            # Step forward.
            obs, reward, done, info = self.env.step(action)
            self.results_dict['obs'].append(obs)
            self.results_dict['reward'].append(reward)
            self.results_dict['done'].append(done)
            self.results_dict['info'].append(info)
            self.results_dict['action'].append(action)

            # Debug with analytical model.
            if self.model_step_chk:
                self.model_step()

            # Update step counter
            self.k += 1

            if verbose:
                if self.env.TASK == Task.TRAJ_TRACKING:
                    print("goal state: " + get_arr_str(self.x_0))
                print("state: " + get_arr_str(self.env.state))
                if self.model_step_chk:
                    print("model_state: " + get_arr_str(self.model_state))
                print("obs: " + get_arr_str(obs))
                print("action: " + get_arr_str(action) + "\n")

            if render:
                self.env.render()
                frames.append(self.env.render("rgb_array"))

            if done:
                # Push last state and input to stack.
                # Note: the last input is not used.
                state_stack = np.vstack((state_stack, self.env.state))
                input_stack = np.vstack((input_stack, action))
                goal_stack = np.vstack((goal_stack, current_goal))

                # Post analysis.
                if self.plot_traj or self.save_plot or self.save_data:
                    analysis_data = post_analysis(goal_stack, state_stack,
                                                  input_stack, self.env, 0,
                                                  self.ep_counter,
                                                  self.plot_traj,
                                                  self.save_plot,
                                                  self.save_data,
                                                  self.plot_dir, self.data_dir)
                    if self.ep_counter == 0:
                        ep_rmse = np.array([analysis_data["state_rmse_scalar"]])
                    else:
                        ep_rmse = np.vstack((ep_rmse, analysis_data["state_rmse_scalar"]))

                # Update iteration return and length lists.
                assert "episode" in info
                ep_returns.append(info["episode"]["r"])
                ep_lengths.append(info["episode"]["l"])
                ep_results.append(deepcopy(self.results_dict))
                self.reset_results_dict()

                print(colored("Test Run %d reward %.2f" % (self.ep_counter, ep_returns[-1]), "yellow"))
                print(colored("initial state: " + get_arr_str(x_init), "yellow"))
                if self.env.TASK == Task.STABILIZATION:
                    print(colored("final state: " + get_arr_str(self.env.state),  "yellow"))
                    print(colored("goal state: " + get_arr_str(self.x_0), "yellow"))
                print(colored("==========================\n", "yellow"))

                # Save reward
                if self.save_data:
                    np.savetxt(self.data_dir + "test%d_rewards.csv" % self.ep_counter, np.array([ep_returns[-1]]), delimiter=',', fmt='%.8f')

                self.ep_counter += 1
                ep_seed += 1
                self.k = 0
                self.env = RecordEpisodeStatistics(self.env, self.deque_size)
                obs, _ = self.env.reset()

        # Collect evaluation results.
        ep_lengths = np.asarray(ep_lengths)
        ep_returns = np.asarray(ep_returns)
        ep_results = np.asarray(ep_results)
        if logging:
            msg = "****** Evaluation ******\n"
            msg += "eval_ep_length {:.2f} +/- {:.2f} | eval_ep_return {:.3f} +/- {:.3f}\n".format(
                ep_lengths.mean(), ep_lengths.std(), ep_returns.mean(),
                ep_returns.std())
            self.logger.info(msg + "\n")

        if self.save_data:
            np.savetxt(self.data_dir + "all_test_mean_rmse.csv", ep_rmse, delimiter=',', fmt='%.8f')

        eval_data = {"ep_returns": ep_returns, "ep_lengths": ep_lengths, "ep_results": ep_results}
        if len(frames) > 0 and frames[0] is not None:
            eval_data["frames"] = frames
        return eval_data
