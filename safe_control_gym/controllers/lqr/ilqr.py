"""Linear Quadratic Regulator (LQR)

[1] https://studywolf.wordpress.com/2016/02/03/the-iterative-linear-quadratic-regulator-method/
[2] https://arxiv.org/pdf/1708.09342.pdf

Example:

    run ilqr on cartpole balance:
    
        python3 experiments/main.py --func test --tag ilqr_pendulum --algo ilqr --task cartpole
        
    run ilqr on quadrotor stabilization:
    
        python3 experiments/main.py --func test --tag ilqr_quad --algo ilqr --task quadrotor --q_lqr 0.1
    
    Add the '--render' flag to produce a gif of the results

"""
import os
import numpy as np
from termcolor import colored
from matplotlib.ticker import FormatStrFormatter

from safe_control_gym.envs.env_wrappers.record_episode_statistics import RecordEpisodeStatistics, VecRecordEpisodeStatistics
from safe_control_gym.utils.logging import ExperimentLogger
from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.lqr.lqr_utils import *
from safe_control_gym.envs.benchmark_env import Cost, Task


class iLQR(BaseController):
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
            # model args
            q_lqr=[1],
            r_lqr=[1],
            discrete_dynamics=1,
            # runner args
            deque_size=10,
            eval_batch_size=1,
            # Task
            task: Task = Task.STABILIZATION,
            task_info=None,
            episode_len_sec=10,
            # iLQR args
            max_iterations=15,
            lamb_factor=10,
            lamb_max=1000,
            epsilon=0.01,
            # shared/base args
            output_dir="./results/temp/",
            verbose=True,
            random_init=True,
            ctrl_freq=240,
            pyb_freq=240,
            save_data=False,
            data_dir=None,
            plot_traj=False,
            plot_dir=None,
            save_plot=False,
            **kwargs):
        """Example of docstring on the __init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Args:
            param1 (str): Description of `param1`.
            param2 (:obj:`int`, optional): Description of `param2`. Multiple
                lines are supported.
            param3 (:obj:`list` of :obj:`str`): Description of `param3`.

        """
        # All params/args (lazy hack).
        for k, v in locals().items():
            if k != "self" and k != "kwargs" and "__" not in k:
                self.__dict__[k] = v

        # Task.
        self.env_func = env_func
        self.ctrl_freq = ctrl_freq
        self.pyb_freq = pyb_freq
        self.deque_size = deque_size
        self.task = Task(task)
        self.task_info = task_info
        self.episode_len_sec = episode_len_sec
        self.discrete_dynamics = discrete_dynamics

        # iLQR iterations.
        self.max_iterations = max_iterations

        # iLQR policy update parameters. See [1] for details.
        self.lamb_factor = lamb_factor  # Factor for scaling lambda
        self.lamb_max = lamb_max  # Maximum lambda
        self.epsilon = epsilon  # Tolerance for convergence

        # Stop iteration (to make sure that subsequent iteration number not
        # exceeding the first one)
        self.stop_iteration = False

        # Plot trajectory.
        self.plot_traj = plot_traj

        # Randomize initial state
        self.random_init = random_init

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
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Logging.
        self.logger = ExperimentLogger(output_dir)

        # Verbose.
        self.verbose = verbose

    def close(self):
        """Cleans up resources.

        """
        self.env.close()
        self.logger.close()

    def run_ilqr(self, render=False, logging=False):
        """Run iLQR to iteratively update policy for each time step k

        Args:
            render (bool): Flag to save frames for visualization.
            logging (bool): Flag to log results.

        Returns:
            ilqr_eval_results (dict): Dictionary containing the results from
            each iLQR iteration.
        """
        # Snapshot env state
        # state_dict = self.env.state_dict()

        # Initialize iteration logging variables.
        ite_returns, ite_lengths, ite_data, frames = [], [], {}, []

        # Initialize iteration and step counter.
        self.ite_counter = 0
        self.k = 0

        # Initialize step size
        self.lamb = 1.0

        # Set update unstable flag to False
        self.update_unstable = False

        # Initialize list of frames for each iteration
        frames_k = []

        # Loop through iLQR iterations
        while self.ite_counter < self.max_iterations:
            
            # Current goal.
            if self.task == Task.STABILIZATION:
                current_goal = self.x_0
            elif self.task == Task.TRAJ_TRACKING:
                current_goal = self.x_0[self.k]

            # Compute input.
            action = self.select_action(self.env.state, self.k)

            # Save rollout data.
            if self.k == 0:
                # Initialize state and input stack.
                state_stack = self.env.state
                input_stack = action
                goal_stack = current_goal

                # Print initial state.
                print(colored("initial state: " + get_arr_str(self.env.state), "green"))

                if self.ite_counter == 0:
                    self.init_state = self.env.state
            else:
                # Save state and input.
                state_stack = np.vstack((state_stack, self.env.state))
                input_stack = np.vstack((input_stack, action))
                goal_stack = np.vstack((goal_stack, current_goal))

            # Step forward.
            obs, reward, done, info = self.env.step(action)

            # Update step counter.
            self.k += 1
            # print("step", k, "done", done)

            # Print out.
            if self.verbose and self.k % 100 == 0:
                print(colored("episode: %d step: %d" % (self.ite_counter, self.k), "green"))
                print("state: " + get_arr_str(self.env.state))
                print("action: " + get_arr_str(self.env.state) + "\n")

            # Save frame for visualization.
            if render:
                self.env.render()
                frames_k.append(self.env.render("rgb_array"))

            # Save data and update policy if iteration is finished.
            if done:
                # Push last state and input to stack.
                # Last input is not really used.
                state_stack = np.vstack((state_stack, self.env.state))
                # input_stack = np.vstack((input_stack, action))
                # goal_stack = np.vstack((goal_stack, current_goal))
                
                # Add set of k frames to frames (for all episodes)
                frames.append(frames_k)
                frames_k = []

                # Update iteration return and length lists.
                assert "episode" in info
                ite_returns.append(info["episode"]["r"])
                ite_lengths.append(info["episode"]["l"])
                ite_data["ite%d_state" % self.ite_counter] = state_stack
                ite_data["ite%d_input" % self.ite_counter] = input_stack

                # Print iteration reward.
                print(colored("final state: " + get_arr_str(self.env.state), "green"))
                print(colored("iteration %d reward %.4f" %
                        (self.ite_counter, info["episode"]["r"]), "green"))
                print(colored("--------------------------", "green"))

                # Break if the first iteration is not successful
                if self.task == Task.STABILIZATION:
                    if self.ite_counter == 0 and not info["goal_reached"]:
                        print(colored("The initial policy might be unstable. "
                                + "Break from iLQR updates.", "red"))
                        break

                # Maximum episode length.
                self.num_steps = np.shape(input_stack)[0]
                self.episode_len_sec = self.num_steps * self.stepsize
                print(colored("Maximum episode length: %d steps!" % (self.num_steps), "blue"))
                print(np.shape(input_stack), np.shape(self.gains_fb))
                # import ipdb; ipdb.set_trace()

                # Check if cost is increased and update lambda correspondingly
                delta_reward = np.diff(ite_returns[-2:])
                if self.ite_counter == 0:

                    # Save best iteration.
                    print("Save iteration gains. Best iteration %d" % self.ite_counter)
                    self.best_iteration = self.ite_counter
                    self.input_ff_best = np.copy(self.input_ff)
                    self.gains_fb_best = np.copy(self.gains_fb)

                    # Update controller gains
                    self.update_policy(state_stack, input_stack)

                    # Initialize improved flag.
                    self.prev_ite_improved = False

                elif delta_reward < 0.0 or self.update_unstable:
                    # If cost is increased, increase lambda
                    self.lamb *= self.lamb_factor

                    # Reset feedforward term and controller gain to that from
                    # the previous iteration.
                    print("Cost increased by %.2f. " % -delta_reward
                        + "Set feedforward term and controller gain to that "
                        "from the previous iteration. "
                        "Increased lambda to %.2f." % self.lamb)
                    print("Current policy is from iteration %d." % self.best_iteration)
                    self.input_ff = np.copy(self.input_ff_best)
                    self.gains_fb = np.copy(self.gains_fb_best)

                    # Set improved flag to False.
                    self.prev_ite_improved = False

                    # Break if maximum lambda is reached.
                    if self.lamb > self.lamb_max:
                        print(colored("Maximum lambda reached.", "red"))
                        self.lamb = self.lamb_max

                    # Reset update_unstable flag to False.
                    self.update_unstable = False

                elif delta_reward >= 0.0:
                    # If cost is reduced, reduce lambda.
                    # Smoother convergence if not scaling down lambda.
                    # self.lamb /= self.lamb_factor

                    # Save feedforward term and gain and state and input stacks.
                    print("Save iteration gains. Best iteration %d" % self.ite_counter)
                    self.best_iteration = self.ite_counter
                    self.input_ff_best = np.copy(self.input_ff)
                    self.gains_fb_best = np.copy(self.gains_fb)

                    # Check consecutive reward increment (cost decrement).
                    if delta_reward < self.epsilon and self.prev_ite_improved:
                        # Cost converged.
                        print(colored("iLQR cost converged with a tolerance "
                                + "of %.2f." % self.epsilon, "yellow"))
                        break

                    # Set improved flag to True.
                    self.prev_ite_improved = True

                    # Update controller gains
                    self.update_policy(state_stack, input_stack)

                # Reset iteration and step counter.
                self.ite_counter += 1
                self.k = 0

                # Reset environment.
                print("Reset environment.")
                self.reset_env()

                # Post analysis.
                if self.plot_traj or self.save_plot or self.save_data:
                    analysis_data = post_analysis(goal_stack, state_stack,
                                                input_stack, self.env, 0,
                                                self.ep_counter,
                                                self.plot_traj,
                                                self.save_plot,
                                                self.save_data,
                                                self.plot_dir, self.data_dir)

        # Collect evaluation results.
        ite_lengths = np.asarray(ite_lengths)
        ite_returns = np.asarray(ite_returns)
        if logging:
            msg = "****** Evaluation ******\n"
            msg += "eval_ep_length {:.2f} +/- {:.2f} | " + \
                   "eval_ep_return {:.3f} +/- {:.3f}\n".format(
                ite_lengths.mean(), ite_lengths.std(), ite_returns.mean(),
                ite_returns.std())
            self.logger.info(msg + "\n")

        ilqr_eval_results = {
            "ite_returns": ite_returns,
            "ite_lengths": ite_lengths,
            "ite_data": ite_data
        }

        if len(frames) > 0:
            ilqr_eval_results["frames"] = frames

        return ilqr_eval_results

    def update_policy(self, state_stack, input_stack):
        """One-line description.

        Args:
            state_stack (np.array): States from previous rollout.
            input_stack (np.array): Inputs from previous rollout.

        """
        print(colored("UPDATE POLICY", "blue"))

        # Get symbolic loss function which also contains the necessary Jacobian
        # and Hessian of the loss w.r.t. state and input.
        loss = self.model.loss

        # Initialize backward pass.
        state_k = state_stack[-1]
        input_k = self.env.U_GOAL

        if self.task == Task.STABILIZATION:
            x_goal = self.x_0
        elif self.task == Task.TRAJ_TRACKING:
            x_goal = self.x_0[-1]
        loss_k = loss(x=state_k,
                      u=input_k,
                      Xr=x_goal,
                      Ur=self.env.U_GOAL,
                      Q=self.Q,
                      R=self.R)
        s = loss_k["l"].toarray()
        Sv = loss_k["l_x"].toarray().transpose()
        Sm = loss_k["l_xx"].toarray().transpose()

        # Backward pass.
        for k in reversed(range(self.num_steps)):
            print(k, self.num_steps, np.shape(state_stack), np.shape(input_stack), np.shape(self.gains_fb))
            # Get current operating point.
            state_k = state_stack[k]
            input_k = input_stack[k]

            # Linearized dynamics about (x_k, u_k).
            df_k = self.model.df_func(state_k, input_k)
            Ac_k, Bc_k = df_k[0].toarray(), df_k[1].toarray()
            Ad_k, Bd_k = discretize_linear_system(Ac_k, Bc_k, self.model.dt)

            # Get symbolic loss function that includes the necessary Jacobian
            # and Hessian of the loss w.r.t. state and input.
            if self.task == Task.STABILIZATION:
                x_goal = self.x_0
            elif self.task == Task.TRAJ_TRACKING:
                x_goal = self.x_0[k]
            loss_k = loss(x=state_k,
                          u=input_k,
                          Xr=x_goal,
                          Ur=self.env.U_GOAL,
                          Q=self.Q,
                          R=self.R)

            # Quadratic approximation of cost.
            q = loss_k["l"].toarray()  # l
            Qv = loss_k["l_x"].toarray().transpose()  # dl/dx
            Qm = loss_k["l_xx"].toarray().transpose()  # ddl/dxdx
            Rv = loss_k["l_u"].toarray().transpose()  # dl/du
            Rm = loss_k["l_uu"].toarray().transpose()  # ddl/dudu
            Pm = loss_k["l_xu"].toarray().transpose()  # ddl/dudx

            # Control dependent terms of cost function.
            g = Rv + Bd_k.transpose().dot(Sv)
            G = Pm + Bd_k.transpose().dot(Sm.dot(Ad_k))
            H = Rm + Bd_k.transpose().dot(Sm.dot(Bd_k))

            # Trick to make sure H is well-conditioned for inversion
            if not (np.isinf(np.sum(H)) or np.isnan(np.sum(H))):
                H = (H + H.transpose()) / 2
                H_eval, H_evec = np.linalg.eig(H)
                H_eval[H_eval < 0] = 0.0
                H_eval += self.lamb
                H_inv = np.dot(H_evec, np.dot(np.diag(1.0 / H_eval), H_evec.T))

                # Update controller gains.
                duff = -H_inv.dot(g)
                K = -H_inv.dot(G)

                # Update control input.
                input_ff_k = input_k + duff[:, 0] - K.dot(state_k)
                self.input_ff[:, k] = input_ff_k
                self.gains_fb[k] = K

                # Update s variables for time step k.
                Sm = Qm + Ad_k.transpose().dot(Sm.dot(Ad_k)) + \
                     K.transpose().dot(H.dot(K)) + \
                     K.transpose().dot(G) + G.transpose().dot(K)
                Sv = Qv + Ad_k.transpose().dot(Sv) + \
                     K.transpose().dot(H.dot(duff)) + K.transpose().dot(g) + \
                     G.transpose().dot(duff)
                s = q + s + 0.5 * duff.transpose().dot(H.dot(duff)) + \
                    duff.transpose().dot(g)
            else:
                self.update_unstable = True
                print(colored("Policy update unstable. Terminate update.", "red"))

    def select_action(self, x, k):
        """Control input u = -K x.

        Args:
            x (np.array): Current state of the system.
            k (int): Current time step.

        Returns:
            action (np.array): Action computed based on current policy.

        """
        if self.ite_counter == 0:
            # Compute gain for the first iteration.
            # action = -self.gain @ (x - self.x_0) + self.u_0
            if self.task == Task.STABILIZATION:
                gains_fb = -self.gain
                input_ff = self.gain @ self.x_0 + self.u_0

            elif self.task == Task.TRAJ_TRACKING:
                self.gain = compute_lqr_gain(self.model, self.x_0[k],
                                             self.u_0, self.Q, self.R,
                                             self.discrete_dynamics)
                gains_fb = -self.gain
                input_ff = self.gain @ self.x_0[k] + self.u_0
            else:
                print(colored("Incorrect task specified.", "red"))

            # Compute action
            action = gains_fb.dot(x) + input_ff

            # Save gains and feedforward term
            if self.k == 0:
                self.gains_fb = gains_fb.reshape(1, self.model.nu, self.model.nx)
                self.input_ff = input_ff.reshape(self.model.nu, 1)
            else:
                self.gains_fb = np.append(self.gains_fb, gains_fb.reshape(1, self.model.nu, self.model.nx), axis=0)
                self.input_ff = np.append(self.input_ff, input_ff.reshape(self.model.nu, 1), axis=1)
        else:
            print(k, self.gains_fb[k])
            action = self.gains_fb[k].dot(x) + self.input_ff[:, k]

        return action

    def init_env(self):
        self.env = self.env_func(randomized_init=self.random_init,
                            cost=Cost.QUADRATIC,
                            randomized_inertial_prop=False,
                            episode_len_sec=self.episode_len_sec,
                            task=self.task,
                            task_info=self.task_info,
                            ctrl_freq=self.ctrl_freq,
                            pyb_freq=self.pyb_freq
                            )
        self.env = RecordEpisodeStatistics(self.env, self.deque_size)

        # Controller params.
        self.model = self.env.symbolic
        self.Q = get_cost_weight_matrix(self.q_lqr, self.model.nx)
        self.R = get_cost_weight_matrix(self.r_lqr, self.model.nu)
        self.env.set_cost_function_param(self.Q, self.R)
        self.env.reset()

        # Linearize at operating point (equilibrium for stabilization).
        self.x_0, self.u_0 = self.env.X_GOAL, self.env.U_GOAL

        if self.task == Task.STABILIZATION:
            self.gain = compute_lqr_gain(self.model, self.x_0, self.u_0,
                                         self.Q, self.R, self.discrete_dynamics)

        # Control stepsize.
        self.stepsize = self.model.dt

    def reset_env(self):
        '''Reset environment between iLQR iterations.'''

        print(colored("Set maximum episode length to %.3f" % self.episode_len_sec, "blue"))
        self.env = self.env_func(init_state=self.init_state,
                                 randomized_init=False,
                                 cost=Cost.QUADRATIC,
                                 randomized_inertial_prop=False,
                                 episode_len_sec=self.episode_len_sec,
                                 task=self.task,
                                 task_info=self.task_info,
                                 ctrl_freq=self.ctrl_freq,
                                 pyb_freq=self.pyb_freq
                                 )
        self.env = RecordEpisodeStatistics(self.env, self.deque_size)

        # Controller params.
        self.model = self.env.symbolic
        self.Q = get_cost_weight_matrix(self.q_lqr, self.model.nx)
        self.R = get_cost_weight_matrix(self.r_lqr, self.model.nu)
        self.env.set_cost_function_param(self.Q, self.R)
        self.env.reset()

        # Linearize at operating point (equilibrium for stabilization).
        self.x_0, self.u_0 = self.env.X_GOAL, self.env.U_GOAL

    def run(self, n_episodes=1, render=False, logging=False, verbose=False, use_adv=False):
        """Runs evaluation with current policy.

        Args:
            render (bool): Flag to save frames for visualization.
            logging (bool): Flag to log results.

        Returns:
            eval_results (dict): Dictionary containing returns and data for each
            evaluation trial.

        """
        # Initialize logging variables.
        ep_returns, ep_lengths, ep_fulldata, frames = [], [], {}, []

        # Loop through episode.
        for self.ep_counter in range(self.eval_batch_size):
            # Initialize new environment for the test trial.
            self.init_env()

            # Run iLQR for the particular initial condition.
            ilqr_eval_results = self.run_ilqr(render=render, logging=logging)

            # Save the results from the last iteration for evaluation.
            ep_returns.append(ilqr_eval_results["ite_returns"][-1])
            ep_lengths.append(ilqr_eval_results["ite_lengths"][-1])
            ep_fulldata["run%d_data"
                        % self.ep_counter] = ilqr_eval_results["ite_data"]
            if "frames" in ilqr_eval_results:
                frames.extend(np.asarray(ilqr_eval_results["frames"][-1]))

            # Print episode reward.
            print(colored("Test Run %d reward %.4f" % (self.ep_counter, ep_returns[-1]), "yellow"))
            print(colored("==========================\n", "yellow"))

            # Save reward
            if self.save_data:
                np.savetxt(self.data_dir + "test%d_rewards.csv" % self.ep_counter, np.array([ep_returns[-1]]), delimiter=',', fmt='%.8f')

        # Collect evaluation results.
        ep_lengths = np.asarray(ep_lengths)
        ep_returns = np.asarray(ep_returns)

        # Log data.
        if logging:
            msg = "****** Evaluation ******\n"
            msg += "eval_ep_length {:.2f} +/- {:.2f} | " + \
                   "eval_ep_return {:.3f} +/- {:.3f}\n".format(
                ep_lengths.mean(), ep_lengths.std(), ep_returns.mean(),
                ep_returns.std())
            self.logger.info(msg + "\n")

        # Save evaluation results.
        # Note: To retrieve the state and input trajectories, use the following
        # eval_results["ep_fulldata"]["run#_data"]["ite#_state"]
        # eval_results["ep_fulldata"]["run#_data"]["ite#_input"]
        eval_results = {
            "ep_returns": ep_returns,
            "ep_lengths": ep_lengths,
            "ep_fulldata": ep_fulldata
        }

        # Save frames.
        if frames is not None and len(frames) > 0:
            eval_results["frames"] = frames

        return eval_results
