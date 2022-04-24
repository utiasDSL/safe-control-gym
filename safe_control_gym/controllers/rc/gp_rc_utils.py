"""Linear Quadratic Regulator (LQR) utilities

"""
import numpy as np
import scipy
from termcolor import colored
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from safe_control_gym.controllers.lqr.lqr import LQR
from safe_control_gym.envs.benchmark_env import Cost, Task
from safe_control_gym.envs.env_wrappers.record_episode_statistics import RecordEpisodeStatistics, VecRecordEpisodeStatistics

def get_arr_str(vector, str_format_arg=".2f"):
    str_format = "{:%s}" % str_format_arg
    vector_str = [str_format.format(i) for i in vector]
    vector_str_joined = " ".join(vector_str)
    return vector_str_joined


def compute_lqr_gain(model, x_0, u_0, Q, R, discrete_dynamics=True):
    # Linearization.
    df = model.df_func(x_0, u_0)
    A, B = df[0].toarray(), df[1].toarray()

    # Compute controller gain.
    if discrete_dynamics:
        # x[k+1] = A x[k] + B u[k]
        A, B = discretize_linear_system(A, B, model.dt)
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        btp = np.dot(B.T, P)
        gain = np.dot(np.linalg.inv(R + np.dot(btp, B)),
                      np.dot(btp, A))
    else:
        # dx/dt = A x + B u
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        gain = np.dot(np.linalg.inv(R), np.dot(B.T, P))

    return gain


def discretize_linear_system(A, B, dt, exact=False):
    """ discretization of a linear system 
    
    dx/dt = A x + B u 
    --> xd[k+1] = Ad xd[k] + Bd ud[k] where xd[k] = x(k*dt)

    Args:
        A: np.array, system transition matrix  
        B: np.array, input matrix 
        dt: scalar, step time interval 
        exact: bool, if to use exact discretization 

    Returns:
        discretized matrices Ad, Bd 

    """
    state_dim, input_dim = A.shape[1], B.shape[1]

    if exact:
        M = np.zeros((state_dim + input_dim, state_dim + input_dim))
        M[:state_dim, :state_dim] = A
        M[:state_dim, state_dim:] = B

        Md = scipy.linalg.expm(M * dt)
        Ad = Md[:state_dim, :state_dim]
        Bd = Md[:state_dim, state_dim:]
    else:
        I = np.eye(state_dim)
        Ad = I + A * dt
        Bd = B * dt

    return Ad, Bd


def get_cost_weight_matrix(weights, dim):
    """Gets weight matrix from input args.

    """
    if len(weights) == dim:
        W = np.diag(weights)
    elif len(weights) == 1:
        W = np.diag(weights * dim)
    else:
        raise Exception("Wrong dimension for cost weights.")
    return W

def post_analysis(goal_stack, state_stack, input_stack, env,
                  ite_counter, ep_counter, plot_traj, save_plot, save_data,
                  plot_dir, data_dir):
    # Get model
    model = env.symbolic
    stepsize = model.dt

    # Get times
    plot_length = np.min([np.shape(goal_stack)[0], np.shape(state_stack)[0]])
    times = np.linspace(0, stepsize * plot_length, plot_length)

    # Plot states
    fig, axs = plt.subplots(model.nx)
    if model.nx == 1:
        axs = [axs]
    for k in range(model.nx):
        axs[k].plot(times, state_stack.transpose()[k, 0:plot_length], label='actual')
        axs[k].plot(times, goal_stack.transpose()[k, 0:plot_length], color='r', label='desired')
        axs[k].set(ylabel=env.STATE_LABELS[k] + '\n[%s]' % env.STATE_UNITS[k])
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if not (k == model.nx - 1):
            axs[k].set_xticks([])
    axs[0].set_title('State Trajectories')
    axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc="lower right")
    axs[-1].set(xlabel='time (sec)')
    if save_plot:
        plt.savefig(plot_dir + "state_ite%d" % ite_counter)

    # Plot inputs
    _, axs = plt.subplots(model.nu)
    if model.nu == 1:
        axs = [axs]
    for k in range(model.nu):
        axs[k].plot(times, input_stack.transpose()[k, 0:plot_length])
        axs[k].set(ylabel='input %d' % k)
        axs[k].set(ylabel=env.ACTION_LABELS[k] + '\n[%s]' % env.ACTION_UNITS[k])
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs[0].set_title('Input Trajectories')
    axs[-1].set(xlabel='time (sec)')

    # Compute RMSE for each state
    state_error = state_stack.transpose()[:, 0:plot_length].transpose() -\
                  goal_stack.transpose()[:, 0:plot_length].transpose()

    # Check if state is an angle and wrap angle error to [-pi, pi]
    angle_state_index = [i for i, x in enumerate(env.STATE_UNITS) if x == "rad"]
    for k in angle_state_index:
        state_error[:, k] = wrap2pi_vec(state_error[:, k])

    state_rmse, state_rmse_scalar = compute_state_rmse(state_error)

    if plot_traj:
        plt.show()

    if save_plot:
        plt.savefig(plot_dir + "input_ite%d" % ite_counter)
        plt.close()

    if save_data:
        np.savetxt(data_dir + "test%d_times.csv" % ep_counter, times.transpose(), delimiter=',', fmt='%.8f')
        np.savetxt(data_dir + "test%d_states.csv" % ep_counter, state_stack.transpose()[:, 0:plot_length].transpose(), delimiter=',', fmt='%.8f')
        np.savetxt(data_dir + "test%d_states_des.csv" % ep_counter, goal_stack.transpose()[:, 0:plot_length].transpose(), delimiter=',', fmt='%.8f')
        np.savetxt(data_dir + "test%d_inputs.csv" % ep_counter, input_stack.transpose()[:, 0:plot_length].transpose(), delimiter=',', fmt='%.8f')
        np.savetxt(data_dir + "test%d_state_rmse.csv" % ep_counter, state_rmse, delimiter=',', fmt='%.8f')
        np.savetxt(data_dir + "test%d_state_rmse_scalar.csv" % ep_counter, np.array([state_rmse_scalar]), delimiter=',', fmt='%.8f')

    # Return analysis data
    analysis_data = {}
    analysis_data["state_rmse"] = state_rmse
    analysis_data["state_rmse_scalar"] = state_rmse_scalar

    return analysis_data


def wrap2pi_vec(angle_vec):
    for k in range(len(angle_vec)):
        angle_vec[k] = wrap2pi(angle_vec[k])
    return angle_vec


def wrap2pi(angle):
    while angle > np.pi:
        angle -= np.pi
    while angle <= -np.pi:
        angle += np.pi
    return angle


def compute_state_rmse(state_error):
    # Compute root-mean-square error
    mse = np.mean(state_error ** 2, axis=0)
    state_rmse = np.sqrt(mse)
    state_rmse_scalar = np.sqrt(np.sum(mse))
    # Print root-mean-square error
    print(colored("rmse by state: " + get_arr_str(state_rmse), "blue"))
    print(colored("scalarized rmse: %.2f" % state_rmse_scalar, "blue"))
    return state_rmse, state_rmse_scalar

class InitCtrl(LQR):
    """ LQR Controller used to gather datapoints for GP.
    """
    def __init__(self,
            env_func,
            # Model args.
            q_lqr=[1],
            r_lqr=[1],
            discrete_dynamics=1,
            # Runner args.
            deque_size=10,
            eval_batch_size=1,
            # Task
            task: Task = Task.STABILIZATION,
            task_info=None,
            episode_len_sec=10,
            **Kwargs):
        super().__init__(
            env_func=env_func,
            # Model args.
            q_lqr=q_lqr,
            r_lqr=r_lqr,
            discrete_dynamics=discrete_dynamics,
            # Runner args.
            deque_size=deque_size,
            eval_batch_size=eval_batch_size,
            # Task
            task = task,
            task_info = task_info,
            episode_len_sec= episode_len_sec,
            **Kwargs)

    def run(self, env=None, n_episodes=1, render=False, logging=False, verbose=False, use_adv=False):
            """Runs evaluation with current policy.

            Args:
                render (bool): if to render during the runs.
                logging (bool): if to log using logger during the runs.

            Returns:
                dict: evaluation results
                
            """
            if env is None:
                env = self.env
            else:
                env = RecordEpisodeStatistics(env, self.deque_size)

            ep_returns, ep_lengths = [], []
            frames = []
            self.ep_counter = 0
            self.k = 0

            # Reseed for batch-wise consistency.
            obs = env.reset()
            ep_seed = 1 #self.env.SEED

            while len(ep_returns) < self.eval_batch_size:
                # Current goal.
                if self.task == Task.STABILIZATION:
                    current_goal = self.x_0
                elif self.task == Task.TRAJ_TRACKING:
                    current_goal = self.x_0[self.k]

                # Select action.
                action = self.select_action(env.state)

                # Save initial condition.
                if self.k == 0:
                    x_init = env.state
                    if self.model_step_chk:
                        self.model_state = env.state

                    # Initialize state and input stack.
                    state_stack = env.state
                    input_stack = action
                    goal_stack = current_goal

                    # Print initial state.
                    print(colored("initial state (%d): " % ep_seed + get_arr_str(env.state), "green"))

                else:
                    # Save state and input.
                    state_stack = np.vstack((state_stack, env.state))
                    input_stack = np.vstack((input_stack, action))
                    goal_stack = np.vstack((goal_stack, current_goal))

                # Step forward.
                obs, reward, done, info = env.step(action)

                # Debug with analytical model.
                if self.model_step_chk:
                    self.model_step()

                # Update step counter
                self.k += 1

                if verbose:
                    if self.task == Task.TRAJ_TRACKING:
                        print("goal state: " + get_arr_str(self.x_0))
                    print("state: " + get_arr_str(env.state))
                    if self.model_step_chk:
                        print("model_state: " + get_arr_str(self.model_state))
                    print("obs: " + get_arr_str(obs))
                    print("action: " + get_arr_str(action) + "\n")

                if render:
                    env.render()
                    frames.append(env.render("rgb_array"))

                if done:
                    # Push last state and input to stack.
                    # Note: the last input is not used.
                    state_stack = np.vstack((state_stack, env.state))
                    input_stack = np.vstack((input_stack, action))
                    goal_stack = np.vstack((goal_stack, current_goal))

                    # Post analysis.
                    if self.plot_traj or self.save_plot or self.save_data:
                        analysis_data = post_analysis(goal_stack, state_stack,
                                                      input_stack, env, 0,
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

                    print(colored("Test Run %d reward %.2f" % (self.ep_counter, ep_returns[-1]), "yellow"))
                    print(colored("initial state: " + get_arr_str(x_init), "yellow"))
                    if self.task == Task.STABILIZATION:
                        print(colored("final state: " + get_arr_str(env.state),  "yellow"))
                        print(colored("goal state: " + get_arr_str(self.x_0), "yellow"))
                    print(colored("==========================\n", "yellow"))

                    # Save reward
                    if self.save_data:
                        np.savetxt(self.data_dir + "test%d_rewards.csv" % self.ep_counter, np.array([ep_returns[-1]]), delimiter=',', fmt='%.8f')

                    self.ep_counter += 1
                    ep_seed += 1
                    self.k = 0
                    self.env = self.env_func(cost=self.cost,
                                        randomized_init=self.random_init,
                                        seed=ep_seed,
                                        init_state_randomization_info=self.init_state_randomization_info,
                                        randomized_inertial_prop=False,
                                        episode_len_sec=self.episode_len_sec,
                                        task=self.task,
                                        task_info=self.task_info,
                                        ctrl_freq=self.ctrl_freq,
                                        pyb_freq=self.pyb_freq)
                    self.env = RecordEpisodeStatistics(env, self.deque_size)
                    obs = self.env.reset()

            # Collect evaluation results.
            ep_lengths = np.asarray(ep_lengths)
            ep_returns = np.asarray(ep_returns)
            if logging:
                msg = "****** Evaluation ******\n"
                msg += "eval_ep_length {:.2f} +/- {:.2f} | eval_ep_return {:.3f} +/- {:.3f}\n".format(
                    ep_lengths.mean(), ep_lengths.std(), ep_returns.mean(),
                    ep_returns.std())
                self.logger.info(msg + "\n")

            if self.save_data:
                np.savetxt(self.data_dir + "all_test_mean_rmse.csv", ep_rmse, delimiter=',', fmt='%.8f')

            eval_results = {"ep_returns": ep_returns, "ep_lengths": ep_lengths, "obs":state_stack, "action": input_stack}
            if len(frames) > 0 and frames[0] is not None:
                eval_results["frames"] = frames
            return eval_results