import numpy as np
import cvxpy as cp
import scipy.linalg
from termcolor import colored

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.lqr.lqr_utils import (compute_lqr_gain, discretize_linear_system,
                                                        get_cost_weight_matrix)
from safe_control_gym.controllers.mpc.mpc_utils import compute_state_rmse
from safe_control_gym.controllers.lqr.lqr import LQR
from safe_control_gym.envs.benchmark_env import Task

from copy import deepcopy

class iLQR_C(LQR):
    def __init__(self, env_func, 
                 q_lqr: list = None, 
                 r_lqr: list = None, 
                 discrete_dynamics: bool = True,
                 CM_search: bool = False, 
                 optimal_reference_path: str = None,
                 **kwargs):
        super().__init__(env_func, q_lqr, r_lqr, discrete_dynamics, **kwargs)

        self.env = env_func()
        # Controller params.
        self.model = self.get_prior(self.env)
        # print('self.model:', self.model)

        # self.discrete_dynamics = discrete_dynamics # not used since we are using continuous dynamics
        self.Q = get_cost_weight_matrix(q_lqr, self.model.nx)
        self.R = get_cost_weight_matrix(r_lqr, self.model.nu)
        self.env.set_cost_function_param(self.Q, self.R)

        self.dt = self.model.dt
        self.total_time = self.env.EPISODE_LEN_SEC
        self.N = int(self.total_time/self.dt) # number of timesteps
        self.T = 1 # MPC like horizon, only for get_references()
        
        self.optimal_reference_path = optimal_reference_path
        self.alpha_range = [0.1, 5.0]
        self.alpha_step = 0.1
        if CM_search:
            self.line_search_for_CM()
        self.x_ref = None
        self.u_ref = None

        
    def set_dynamics_func(self):
        # continuous-time dynamics
        self.dynamics_func = self.model.fc_func

    def get_cl_jacobian(self, x_lin, u_lin):
        '''Get the Jacobian of the closed-loop dynamics.'''
        # Linearize continuous-time dynamics
        df = self.model.df_func(x_lin, u_lin)
        A, B = df[0].toarray(), df[1].toarray()
        P = scipy.linalg.solve_continuous_are(A, B, self.Q, self.R)
        K = np.dot(np.linalg.inv(self.R), np.dot(B.T, P))
        J_cl = A - B @ K
        J_cl = np.array(J_cl)
        return J_cl

    def line_search_for_CM(self, d_bar=1.0):
        alpha = self.alpha_range[0]
        # total search steps
        Na = int((self.alpha_range[1] - self.alpha_range[0]) /self.alpha_step)+1
        print("========================================================")
        print("============= LINE SEARCH OF OPTIMAL ALPHA =============")
        print("========================================================")
        result_prev = np.Inf
        M_prev = None
        chi_prev = None
        for i in range(Na):
            result, M, chi, min_bound = self.compute_CM(alpha=alpha,
                                            d_bar=d_bar)
            print("Optimal value: Jcv =","{:.2f}".format(result),
                  "( alpha =","{:.3f}".format(alpha),
                    ", min_bound =","{:.3f}".format(min_bound),
                  ")")
            if result_prev <= result:
                alpha -= self.alpha_step
                self.result = result_prev
                self.M = M_prev
                self.chi = chi_prev
                break
            alpha += self.alpha_step
            # save the previous result
            result_prev = result
            M_prev = M
            chi_prev = chi
        self.alpha_opt = alpha
        print("Optimal contraction rate: alpha =","{:.3f}".format(alpha))
        print("Minimum bound: min_bound =","{:.3f}".format(min_bound))
        print("========================================================")
        print("=========== LINE SEARCH OF OPTIMAL ALPHA END ===========")
        print("========================================================\n\n")
        self.min_bound = d_bar / self.alpha_opt * np.sqrt(self.chi)


    def compute_CM(self, alpha, d_bar):
        grid_search = True
        N_grid = 10
        x_dot_search_range  = np.array([-1, 1])
        z_dot_search_range  = np.array([-1, 1])
        theta_search_range = np.array([-0.3, 0.3])
        x_dot_search_space = np.linspace(x_dot_search_range[0], x_dot_search_range[1], N_grid)
        z_dot_search_space = np.linspace(z_dot_search_range[0], z_dot_search_range[1], N_grid)
        theta_search_space = np.linspace(theta_search_range[0], theta_search_range[1], N_grid)

        if grid_search == False:          
            J_ref = [self.get_cl_jacobian(self.env.X_GOAL[i], self.model.U_EQ)
                    for i in range(self.N)]
        else:
            J_ref = []
            for i in range(N_grid):
                for j in range(N_grid):
                    for k in range(N_grid):
                        x_dot = x_dot_search_space[i]
                        z_dot = z_dot_search_space[j]
                        theta = theta_search_space[k]
                        x = np.array([0, x_dot, 0, z_dot, theta])
                        u = self.model.U_EQ
                        J_ref.append(self.get_cl_jacobian(x, u))
        nx = self.model.nx
        chi = cp.Variable(nonneg=True)
        W_tilde = cp.Variable((nx, nx), symmetric=True)
        objective = cp.Minimize(chi * d_bar / alpha)
        constraints = [chi*np.identity(nx) - W_tilde >> 0,
                    W_tilde - np.identity(nx) >> 0,]
        if grid_search == False:
            for i in range(self.N):
                constraints += [ - W_tilde @ J_ref[i].T - J_ref[i] @ W_tilde - 2 * alpha * W_tilde
                                >> 1e-6*np.identity(nx)]
        else:
            for i in range(N_grid**3):
                constraints += [ - W_tilde @ J_ref[i].T - J_ref[i] @ W_tilde - 2 * alpha * W_tilde
                                >> 1e-6*np.identity(nx)]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.MOSEK, warm_start=True)
        # print('result:', result)
        min_bound = d_bar / alpha * np.sqrt(chi.value)
        M = np.linalg.inv(W_tilde.value)
        return result, M, chi.value, min_bound


    def select_action(self, obs, info=None):
        '''Determine the action to take at the current timestep.

        Args:
            obs (ndarray): The observation at this timestep.
            info (dict): The info at this timestep.

        Returns:
            action (ndarray): The action chosen by the controller.
        '''
        # print('self.model.quad_mass:', self.model.quad_mass)
        # print('self.model.quad_Iyy:', self.model.quad_Iyy)
        # print('self.model:', self.model.__dir__())
        # print(self.env_func().__dir__())
        # exit()
        # step = self.extract_step(info)
        # self.goal_state = self.env.X_GOAL[self.traj_step]
        self.goal_state = self.get_references()

        if self.env.TASK == Task.STABILIZATION:
            # return -self.gain @ (obs - self.env.X_GOAL) + self.model.U_EQ
            raise NotImplementedError('iLQR not implemented for stabilization task.')
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.traj_step += 1
            # get the liearzation points
            # x_0 = self.env.X_GOAL[step]
            # x_0 = self.env.X_GOAL[self.traj_step]
            if self.optimal_reference_path is None:
                x_0 = self.goal_state[:, 0]
                # print('x_0:', x_0)
                u_0 = self.model.U_EQ
            else:
                x_0 = self.goal_state[:self.model.nx, 0]
                u_0 = self.goal_state[self.model.nx:, 0]
            # print('x_0:', x_0)
            # print('u_0:', u_0)
            # Linearize continuous-time dynamics
            df = self.model.df_func(x_0, u_0)
            A, B = df[0].toarray(), df[1].toarray()
            # print('A:\n', A)
            # print('B:\n', B)
            P = scipy.linalg.solve_continuous_are(A, B, self.Q, self.R)
            gain = np.dot(np.linalg.inv(self.R), np.dot(B.T, P))
            # control = -gain @ (obs - x_0) + u_0
            # action = -gain @ (obs - self.env.X_GOAL[step]) + self.model.U_EQ
            action = -gain @ (obs - x_0) + u_0

        # print('self.traj_step:', self.traj_step)
        action_bound_high = [ 0.4767, 0.4]
        action_bound_low = [ 0.079, -0.4]
        action = np.clip(action, action_bound_low, action_bound_high)
        # print('action:', action)
        return action
        pass
            # return -self.gain @ (obs - self.env.X_GOAL[step]) + self.model.U_EQ
    
    def get_references(self):
        '''Constructs reference states along mpc horizon.(nx, T+1).'''
        if self.env.TASK == Task.STABILIZATION:
            # Repeat goal state for horizon steps.
            goal_states = np.tile(self.env.X_GOAL.reshape(-1, 1), (1, self.T + 1))
        elif self.env.TASK == Task.TRAJ_TRACKING:
            # Slice trajectory for horizon steps, if not long enough, repeat last state.
            start = min(self.traj_step, self.traj.shape[-1])
            end = min(self.traj_step + self.T + 1, self.traj.shape[-1])
            remain = max(0, self.T + 1 - (end - start))
            # end = start + 1
            # remain = max(0, 1 - (end - start))
            goal_states = np.concatenate([
                self.traj[:, start:end],
                np.tile(self.traj[:, -1:], (1, remain))
            ], -1)
            # goal_states = self.traj[:, start:end]
        else:
            raise Exception('Reference for this mode is not implemented.')
        return goal_states  # (nx, T+1).

    def reset(self):
        '''Prepares for evaluation.'''
        self.env.reset()
        self.set_dynamics_func()
        if self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = 'tracking'
            if self.optimal_reference_path is None:
                self.traj = self.env.X_GOAL.T
            else:
                print('Loading optimal reference path:', self.optimal_reference_path)
                self.traj_load = np.load(self.optimal_reference_path, allow_pickle=True)
                x_ref = self.traj_load['trajs_data']['obs'][0].T
                u_ref = self.traj_load['trajs_data']['action'][0].T
                self.traj = np.concatenate([x_ref[:, :-1], u_ref], axis=0)
                self.x_ref = x_ref
                self.u_ref = u_ref
                print('Optimal reference loaded.')
            self.traj_step = 0

    def close(self):
        '''Cleans up resources.'''
        self.env.close()

    def run(self,
            env=None,
            render=False,
            logging=False,
            max_steps=None,
            terminate_run_on_done=None
            ):
        '''Runs evaluation with current policy.

        Args:
            render (bool): if to do real-time rendering.
            logging (bool): if to log on terminal.

        Returns:
            dict: evaluation statisitcs, rendered frames.
        '''
        if env is None:
            env = self.env
        if terminate_run_on_done is None:
            terminate_run_on_done = self.terminate_run_on_done

        self.x_prev = None
        self.u_prev = None
        if not env.initial_reset:
            env.set_cost_function_param(self.Q, self.R)
        obs, info = env.reset()
        # obs = env.reset()
        # print('Init State:')
        # print(obs)
        ep_returns, ep_lengths = [], []
        frames = []
        self.setup_results_dict()
        self.results_dict['obs'].append(obs)
        self.results_dict['state'].append(env.state)
        i = 0
        if env.TASK == Task.STABILIZATION:
            if max_steps is None:
                MAX_STEPS = int(env.CTRL_FREQ * env.EPISODE_LEN_SEC)
            else:
                MAX_STEPS = max_steps
        elif env.TASK == Task.TRAJ_TRACKING:
            if max_steps is None:
                MAX_STEPS = self.traj.shape[1] - 1 #TODO: why I have to subtract 1?
                # print('MAX_STEPS:', MAX_STEPS)
            else:
                MAX_STEPS = max_steps
        else:
            raise Exception('Undefined Task')
        self.terminate_loop = False
        done = False
        common_metric = 0
        while not (done and terminate_run_on_done) and i < MAX_STEPS and not (self.terminate_loop):
            action = self.select_action(obs)
            if self.terminate_loop:
                print('Infeasible MPC Problem')
                break
            # Repeat input for more efficient control.
            obs, reward, done, info = env.step(action)
            self.results_dict['obs'].append(obs)
            self.results_dict['reward'].append(reward)
            self.results_dict['done'].append(done)
            self.results_dict['info'].append(info)
            self.results_dict['action'].append(action)
            self.results_dict['state'].append(env.state)
            self.results_dict['state_mse'].append(info['mse'])
            # self.results_dict['state_error'].append(env.state - env.X_GOAL[i,:])

            common_metric += info['mse']
            # print(i, '-th step.')
            # print('action:', action)
            # print('obs', obs)
            if render:
                env.render()
                frames.append(env.render('rgb_array'))
            i += 1
        # Collect evaluation results.
        ep_lengths = np.asarray(ep_lengths)
        ep_returns = np.asarray(ep_returns)
        if logging:
            msg = '****** Evaluation ******\n'
            msg += 'eval_ep_length {:.2f} +/- {:.2f} | eval_ep_return {:.3f} +/- {:.3f}\n'.format(
                ep_lengths.mean(), ep_lengths.std(), ep_returns.mean(),
                ep_returns.std())
        if len(frames) != 0:
            self.results_dict['frames'] = frames
        self.results_dict['obs'] = np.vstack(self.results_dict['obs'])
        self.results_dict['state'] = np.vstack(self.results_dict['state'])
        try:
            self.results_dict['reward'] = np.vstack(self.results_dict['reward'])
            self.results_dict['action'] = np.vstack(self.results_dict['action'])
            self.results_dict['full_traj_common_cost'] = common_metric
            self.results_dict['total_rmse_state_error'] = compute_state_rmse(self.results_dict['state'])
            self.results_dict['total_rmse_obs_error'] = compute_state_rmse(self.results_dict['obs'])
        except ValueError:
            raise Exception('[ERROR] mpc.run().py: MPC could not find a solution for the first step given the initial conditions. '
                            'Check to make sure initial conditions are feasible.')
        return deepcopy(self.results_dict)

    def setup_results_dict(self):
        '''Setup the results dictionary to store run information.'''
        self.results_dict = {'obs': [],
                             'reward': [],
                             'done': [],
                             'info': [],
                             'action': [],
                             'horizon_inputs': [],
                             'horizon_states': [],
                             'goal_states': [],
                             'frames': [],
                             'state_mse': [],
                             'common_cost': [],
                             'state': [],
                             'state_error': [],
                             't_wall': []
                             }