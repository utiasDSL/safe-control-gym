'''Model Predictive Control.'''

from copy import deepcopy

import casadi as cs
import numpy as np

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.mpc.mpc_utils import (compute_discrete_lqr_gain_from_cont_linear_system,
                                                        compute_state_rmse, get_cost_weight_matrix,
                                                        reset_constraints, rk_discrete)
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.envs.constraints import GENERAL_CONSTRAINTS, create_constraint_list
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel


class MPCAcados(BaseController):
    '''MPC with acados with full nonlinear model.'''

    def __init__(
            self,
            env_func,
            horizon: int = 5,
            q_mpc: list = [1],
            r_mpc: list = [1],
            warmstart: bool = True,
            soft_constraints: bool = False,
            terminate_run_on_done: bool = True,
            constraint_tol: float = 1e-6,
            # runner args
            # shared/base args
            output_dir: str = 'results/temp',
            additional_constraints: list = None,
            use_gpu: bool = False,
            seed: int = 0,
            **kwargs
    ):
        '''Creates task and controller.

        Args:
            env_func (Callable): function to instantiate task/environment.
            horizon (int): mpc planning horizon.
            q_mpc (list): diagonals of state cost weight.
            r_mpc (list): diagonals of input/action cost weight.
            warmstart (bool): if to initialize from previous iteration.
            soft_constraints (bool): Formulate the constraints as soft constraints.
            terminate_run_on_done (bool): Terminate the run when the environment returns done or not.
            constraint_tol (float): Tolerance to add the the constraint as sometimes solvers are not exact.
            output_dir (str): output directory to write logs and results.
            additional_constraints (list): List of additional constraints
            use_gpu (bool): False (use cpu) True (use cuda).
            seed (int): random seed.
        '''
        super().__init__(env_func=env_func, output_dir=output_dir,
                         use_gpu=use_gpu, seed=seed, **kwargs)
        for k, v in locals().items():
            if k != 'self' and k != 'kwargs' and '__' not in k:
                self.__dict__.update({k: v})

        # if prior_info is None:
        #    self.prior_info = {}
        # Task.
        self.env = env_func()
        if additional_constraints is not None:
            additional_ConstraintsList = create_constraint_list(additional_constraints,
                                                                GENERAL_CONSTRAINTS,
                                                                self.env)
            self.additional_constraints = additional_ConstraintsList.constraints
            self.constraints, self.state_constraints_sym, self.input_constraints_sym = reset_constraints(
                self.env.constraints.constraints + self.additional_constraints)
        else:
            self.constraints, self.state_constraints_sym, self.input_constraints_sym = reset_constraints(
                self.env.constraints.constraints)
            self.additional_constraints = []
        # Model parameters
        gym_model = self.get_prior(self.env)
        ocp = AcadosOcp()
        ocp.model.f_expl_expr = gym_model.x_dot
        ocp.model.x = gym_model.x_sym
        ocp.model.xdot = cs.MX.sym("xdot", ocp.model.x.shape)
        ocp.model.f_impl_expr = ocp.model.xdot - ocp.model.f_expl_expr
        ocp.model.u = gym_model.u_sym
        ocp.model.name = "quadrotor_3D"
        ocp.solver_options.N_horizon = horizon
        ocp.solver_options.tf = gym_model.dt * horizon
        Q_mat = get_cost_weight_matrix(self.q_mpc, gym_model.nx)
        R_mat = get_cost_weight_matrix(self.r_mpc, gym_model.nu)
        # path cost
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.model.cost_y_expr = cs.vertcat(ocp.model.x, ocp.model.u)
        ocp.cost.yref = np.zeros((gym_model.nx+gym_model.nu,))
        ocp.cost.W = cs.diagcat(Q_mat, R_mat).full()
        # terminal cost
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        ocp.cost.yref_e = np.zeros((gym_model.nx,))
        ocp.model.cost_y_expr_e = ocp.model.x
        ocp.cost.W_e = Q_mat
        #  set solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'  # FULL_CONDENSING_QPOASES
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'  # 'GAUSS_NEWTON', 'EXACT'
        ocp.solver_options.integrator_type = 'ERK'
        # ocp.solver_options.print_level = 1
        ocp.solver_options.nlp_solver_type = 'SQP'  # SQP_RTI, SQP
        ocp.solver_options.globalization = 'MERIT_BACKTRACKING'  # turns on globalization

        self.ocp = ocp
        self.model = gym_model
        self.dt = self.model.dt
        self.T = horizon
        self.Q = get_cost_weight_matrix(self.q_mpc, self.model.nx)
        self.R = get_cost_weight_matrix(self.r_mpc, self.model.nu)
        # self.prior_info = prior_info

        self.soft_constraints = soft_constraints
        self.warmstart = warmstart
        self.terminate_run_on_done = terminate_run_on_done

        # self.X_EQ = self.env.X_EQ
        # self.U_EQ = self.env.U_EQ
        # logging
        # self.logger = ExperimentLogger(output_dir)

    def add_constraints(self,
                        constraints
                        ):
        '''Add the constraints (from a list) to the system.

        Args:
            constraints (list): List of constraints controller is subject too.
        '''
        self.constraints, self.state_constraints_sym, self.input_constraints_sym = reset_constraints(
            constraints + self.constraints.constraints)

    def remove_constraints(self,
                           constraints
                           ):
        '''Remove constraints from the current constraint list.

        Args:
            constraints (list): list of constraints to be removed.
        '''
        old_constraints_list = self.constraints.constraints
        for constraint in constraints:
            assert constraint in self.constraints.constraints, \
                ValueError(
                    'This constraint is not in the current list of constraints')
            old_constraints_list.remove(constraint)
        self.constraints, self.state_constraints_sym, self.input_constraints_sym = reset_constraints(
            old_constraints_list)

    def close(self):
        '''Cleans up resources.'''
        self.env.close()

    def reset(self):
        '''Prepares for training or evaluation.'''
        # Setup reference input.
        if self.env.TASK == Task.STABILIZATION:
            self.mode = 'stabilization'
            self.x_goal = self.env.X_GOAL
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = 'tracking'
            self.traj = self.env.X_GOAL.T
            # Step along the reference.
            self.traj_step = 0
        # Dynamics model.
        self.set_dynamics_func()
        # CasADi optimizer.
        self.setup_optimizer()
        # Previously solved states & inputs, useful for warm start.
        self.x_prev = None
        self.u_prev = None

        self.setup_results_dict()

    def set_dynamics_func(self):
        '''Updates symbolic dynamics with actual control frequency.'''
        # self.dynamics_func = cs.integrator('fd', 'rk',
        #                                   {
        #                                    'x': self.model.x_sym,
        #                                    'p': self.model.u_sym,
        #                                    'ode': self.model.x_dot
        #                                    },
        #                                   {'tf': self.dt})
        self.dynamics_func = rk_discrete(self.model.fc_func,
                                         self.model.nx,
                                         self.model.nu,
                                         self.dt)

    def compute_initial_guess(self, init_state, goal_states, x_lin, u_lin):
        '''Use LQR to get an initial guess of the '''
        dfdxdfdu = self.model.df_func(x=x_lin, u=u_lin)
        dfdx = dfdxdfdu['dfdx'].toarray()
        dfdu = dfdxdfdu['dfdu'].toarray()
        lqr_gain, _, _ = compute_discrete_lqr_gain_from_cont_linear_system(
            dfdx, dfdu, self.Q, self.R, self.dt)

        x_guess = np.zeros((self.model.nx, self.T + 1))
        u_guess = np.zeros((self.model.nu, self.T))
        x_guess[:, 0] = init_state

        for i in range(self.T):
            u = lqr_gain @ (x_guess[:, i] - goal_states[:, i]) + u_lin
            u_guess[:, i] = u
            x_guess[:, i + 1,
                    None] = self.dynamics_func(x0=x_guess[:, i], p=u)['xf'].toarray()

        return x_guess, u_guess

    def setup_optimizer(self):
        '''Sets up nonlinear optimization problem.'''
        # TODO replace casadi opti with acados, add constraint

        self.ocp.constraints.x0 = np.zeros(self.ocp.model.x.shape)

        for sc_i, state_constraint in enumerate(self.state_constraints_sym):
            if state_constraint.__qualname__.split('.')[0] == "LinearConstraint":
                # state_constraint(x_var[:, i]) < -self.constraint_tol
                # state_constraint(x_var[:, -1]) <= -self.constraint_tol  # remember final state cns
                pass
            elif state_constraint.__qualname__.split('.')[0] == "BoundConstraint":
                pass
            else:
                raise NotImplementedError

        for ic_i, input_constraint in enumerate(self.input_constraints_sym):
            if input_constraint.__qualname__.split('.')[0] == "LinearConstraint":
                # input_constraint(u_var[:, i]) < -self.constraint_tol
                pass
            elif input_constraint.__qualname__.split('.')[0] == "BoundConstraint":
                pass
            else:
                raise NotImplementedError

        # Create solver (IPOPT solver in this version)
        self.ocp_solver = AcadosOcpSolver(self.ocp)

    def select_action(self,
                      obs,
                      info=None
                      ):
        '''Solves nonlinear mpc problem to get next action.

        Args:
            obs (ndarray): Current state/observation.
            info (dict): Current info

        Returns:
            action (ndarray): Input/action to the task/env.
        '''
        # set initial state
        self.ocp_solver.set(0, "lbx", obs)
        self.ocp_solver.set(0, "ubx", obs)
        # update yref
        goal_states = self.get_references()
        for i in range(self.T):
            self.ocp_solver.cost_set(i, "yref", np.concatenate(
                [goal_states[:, i], np.zeros(self.ocp.model.u.shape[0])]))
        self.ocp_solver.cost_set(self.T, "yref", goal_states[:, self.T])
        if self.mode == 'tracking':
            self.traj_step += 1

        # Solve the optimization problem.
        status = self.ocp_solver.solve()
        self.ocp_solver.print_statistics()
        if status != 0:
            print(f'acados returned status {status}.')
            # raise Exception(f'acados returned status {status}.')

        # get solution
        x_val = np.zeros((self.T+1, self.ocp.model.x.shape[0]))
        u_val = np.zeros((self.T, self.ocp.model.u.shape[0]))
        for i in range(self.T):
            x_val[i, :] = self.ocp_solver.get(i, "x")
            u_val[i, :] = self.ocp_solver.get(i, "u")
        x_val[self.T, :] = self.ocp_solver.get(self.T, "x")

        x_val = x_val.T
        u_val = u_val.T
        self.x_prev = x_val
        self.u_prev = u_val
        self.results_dict['horizon_states'].append(deepcopy(self.x_prev))
        self.results_dict['horizon_inputs'].append(deepcopy(self.u_prev))
        self.results_dict['goal_states'].append(deepcopy(goal_states))
        self.results_dict['t_wall'].append(
            self.ocp_solver.get_stats("time_tot"))
        # Take the first action from the solved action sequence.
        if u_val.shape[0] > 1:
            action = u_val[:, 0]
        else:
            action = np.array([u_val[0]])
            self.prev_action = action
        return action

    def get_references(self):
        '''Constructs reference states along mpc horizon.(nx, T+1).'''
        if self.env.TASK == Task.STABILIZATION:
            # Repeat goal state for horizon steps.
            goal_states = np.tile(
                self.env.X_GOAL.reshape(-1, 1), (1, self.T + 1))
        elif self.env.TASK == Task.TRAJ_TRACKING:
            # Slice trajectory for horizon steps, if not long enough, repeat last state.
            start = min(self.traj_step, self.traj.shape[-1])
            end = min(self.traj_step + self.T + 1, self.traj.shape[-1])
            remain = max(0, self.T + 1 - (end - start))
            goal_states = np.concatenate([
                self.traj[:, start:end],
                np.tile(self.traj[:, -1:], (1, remain))
            ], -1)
        else:
            raise Exception('Reference for this mode is not implemented.')
        return goal_states  # (nx, T+1).

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
        # obs, info = env.reset()
        obs = env.reset()
        print('Init State:')
        print(obs)
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
                MAX_STEPS = self.traj.shape[1]
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
            print(i, '-th step.')
            print(action)
            print(obs)
            print(reward)
            print(done)
            print(info)
            print()
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
            self.results_dict['reward'] = np.vstack(
                self.results_dict['reward'])
            self.results_dict['action'] = np.vstack(
                self.results_dict['action'])
            self.results_dict['full_traj_common_cost'] = common_metric
            self.results_dict['total_rmse_state_error'] = compute_state_rmse(
                self.results_dict['state'])
            self.results_dict['total_rmse_obs_error'] = compute_state_rmse(
                self.results_dict['obs'])
        except ValueError:
            raise Exception('[ERROR] mpc.run().py: MPC could not find a solution for the first step given the initial conditions. '
                            'Check to make sure initial conditions are feasible.')
        return deepcopy(self.results_dict)

    def reset_before_run(self, obs, info=None, env=None):
        '''Reinitialize just the controller before a new run.

        Args:
            obs (ndarray): The initial observation for the new run.
            info (dict): The first info of the new run.
            env (BenchmarkEnv): The environment to be used for the new run.
        '''
        self.reset()
