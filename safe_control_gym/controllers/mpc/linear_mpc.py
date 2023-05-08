'''Linear Model Predictive Control.

Based on:
    * https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/slides/LQR.pdf
    * https://pythonrobotics.readthedocs.io/en/latest/modules/path_tracking.html#mpc-modeling
    * https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/model_predictive_speed_and_steer_control/model_predictive_speed_and_steer_control.py
'''

from sys import platform
from copy import deepcopy

import numpy as np
import casadi as cs

from safe_control_gym.controllers.mpc.mpc import MPC
from safe_control_gym.controllers.mpc.mpc_utils import discretize_linear_system, compute_discrete_lqr_gain_from_cont_linear_system
from safe_control_gym.envs.benchmark_env import Task


class LinearMPC(MPC):
    '''Simple linear MPC.'''

    def __init__(
            self,
            env_func,
            horizon=5,
            q_mpc=[1],
            r_mpc=[1],
            warmstart=True,
            soft_constraints=False,
            terminate_run_on_done=True,
            constraint_tol: float = 1e-8,
            solver: str = 'sqpmethod',
            # runner args
            # shared/base args
            output_dir='results/temp',
            additional_constraints=None,
            **kwargs):
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
            solver (str): Specify which solver you wish to use (qrqp, qpoases, ipopt, sqpmethod)
            output_dir (str): output directory to write logs and results.
            additional_constraints (list): list of constraints.
        '''
        # Store all params/args.
        for k, v in locals().items():
            if k != 'self' and k != 'kwargs' and '__' not in k:
                self.__dict__[k] = v

        super().__init__(
            env_func,
            horizon=horizon,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            warmstart=warmstart,
            soft_constraints=soft_constraints,
            terminate_run_on_done=terminate_run_on_done,
            constraint_tol=constraint_tol,
            # prior_info=prior_info,
            output_dir=output_dir,
            additional_constraints=additional_constraints,
            **kwargs
        )

        # TODO: setup environment equilibrium
        #self.X_EQ = np.atleast_2d(self.env.X_GOAL)[0,:].T
        #self.U_EQ = np.atleast_2d(self.env.U_GOAL)[0,:]

        self.X_EQ = np.atleast_2d(self.model.X_EQ)[0, :].T
        self.U_EQ = np.atleast_2d(self.model.U_EQ)[0, :].T
        assert solver in ['qpoases', 'qrqp', 'sqpmethod', 'ipopt'], '[Error]. MPC Solver not supported.'
        self.solver = solver

    def set_dynamics_func(self):
        '''Updates symbolic dynamics with actual control frequency.'''
        # Original version, used in shooting.
        dfdxdfdu = self.model.df_func(x=self.X_EQ, u=self.U_EQ)
        dfdx = dfdxdfdu['dfdx'].toarray()
        dfdu = dfdxdfdu['dfdu'].toarray()
        delta_x = cs.MX.sym('delta_x', self.model.nx, 1)
        delta_u = cs.MX.sym('delta_u', self.model.nu, 1)
        #x_dot_lin_vec = dfdx @ delta_x + dfdu @ delta_u
        # self.linear_dynamics_func = cs.integrator(
        #    'linear_discrete_dynamics', self.model.integration_algo,
        #    {
        #        'x': delta_x,
        #        'p': delta_u,
        #        'ode': x_dot_lin_vec
        #    }, {'tf': self.dt}
        # )
        Ad, Bd = discretize_linear_system(dfdx, dfdu, self.dt, exact=True)
        x_dot_lin = Ad @ delta_x + Bd @ delta_u
        self.linear_dynamics_func = cs.Function('linear_discrete_dynamics',
                                                [delta_x, delta_u],
                                                [x_dot_lin],
                                                ['x0', 'p'],
                                                ['xf'])
        self.dfdx = dfdx
        self.dfdu = dfdu

    def compute_initial_guess(self, init_state, goal_states, x_lin, u_lin):
        '''Use LQR to get an initial guess of the '''
        dfdxdfdu = self.model.df_func(x=x_lin, u=u_lin)
        dfdx = dfdxdfdu['dfdx'].toarray()
        dfdu = dfdxdfdu['dfdu'].toarray()
        lqr_gain, _, _ = compute_discrete_lqr_gain_from_cont_linear_system(dfdx, dfdu, self.Q, self.R, self.dt)

        x_guess = np.zeros((self.model.nx, self.T + 1))
        u_guess = np.zeros((self.model.nu, self.T))
        x_guess[:, 0] = init_state

        for i in range(self.T):
            u = lqr_gain @ (x_guess[:, i] - goal_states[:, i]) + u_lin
            u_guess[:, i] = u
            x_guess[:, i + 1, None] = self.linear_dynamics_func(x0=x_guess[:, i], p=u)['xf'].toarray()

        return x_guess, u_guess

    def setup_optimizer(self):
        '''Sets up convex optimization problem.

        Including cost objective, variable bounds and dynamics constraints.
        '''
        nx, nu = self.model.nx, self.model.nu
        T = self.T
        # Define optimizer and variables.
        if self.solver in ['qrqp', 'qpoases']:
            opti = cs.Opti('conic')
        else:
            opti = cs.Opti()
        # States.
        x_var = opti.variable(nx, T + 1)
        # Inputs.
        u_var = opti.variable(nu, T)
        # Initial state.
        x_init = opti.parameter(nx, 1)
        # Reference (equilibrium point or trajectory, last step for terminal cost).
        x_ref = opti.parameter(nx, T + 1)
        # Add slack variables
        state_slack = opti.variable(len(self.state_constraints_sym))
        input_slack = opti.variable(len(self.input_constraints_sym))

        # cost (cumulative)
        cost = 0
        cost_func = self.model.loss
        for i in range(T):
            cost += cost_func(x=x_var[:, i] + self.X_EQ[:, None],
                              u=u_var[:, i] + self.U_EQ[:, None],
                              Xr=x_ref[:, i],
                              Ur=np.zeros((nu, 1)),
                              Q=self.Q,
                              R=self.R)['l']
        # Terminal cost.
        cost += cost_func(x=x_var[:, -1] + self.X_EQ[:, None],
                          u=np.zeros((nu, 1)) + self.U_EQ[:, None],
                          Xr=x_ref[:, -1],
                          Ur=np.zeros((nu, 1)),
                          Q=self.Q,
                          R=self.R)['l']
        for i in range(self.T):
            # Dynamics constraints.
            next_state = self.linear_dynamics_func(x0=x_var[:, i], p=u_var[:, i])['xf']
            opti.subject_to(x_var[:, i + 1] == next_state)

            # State and input constraints
            soft_con_coeff = 10
            for sc_i, state_constraint in enumerate(self.state_constraints_sym):
                if self.soft_constraints:
                    opti.subject_to(state_constraint(x_var[:, i] + self.X_EQ.T) <= state_slack[sc_i])
                    cost += soft_con_coeff * state_slack[sc_i]**2
                    opti.subject_to(state_slack[sc_i] >= 0)
                else:
                    opti.subject_to(state_constraint(x_var[:, i] + self.X_EQ.T) <= -self.constraint_tol)

            for ic_i, input_constraint in enumerate(self.input_constraints_sym):
                if self.soft_constraints:
                    opti.subject_to(input_constraint(u_var[:, i] + self.U_EQ.T) <= input_slack[ic_i])
                    cost += soft_con_coeff * input_slack[ic_i]**2
                    opti.subject_to(input_slack[ic_i] >= 0)
                else:
                    opti.subject_to(input_constraint(u_var[:, i] + self.U_EQ.T) <= -self.constraint_tol)

        # final state constraints
        for sc_i, state_constraint in enumerate(self.state_constraints_sym):
            if self.soft_constraints:
                opti.subject_to(state_constraint(x_var[:, -1] + self.X_EQ.T) <= state_slack[sc_i])
                cost += soft_con_coeff * state_slack[sc_i] ** 2
                opti.subject_to(state_slack[sc_i] >= 0)
            else:
                opti.subject_to(state_constraint(x_var[:, -1] + self.X_EQ.T) <= -self.constraint_tol)

        # initial condition constraints
        opti.subject_to(x_var[:, 0] == x_init)
        opti.minimize(cost)
        # create solver (IPOPT solver for now )
        opts = {'expand': True}
        if platform == 'linux':
            opts.update({'print_time': 1, 'print_header': 0})
            opti.solver(self.solver, opts)
        elif platform == 'darwin':
            opts.update({'ipopt.max_iter': 100})
            opti.solver('ipopt', opts)
        else:
            print('[ERROR]: CasADi solver tested on Linux and OSX only.')
            exit()
        self.opti_dict = {
            'opti': opti,
            'x_var': x_var,
            'u_var': u_var,
            'x_init': x_init,
            'x_ref': x_ref,
            'cost': cost
        }

    def select_action(self,
                      obs,
                      info=None
                      ):
        '''Solve nonlinear mpc problem to get next action.

        Args:
            obs (ndarray): Current state/observation.
            info (dict): Current info.

        Returns:
            action (ndarray): Input/action to the task/env.
        '''

        opti_dict = self.opti_dict
        opti = opti_dict['opti']
        x_var = opti_dict['x_var']
        u_var = opti_dict['u_var']
        x_init = opti_dict['x_init']
        x_ref = opti_dict['x_ref']

        # Assign the initial state.
        opti.set_value(x_init, obs - self.X_EQ)
        # Assign reference trajectory within horizon.
        goal_states = self.get_references()
        opti.set_value(x_ref, goal_states)
        if self.env.TASK == Task.TRAJ_TRACKING:
            self.traj_step += 1
        if self.warmstart and self.u_prev is not None and self.x_prev is not None:
            opti.set_initial(x_var, self.x_prev)
            opti.set_initial(u_var, self.u_prev)
        # Solve the optimization problem.
        try:
            sol = opti.solve()
            x_val, u_val = sol.value(x_var), sol.value(u_var)
            self.x_prev = x_val
            self.u_prev = u_val
            self.results_dict['horizon_states'].append(deepcopy(self.x_prev) + self.X_EQ[:, None])
            self.results_dict['horizon_inputs'].append(deepcopy(self.u_prev) + self.U_EQ[:, None])
        except RuntimeError as e:
            print(e)
            return_status = opti.return_status()
            if return_status == 'unknown':
                self.terminate_loop = True
                u_val = self.u_prev
                if u_val is None:
                    print('[WARN]: MPC Infeasible first step.')
                    u_val = np.zeros((1, self.model.nu))
            elif return_status == 'Maximum_Iterations_Exceeded':
                self.terminate_loop = True
                u_val = opti.debug.value(u_var)
            elif return_status == 'Search_Direction_Becomes_Too_Small':
                self.terminate_loop = True
                u_val = opti.debug.value(u_var)

        # take first one from solved action sequence
        if u_val.ndim > 1:
            action = u_val[:, 0]
        else:
            action = np.array([u_val[0]])
        action += self.U_EQ
        self.prev_action = action
        return action
