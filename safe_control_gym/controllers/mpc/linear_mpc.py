'''Linear Model Predictive Control.

Based on:
    * https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/slides/LQR.pdf
    * https://pythonrobotics.readthedocs.io/en/latest/modules/path_tracking.html#mpc-modeling
    * https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/model_predictive_speed_and_steer_control/model_predictive_speed_and_steer_control.py
'''

from copy import deepcopy
from sys import platform

import casadi as cs
import numpy as np
from termcolor import colored

from safe_control_gym.controllers.lqr.lqr_utils import discretize_linear_system
from safe_control_gym.controllers.mpc.mpc import MPC
from safe_control_gym.controllers.mpc.mpc_utils import compute_discrete_lqr_gain_from_cont_linear_system
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
            soft_penalty: float = 10000,
            constraint_tol: float = 1e-8,
            solver: str = 'sqpmethod',
            additional_constraints=None,
            use_lqr_gain_and_terminal_cost: bool = False,
            **kwargs  # Additional args from base_controller.py
    ):
        '''Creates task and controller.

        Args:
            env_func (Callable): Function to instantiate task/environment.
            horizon (int): MPC planning horizon.
            q_mpc (list): Diagonals of state cost weight.
            r_mpc (list): Diagonals of input/action cost weight.
            warmstart (bool): If to initialize from previous iteration.
            soft_constraints (bool): Formulate the constraints as soft constraints.
            constraint_tol (float): Tolerance to add the the constraint as sometimes solvers are not exact.
            solver (str): Specify which solver you wish to use (qrqp, qpoases, ipopt, sqpmethod)
            additional_constraints (list): List of constraints.
            use_lqr_gain_and_terminal_cost (bool): Use LQR ancillary gain and terminal cost in the MPC.
        '''
        super().__init__(
            env_func,
            horizon=horizon,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            warmstart=warmstart,
            soft_constraints=soft_constraints,
            soft_penalty=soft_penalty,
            constraint_tol=constraint_tol,
            additional_constraints=additional_constraints,
            use_lqr_gain_and_terminal_cost=use_lqr_gain_and_terminal_cost,
            **kwargs
        )

        # TODO: Setup environment equilibrium
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

        Ad, Bd = discretize_linear_system(dfdx, dfdu, self.dt, exact=True)
        x_dot_lin = Ad @ delta_x + Bd @ delta_u
        self.linear_dynamics_func = cs.Function('linear_discrete_dynamics',
                                                [delta_x, delta_u],
                                                [x_dot_lin],
                                                ['x0', 'p'],
                                                ['xf'])
        self.lqr_gain, _, _, self.P = compute_discrete_lqr_gain_from_cont_linear_system(dfdx,
                                                                                        dfdu,
                                                                                        self.Q,
                                                                                        self.R,
                                                                                        self.dt)
        self.dfdx = dfdx
        self.dfdu = dfdu

    def setup_optimizer(self, solver='qrqp'):
        '''Sets up convex optimization problem.
        Including cost objective, variable bounds and dynamics constraints.

        Args:
            solver (str): Solver to use for optimization. Options are 'qrqp', 'qpoases', 'sqpmethod', or 'ipopt'.
        '''
        nx, nu = self.model.nx, self.model.nu
        T = self.T
        # Define optimizer and variables.
        if solver in ['qrqp', 'qpoases']:
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

        # Cost (cumulative)
        cost = 0
        cost_func = self.model.loss
        for i in range(T):
            cost += cost_func(x=x_var[:, i] + self.X_EQ[:, None],
                              u=u_var[:, i] + self.U_EQ[:, None],
                              Xr=x_ref[:, i],
                              Ur=self.U_EQ,
                              Q=self.Q,
                              R=self.R)['l']
        # Terminal cost.
        cost += cost_func(x=x_var[:, -1] + self.X_EQ[:, None],
                          u=np.zeros((nu, 1)) + self.U_EQ[:, None],
                          Xr=x_ref[:, -1],
                          Ur=self.U_EQ,
                          Q=self.Q if not self.use_lqr_gain_and_terminal_cost else self.P,
                          R=self.R)['l']
        for i in range(self.T):
            # Dynamics constraints.
            next_state = self.linear_dynamics_func(x0=x_var[:, i], p=u_var[:, i])['xf']
            opti.subject_to(x_var[:, i + 1] == next_state)

            # State and input constraints
            soft_con_coeff = self.soft_penalty
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

        # Final state constraints
        for sc_i, state_constraint in enumerate(self.state_constraints_sym):
            if self.soft_constraints:
                opti.subject_to(state_constraint(x_var[:, -1] + self.X_EQ.T) <= state_slack[sc_i])
                cost += soft_con_coeff * state_slack[sc_i] ** 2
                opti.subject_to(state_slack[sc_i] >= 0)
            else:
                opti.subject_to(state_constraint(x_var[:, -1] + self.X_EQ.T) <= -self.constraint_tol)

        # Initial condition constraints
        opti.subject_to(x_var[:, 0] == x_init)
        opti.minimize(cost)
        # Create solver (IPOPT solver for now)
        opts = {'expand': True}
        if platform == 'linux':
            opts.update({'print_time': 1, 'print_header': 0})
            opti.solver(solver, opts)
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
            print(colored('Infeasible MPC Problem', 'red'))
            return_status = opti.return_status()
            print(colored(f'Optimization failed with status: {return_status}', 'red'))
            if return_status == 'unknown':
                if self.u_prev is None:
                    print(colored('[WARN]: MPC Infeasible first step.', 'yellow'))
                    u_val = np.zeros((self.model.nu, self.T))
                    x_val = np.zeros((self.model.nx, self.T + 1))
                else:
                    u_val = self.u_prev
                    x_val = self.x_prev
            elif return_status in ['Infeasible_Problem_Detected', 'Infeasible_Problem']:
                u_val = opti.debug.value(u_var)

        # Take first one from solved action sequence
        if u_val.ndim > 1:
            action = u_val[:, 0]
        else:
            action = np.array([u_val[0]])
        action += self.U_EQ
        if self.use_lqr_gain_and_terminal_cost:
            action += self.lqr_gain @ (obs - x_val[:, 0])
        self.prev_action = action
        return action
