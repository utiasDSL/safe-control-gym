'''Model Predictive Control using Acados.'''

from copy import deepcopy

import casadi as cs
import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from safe_control_gym.controllers.mpc.mpc import MPC
from safe_control_gym.controllers.mpc.mpc_utils import set_acados_constraint_bound


class MPC_ACADOS(MPC):
    '''MPC with full nonlinear model solved with acados.'''

    def __init__(
            self,
            env_func,
            horizon: int = 5,
            q_mpc: list = [1],
            r_mpc: list = [1],
            soft_constraints: bool = False,
            soft_penalty: float = 10000,
            constraint_tol: float = 1e-6,
            additional_constraints: list = None,
            use_RTI: bool = False,
            use_lqr_gain_and_terminal_cost: bool = False,
            **kwargs
    ):
        '''Creates task and controller.

        Args:
            env_func (Callable): Function to instantiate task/environment.
            horizon (int): MPC planning horizon.
            q_mpc (list): Diagonals of state cost weight.
            r_mpc (list): Diagonals of input/action cost weight.
            soft_constraints (bool): Formulate the constraints as soft constraints.
            soft_penalty (float): Penalty added to acados formulation for soft constraints.
            constraint_tol (float): Tolerance to add to the constraint as sometimes solvers are not exact.
            additional_constraints (list): List of additional constraints.
            use_RTI (bool): Real-time iteration for acados.
            use_lqr_gain_and_terminal_cost (bool): Use LQR ancillary gain and terminal cost for the MPC.
        '''
        super().__init__(
            env_func,
            horizon=horizon,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            soft_constraints=soft_constraints,
            soft_penalty=soft_penalty,
            constraint_tol=constraint_tol,
            additional_constraints=additional_constraints,
            use_lqr_gain_and_terminal_cost=use_lqr_gain_and_terminal_cost,
            **kwargs
        )

        self.use_RTI = use_RTI

    def reset_before_run(self, obs=None, info=None, env=None):
        '''Reinitialize just the controller before a new run.

        Args:
            obs (ndarray): The initial observation for the new run.
            info (dict): The first info of the new run.
            env (BenchmarkEnv): The environment to be used for the new run.'''

        super().reset_before_run(obs, info, env)
        self.acados_ocp_solver.reset()

    def setup_acados_model(self) -> AcadosModel:
        '''Sets up symbolic model for acados.

        Returns:
            acados_model (AcadosModel): acados model object.
        '''
        acados_model = AcadosModel()
        acados_model.x = self.model.x_sym
        acados_model.u = self.model.u_sym
        acados_model.name = self.env.NAME

        # Dynamics model
        acados_model.f_expl_expr = self.model.fc_func(acados_model.x, acados_model.u)

        # Store meta information # NOTE: unit is missing
        acados_model.x_labels = self.env.STATE_LABELS
        acados_model.u_labels = self.env.ACTION_LABELS
        acados_model.t_label = 'time'

        return acados_model

    def setup_optimizer(self, *args):
        '''Sets up nonlinear optimization problem.'''
        nx, nu = self.model.nx, self.model.nu
        ny = nx + nu
        ny_e = nx

        # Create ocp object to formulate the OCP
        ocp = AcadosOcp()
        ocp.model = self.setup_acados_model()

        # Set dimensions
        ocp.dims.N = self.T  # Prediction horizon

        # Set cost (NOTE: safe-control-gym uses quadratic cost)
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W = scipy.linalg.block_diag(self.Q/self.dt, self.R/self.dt)
        ocp.cost.W_e = self.Q if not self.use_lqr_gain_and_terminal_cost else self.P
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx:(nx + nu), :nu] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)

        # Placeholder y_ref and y_ref_e (will be set in select_action)
        ocp.cost.yref = np.zeros((ny, ))
        ocp.cost.yref_e = np.zeros((ny_e, ))

        # Constraints
        # General constraint expressions
        for state_constraint in self.state_constraints_sym:
            grad_cns_x = cs.Function("grad_cns_x", [ocp.model.x], [cs.jacobian(state_constraint(ocp.model.x), ocp.model.x)])
            if np.all(grad_cns_x(np.ones(nx)).full() == np.vstack([-np.eye(nx), np.eye(nx)])):
                cns_x = cs.Function("cns_x", [ocp.model.x], [state_constraint(ocp.model.x)])
                bounds = cns_x(np.zeros(nx))
                ocp.constraints.lbx = bounds[:nx].full()
                ocp.constraints.ubx = -bounds[nx:].full()
                ocp.constraints.idxbx = np.array([i for i in range(nx)])
                ocp.constraints.lbx_e = bounds[:nx].full()
                ocp.constraints.ubx_e = -bounds[nx:].full()
                ocp.constraints.idxbx_e = np.array([i for i in range(nx)])
            else:
                raise ValueError("Constraint type not supported. Support only for bounded constraints expressed as A * x - b <= 0. Check constraints.py.")
        for input_constraint in self.input_constraints_sym:
            grad_cns_u = cs.Function("grad_cns_u", [ocp.model.u], [cs.jacobian(input_constraint(ocp.model.u), ocp.model.u)])
            if np.all(grad_cns_u(np.ones(nu)).full() == np.vstack([-np.eye(nu), np.eye(nu)])):
                cns_u = cs.Function("cns_u", [ocp.model.u], [input_constraint(ocp.model.u)])
                bounds = cns_u(np.zeros(nu))
                ocp.constraints.lbu = bounds[:nu].full()
                ocp.constraints.ubu = -bounds[nu:].full()
                ocp.constraints.idxbu = np.array([i for i in range(nu)])
            else:
                raise ValueError("Constraint type not supported. Support only for bounded constraints expressed as A * x - b <= 0. Check constraints.py.")

        # Slack costs for nonlinear constraints
        if self.soft_constraints:
            # Slack variables for all constraints
            ocp.constraints.Jsbu = np.eye(nu)
            ocp.constraints.Jsbx= np.eye(nu)
            ocp.constraints.Jsbx_e = np.eye(nx)
            # Slack penalty
            L2_pen = self.soft_penalty
            L1_pen = self.soft_penalty
            ocp.cost.Zl_0 = L2_pen * np.ones(nx)
            ocp.cost.Zu_0 = L2_pen * np.ones(nx)
            ocp.cost.zl_0 = L1_pen * np.ones(nx)
            ocp.cost.zu_0 = L1_pen * np.ones(nx)
            ocp.cost.Zu = L2_pen * np.ones(nx + nu)
            ocp.cost.Zl = L2_pen * np.ones(nx + nu)
            ocp.cost.zl = L1_pen * np.ones(nx + nu)
            ocp.cost.zu = L1_pen * np.ones(nx + nu)
            ocp.cost.Zl_e = L2_pen * np.ones(nx)
            ocp.cost.Zu_e = L2_pen * np.ones(nx)
            ocp.cost.zl_e = L1_pen * np.ones(nx)
            ocp.cost.zu_e = L1_pen * np.ones(nx)

        # Placeholder initial state constraint
        x_init = np.zeros((nx))
        ocp.constraints.x0 = x_init

        # Set up solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP' if not self.use_RTI else 'SQP_RTI'
        ocp.solver_options.nlp_solver_max_iter = 200 if not self.use_RTI else 1
        ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        ocp.solver_options.tf = self.T * self.dt  # Prediction horizon

        ocp.code_export_directory = self.output_dir + '/mpc_c_generated_code'

        self.acados_ocp_solver = AcadosOcpSolver(ocp)

    def processing_acados_constraints_expression(self,
                                                 ocp: AcadosOcp,
                                                 h0_expr: cs.MX,
                                                 h_expr: cs.MX,
                                                 he_expr: cs.MX,
                                                 ) -> AcadosOcp:
        '''Preprocess the constraints to be compatible with acados.

        Args:
            ocp (AcadosOcp): acados ocp object
            h0_expr (cs.MX expression): initial state constraints
            h_expr (cs.MX expression): state and input constraints
            he_expr (cs.MX expression): terminal state constraints

        Returns:
            ocp (AcadosOcp): acados ocp object with constraints set.
        '''

        ub = {'h': set_acados_constraint_bound(h_expr, 'ub', self.constraint_tol),
              'h0': set_acados_constraint_bound(h0_expr, 'ub', self.constraint_tol),
              'he': set_acados_constraint_bound(he_expr, 'ub', self.constraint_tol), }

        lb = {'h': set_acados_constraint_bound(h_expr, 'lb'),
              'h0': set_acados_constraint_bound(h0_expr, 'lb'),
              'he': set_acados_constraint_bound(he_expr, 'lb'), }

        # Make sure all the ub and lb are 1D numpy arrays
        # (see: https://discourse.acados.org/t/infeasible-qps-when-using-nonlinear-casadi-constraint-expressions/1595/5?u=mxche)
        for key in ub.keys():
            ub[key] = ub[key].flatten() if ub[key].ndim != 1 else ub[key]
            lb[key] = lb[key].flatten() if lb[key].ndim != 1 else lb[key]

        # Check ub and lb dimensions
        for key in ub.keys():
            assert ub[key].ndim == 1, f'ub[{key}] is not 1D numpy array'
            assert lb[key].ndim == 1, f'lb[{key}] is not 1D numpy array'
        assert ub['h'].shape == lb['h'].shape, 'h_ub and h_lb have different shapes'

        # Pass the constraints to the ocp object
        ocp.model.con_h_expr_0, ocp.model.con_h_expr, ocp.model.con_h_expr_e = \
            h0_expr, h_expr, he_expr
        ocp.dims.nh_0, ocp.dims.nh, ocp.dims.nh_e = \
            h0_expr.shape[0], h_expr.shape[0], he_expr.shape[0]
        # Assign constraints upper and lower bounds
        ocp.constraints.uh_0 = ub['h0']
        ocp.constraints.lh_0 = lb['h0']
        ocp.constraints.uh = ub['h']
        ocp.constraints.lh = lb['h']
        ocp.constraints.uh_e = ub['he']
        ocp.constraints.lh_e = lb['he']

        return ocp

    def select_action(self,
                      obs,
                      info=None
                      ):
        '''Solves nonlinear mpc problem to get next action.

        Args:
            obs (ndarray): Current state/observation.
            info (dict): Current info.

        Returns:
            action (ndarray): Input/action to the task/env.
        '''
        # Set initial condition (0-th state)
        self.acados_ocp_solver.set(0, 'lbx', obs)
        self.acados_ocp_solver.set(0, 'ubx', obs)

        # Set reference for the control horizon
        goal_states = self.get_references()
        if self.mode == 'tracking':
            self.traj_step += 1

        y_ref = np.concatenate((goal_states[:, :-1],
                                np.repeat(self.U_EQ.reshape(-1, 1), self.T, axis=1)), axis=0)
        for idx in range(self.T):
            self.acados_ocp_solver.set(idx, 'yref', y_ref[:, idx])
        y_ref_e = goal_states[:, -1]
        self.acados_ocp_solver.set(self.T, 'yref', y_ref_e)

        # Solve the optimization problem
        status = self.acados_ocp_solver.solve()
        if status not in [0, 2]:
            raise Exception(f'acados returned status {status}. Exiting.')
        if status == 2:
            print(f'acados returned status {status}. ')
        action = self.acados_ocp_solver.get(0, 'u')

        # Get the open-loop solution
        self.x_prev = np.zeros((self.model.nx, self.T + 1))
        self.u_prev = np.zeros((self.model.nu, self.T))
        for i in range(self.T):
            self.x_prev[:, i] = self.acados_ocp_solver.get(i, 'x')
            self.u_prev[:, i] = self.acados_ocp_solver.get(i, 'u')
        self.x_prev[:, -1] = self.acados_ocp_solver.get(self.T, 'x')

        self.results_dict['horizon_states'].append(deepcopy(self.x_prev))
        self.results_dict['horizon_inputs'].append(deepcopy(self.u_prev))
        self.results_dict['goal_states'].append(deepcopy(goal_states))

        if self.use_lqr_gain_and_terminal_cost:
            action += self.lqr_gain @ (obs - self.x_prev[:, 0])

        return action
