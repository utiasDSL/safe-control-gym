'''Model Predictive Control using Acados.'''

from copy import deepcopy

import casadi as cs
import numpy as np
import scipy
from termcolor import colored

from safe_control_gym.controllers.mpc.mpc import MPC
from safe_control_gym.controllers.mpc.mpc_utils import set_acados_constraint_bound
from safe_control_gym.utils.utils import timing

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
except ImportError as e:
    print(colored(f'Error: {e}', 'red'))
    print(colored('acados not installed, cannot use acados-based controller. Exiting.', 'red'))
    print(colored('- To build and install acados, follow the instructions at https://docs.acados.org/installation/index.html', 'yellow'))
    print(colored('- To set up the acados python interface, follow the instructions at https://docs.acados.org/python_interface/index.html', 'yellow'))
    print()
    exit()


class MPC_ACADOS(MPC):
    '''MPC with full nonlinear model.'''

    def __init__(
            self,
            env_func,
            horizon: int = 5,
            q_mpc: list = [1],
            r_mpc: list = [1],
            warmstart: bool = True,
            soft_constraints: bool = False,
            soft_penalty: float = 10000,
            terminate_run_on_done: bool = True,
            constraint_tol: float = 1e-6,
            # runner args
            # shared/base args
            output_dir: str = 'results/temp',
            additional_constraints: list = None,
            use_gpu: bool = False,
            seed: int = 0,
            use_RTI: bool = False,
            use_lqr_gain_and_terminal_cost: bool = False,
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
            use_RTI (bool): Real-time iteration for acados.
            use_lqr_gain_and_terminal_cost (bool): Use LQR ancillary gain and terminal cost for the MPC.
        '''
        for k, v in locals().items():
            if k != 'self' and k != 'kwargs' and '__' not in k:
                self.__dict__.update({k: v})
        super().__init__(
            env_func,
            horizon=horizon,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            warmstart=warmstart,
            soft_constraints=soft_constraints,
            soft_penalty=soft_penalty,
            terminate_run_on_done=terminate_run_on_done,
            constraint_tol=constraint_tol,
            output_dir=output_dir,
            additional_constraints=additional_constraints,
            compute_initial_guess_method='ipopt',  # use ipopt initial guess by default
            use_lqr_gain_and_terminal_cost=use_lqr_gain_and_terminal_cost,
            use_gpu=use_gpu,
            seed=seed,
            **kwargs
        )

        self.x_guess = None
        self.u_guess = None
        # acados settings
        self.use_RTI = use_RTI

    @timing
    def reset(self):
        '''Prepares for training or evaluation.'''
        print(colored('Resetting MPC', 'green'))
        super().reset()
        self.acados_model = None
        self.ocp = None
        self.acados_ocp_solver = None
        # Dynamics model.
        self.setup_acados_model()
        # Acados optimizer.
        self.setup_acados_optimizer()
        self.acados_ocp_solver = AcadosOcpSolver(self.ocp)

    def setup_acados_model(self) -> AcadosModel:
        '''Sets up symbolic model for acados.'''

        acados_model = AcadosModel()
        acados_model.x = self.model.x_sym
        acados_model.u = self.model.u_sym
        acados_model.name = self.env.NAME

        # set up rk4 (acados need symbolic expression of dynamics, not function)
        k1 = self.model.fc_func(acados_model.x, acados_model.u)
        k2 = self.model.fc_func(acados_model.x + self.dt / 2 * k1, acados_model.u)
        k3 = self.model.fc_func(acados_model.x + self.dt / 2 * k2, acados_model.u)
        k4 = self.model.fc_func(acados_model.x + self.dt * k3, acados_model.u)
        f_disc = acados_model.x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        acados_model.disc_dyn_expr = f_disc
        '''
        Other options to set up the model:
        f_expl = self.model.x_dot (explicit continuous-time dynamics)
        f_impl = self.model.x_dot_acados - f_expl (implicit continuous-time dynamics)
        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        '''
        # store meta information # NOTE: unit is missing
        acados_model.x_labels = self.env.STATE_LABELS
        acados_model.u_labels = self.env.ACTION_LABELS
        acados_model.t_label = 'time'

        self.acados_model = acados_model

    @timing
    def compute_initial_guess(self, init_state, goal_states=None):
        '''Use IPOPT to get an initial guess of the solution.'''
        x_val, u_val = super().compute_initial_guess(init_state, goal_states)
        self.x_guess = x_val
        self.u_guess = u_val

    def setup_acados_optimizer(self):
        '''Sets up nonlinear optimization problem.'''
        nx, nu = self.model.nx, self.model.nu
        ny = nx + nu
        ny_e = nx

        # create ocp object to formulate the OCP
        ocp = AcadosOcp()
        ocp.model = self.acados_model

        # set dimensions
        ocp.dims.N = self.T  # prediction horizon

        # set cost (NOTE: safe-control-gym uses quadratic cost)
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W = scipy.linalg.block_diag(self.Q, self.R)
        ocp.cost.W_e = self.Q if not self.use_lqr_gain_and_terminal_cost else self.P
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx:(nx + nu), :nu] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)
        # placeholder y_ref and y_ref_e (will be set in select_action)
        ocp.cost.yref = np.zeros((ny, ))
        ocp.cost.yref_e = np.zeros((ny_e, ))

        # Constraints
        # general constraint expressions
        state_constraint_expr_list = []
        input_constraint_expr_list = []
        for sc_i, state_constraint in enumerate(self.state_constraints_sym):
            state_constraint_expr_list.append(state_constraint(ocp.model.x))
        for ic_i, input_constraint in enumerate(self.input_constraints_sym):
            input_constraint_expr_list.append(input_constraint(ocp.model.u))

        h_expr_list = state_constraint_expr_list + input_constraint_expr_list
        h_expr = cs.vertcat(*h_expr_list)
        h0_expr = cs.vertcat(*h_expr_list)
        he_expr = cs.vertcat(*state_constraint_expr_list)  # terminal constraints are only state constraints
        # pass the constraints to the ocp object
        ocp = self.processing_acados_constraints_expression(ocp, h0_expr, h_expr, he_expr)

        # slack costs for nonlinear constraints
        if self.soft_constraints:
            # slack variables for all constraints
            ocp.constraints.Jsh_0 = np.eye(h0_expr.shape[0])
            ocp.constraints.Jsh = np.eye(h_expr.shape[0])
            ocp.constraints.Jsh_e = np.eye(he_expr.shape[0])
            # slack penalty
            L2_pen = self.soft_penalty
            L1_pen = self.soft_penalty
            ocp.cost.Zl_0 = L2_pen * np.ones(h0_expr.shape[0])
            ocp.cost.Zu_0 = L2_pen * np.ones(h0_expr.shape[0])
            ocp.cost.zl_0 = L1_pen * np.ones(h0_expr.shape[0])
            ocp.cost.zu_0 = L1_pen * np.ones(h0_expr.shape[0])
            ocp.cost.Zu = L2_pen * np.ones(h_expr.shape[0])
            ocp.cost.Zl = L2_pen * np.ones(h_expr.shape[0])
            ocp.cost.zl = L1_pen * np.ones(h_expr.shape[0])
            ocp.cost.zu = L1_pen * np.ones(h_expr.shape[0])
            ocp.cost.Zl_e = L2_pen * np.ones(he_expr.shape[0])
            ocp.cost.Zu_e = L2_pen * np.ones(he_expr.shape[0])
            ocp.cost.zl_e = L1_pen * np.ones(he_expr.shape[0])
            ocp.cost.zu_e = L1_pen * np.ones(he_expr.shape[0])

        # placeholder initial state constraint
        x_init = np.zeros((nx))
        ocp.constraints.x0 = x_init

        # set up solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.nlp_solver_type = 'SQP' if not self.use_RTI else 'SQP_RTI'
        ocp.solver_options.nlp_solver_max_iter = 25 if not self.use_RTI else 1
        # ocp.solver_options.globalization = 'FUNNEL_L1PEN_LINESEARCH' if not self.use_RTI else 'MERIT_BACKTRACKING'
        # ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        ocp.solver_options.tf = self.T * self.dt  # prediction horizon

        # c code generation
        # NOTE: when using GP-MPC, a separated directory is needed;
        # otherwise, Acados solver can read the wrong c code
        ocp.code_export_directory = self.output_dir + '/mpc_c_generated_code'

        self.ocp = ocp

    def processing_acados_constraints_expression(self, ocp: AcadosOcp, h0_expr, h_expr, he_expr) -> AcadosOcp:
        '''Preprocess the constraints to be compatible with acados.
            Args:
                ocp (AcadosOcp): acados ocp object
                h0_expr (casadi expression): initial state constraints
                h_expr (casadi expression): state and input constraints
                he_expr (casadi expression): terminal state constraints
            Returns:
                ocp (AcadosOcp): acados ocp object with constraints set.

        An alternative way to set the constraints is to use bounded constraints of acados:
        # bounded input constraints
        idxbu = np.where(np.sum(self.env.constraints.input_constraints[0].constraint_filter, axis=0) != 0)[0]
        ocp.constraints.Jbu = np.eye(nu)
        ocp.constraints.lbu = self.env.constraints.input_constraints[0].lower_bounds
        ocp.constraints.ubu = self.env.constraints.input_constraints[0].upper_bounds
        ocp.constraints.idxbu = idxbu # active constraints dimension
        '''

        ub = {'h': set_acados_constraint_bound(h_expr, 'ub', self.constraint_tol),
              'h0': set_acados_constraint_bound(h0_expr, 'ub', self.constraint_tol),
              'he': set_acados_constraint_bound(he_expr, 'ub', self.constraint_tol), }

        lb = {'h': set_acados_constraint_bound(h_expr, 'lb'),
              'h0': set_acados_constraint_bound(h0_expr, 'lb'),
              'he': set_acados_constraint_bound(he_expr, 'lb'), }

        # make sure all the ub and lb are 1D numpy arrays
        # (see: https://discourse.acados.org/t/infeasible-qps-when-using-nonlinear-casadi-constraint-expressions/1595/5?u=mxche)
        for key in ub.keys():
            ub[key] = ub[key].flatten() if ub[key].ndim != 1 else ub[key]
            lb[key] = lb[key].flatten() if lb[key].ndim != 1 else lb[key]
        # check ub and lb dimensions
        for key in ub.keys():
            assert ub[key].ndim == 1, f'ub[{key}] is not 1D numpy array'
            assert lb[key].ndim == 1, f'lb[{key}] is not 1D numpy array'
        assert ub['h'].shape == lb['h'].shape, 'h_ub and h_lb have different shapes'

        # pass the constraints to the ocp object
        ocp.model.con_h_expr_0, ocp.model.con_h_expr, ocp.model.con_h_expr_e = \
            h0_expr, h_expr, he_expr
        ocp.dims.nh_0, ocp.dims.nh, ocp.dims.nh_e = \
            h0_expr.shape[0], h_expr.shape[0], he_expr.shape[0]
        # assign constraints upper and lower bounds
        ocp.constraints.uh_0 = ub['h0']
        ocp.constraints.lh_0 = lb['h0']
        ocp.constraints.uh = ub['h']
        ocp.constraints.lh = lb['h']
        ocp.constraints.uh_e = ub['he']
        ocp.constraints.lh_e = lb['he']

        return ocp

    @timing
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
        nx, nu = self.model.nx, self.model.nu
        # set initial condition (0-th state)
        self.acados_ocp_solver.set(0, 'lbx', obs)
        self.acados_ocp_solver.set(0, 'ubx', obs)

        # warm-starting solver
        # NOTE: only for ipopt warm-starting; since acados
        # has a built-in warm-starting mechanism.
        if self.warmstart:
            if self.x_guess is None or self.u_guess is None:
                # compute initial guess with IPOPT
                self.compute_initial_guess(obs)
            for idx in range(self.T + 1):
                init_x = self.x_guess[:, idx]
                self.acados_ocp_solver.set(idx, 'x', init_x)
            for idx in range(self.T):
                if nu == 1:
                    init_u = np.array([self.u_guess[idx]])
                else:
                    init_u = self.u_guess[:, idx]
                self.acados_ocp_solver.set(idx, 'u', init_u)

        # set reference for the control horizon
        goal_states = self.get_references()
        if self.mode == 'tracking':
            self.traj_step += 1

        y_ref = np.concatenate((goal_states[:, :-1], np.zeros((nu, self.T))))
        for idx in range(self.T):
            self.acados_ocp_solver.set(idx, 'yref', y_ref[:, idx])
        y_ref_e = goal_states[:, -1]
        self.acados_ocp_solver.set(self.T, 'yref', y_ref_e)

        # solve the optimization problem
        try:
            if self.use_RTI:
                # preparation phase
                self.acados_ocp_solver.options_set('rti_phase', 1)
                status = self.acados_ocp_solver.solve()

                # feedback phase
                self.acados_ocp_solver.options_set('rti_phase', 2)
                status = self.acados_ocp_solver.solve()
            else:
                status = self.acados_ocp_solver.solve()

            # get the open-loop solution
            if self.x_prev is None and self.u_prev is None:
                self.x_prev = np.zeros((nx, self.T + 1))
                self.u_prev = np.zeros((nu, self.T))
            if self.u_prev is not None and nu == 1:
                self.u_prev = self.u_prev.reshape((1, -1))
            for i in range(self.T + 1):
                self.x_prev[:, i] = self.acados_ocp_solver.get(i, 'x')
            for i in range(self.T):
                self.u_prev[:, i] = self.acados_ocp_solver.get(i, 'u')
            if nu == 1:
                self.u_prev = self.u_prev.flatten()
        except Exception:
            print(colored('Infeasible MPC Problem', 'red'))
            # get the solver status
            self.acados_ocp_solver.print_statistics()
            status = self.acados_ocp_solver.get_stats('status')
            print(f'acados returned status {status}. ')
            # OPTIONAL: shift the x_prev and u_prev and copy the last state
            # self.x_prev = np.concatenate((self.x_guess[:, 1:], np.atleast_2d(self.x_guess[:, -1]).T), axis=1)
            # self.u_prev = np.concatenate((self.u_guess[:, 1:], np.atleast_2d(self.u_guess[:, -1]).T), axis=1)
        action = self.acados_ocp_solver.get(0, 'u')

        self.x_guess = self.x_prev
        self.u_guess = self.u_prev
        self.results_dict['horizon_states'].append(deepcopy(self.x_prev))
        self.results_dict['horizon_inputs'].append(deepcopy(self.u_prev))
        self.results_dict['goal_states'].append(deepcopy(goal_states))

        self.prev_action = action

        # get the open-loop solution
        if self.x_prev is None and self.u_prev is None:
            self.x_prev = np.zeros((nx, self.T + 1))
            self.u_prev = np.zeros((nu, self.T))
        if self.u_prev is not None and nu == 1:
            self.u_prev = self.u_prev.reshape((1, -1))
        for i in range(self.T + 1):
            self.x_prev[:, i] = self.acados_ocp_solver.get(i, 'x')
        for i in range(self.T):
            self.u_prev[:, i] = self.acados_ocp_solver.get(i, 'u')
        if nu == 1:
            self.u_prev = self.u_prev.flatten()

        self.x_guess = self.x_prev
        self.u_guess = self.u_prev
        self.results_dict['horizon_states'].append(deepcopy(self.x_prev))
        self.results_dict['horizon_inputs'].append(deepcopy(self.u_prev))
        self.results_dict['goal_states'].append(deepcopy(goal_states))

        self.prev_action = action
        if self.use_lqr_gain_and_terminal_cost:
            action += self.lqr_gain @ (obs - self.x_prev[:, 0])

        return action
