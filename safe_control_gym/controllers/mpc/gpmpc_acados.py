


import time
from copy import deepcopy
from functools import partial

import casadi as cs
import gpytorch
import numpy as np
import scipy
import torch
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split
from skopt.sampler import Lhs

from safe_control_gym.controllers.lqr.lqr_utils import discretize_linear_system
from safe_control_gym.controllers.mpc.gp_utils import (GaussianProcessCollection, ZeroMeanIndependentGPModel,
                                                       covSEard, kmeans_centriods)
from safe_control_gym.controllers.mpc.linear_mpc import MPC, LinearMPC
from safe_control_gym.controllers.mpc.mpc import MPC
from safe_control_gym.controllers.mpc.gp_mpc import GPMPC
# from safe_control_gym.controllers.mpc.sqp_mpc import SQPMPC
from safe_control_gym.envs.benchmark_env import Task
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel

class GPMPC_ACADOS(GPMPC):
    '''Implements a GP-MPC controller with Acados optimization.'''
    def __init__(
            self,
            env_func,
            seed: int = 1337,
            horizon: int = 5,
            q_mpc: list = [1],
            r_mpc: list = [1],
            constraint_tol: float = 1e-8,
            additional_constraints: list = None,
            soft_constraints: dict = None,
            warmstart: bool = True,
            train_iterations: int = None,
            test_data_ratio: float = 0.2,
            overwrite_saved_data: bool = True,
            optimization_iterations: list = None,
            learning_rate: list = None,
            normalize_training_data: bool = False,
            use_gpu: bool = False,
            gp_model_path: str = None,
            n_ind_points: int = 30,
            inducing_point_selection_method = 'kmeans',
            recalc_inducing_points_at_every_step = False,
            prob: float = 0.955,
            initial_rollout_std: float = 0.005,
            input_mask: list = None,
            target_mask: list = None,
            gp_approx: str = 'mean_eq',
            online_learning: bool = False,
            prior_info: dict = None,
            sparse_gp: bool = False,
            # inertial_prop: list = [1.0],
            prior_param_coeff: float = 1.0,
            terminate_run_on_done: bool = True,
            output_dir: str = 'results/temp',
            compute_ipopt_initial_guess: bool = False,
            use_RTI: bool = False,
            **kwargs
    ):
        
        if prior_info is None or prior_info == {}:
            raise ValueError('GPMPC_ACADOS requires prior_prop to be defined. You may use the real mass properties and then use prior_param_coeff to modify them accordingly.')
        prior_info['prior_prop'].update((prop, val * prior_param_coeff) for prop, val in prior_info['prior_prop'].items())
        self.prior_env_func = partial(env_func, inertial_prop=prior_info['prior_prop'])
        if soft_constraints is None:
            self.soft_constraints_params = {'gp_soft_constraints': False,
                                            'gp_soft_constraints_coeff': 0,
                                            'prior_soft_constraints': False,
                                            'prior_soft_constraints_coeff': 0}
        else:
            self.soft_constraints_params = soft_constraints

        # Initialize the method using linear MPC.
        self.prior_ctrl = LinearMPC(
            self.prior_env_func,
            horizon=horizon,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            warmstart=warmstart,
            soft_constraints=self.soft_constraints_params['prior_soft_constraints'],
            terminate_run_on_done=terminate_run_on_done,
            prior_info=prior_info,
            # runner args
            # shared/base args
            output_dir=output_dir,
            additional_constraints=additional_constraints,
        )
        self.prior_ctrl.reset()
        self.sparse_gp = sparse_gp
        # super().__init__() # TODO: check the inheritance of the class
        super().__init__(
            env_func = env_func,
            seed= seed,
            horizon = horizon,
            q_mpc = q_mpc,
            r_mpc = r_mpc,
            constraint_tol = constraint_tol,
            additional_constraints = additional_constraints,
            soft_constraints = soft_constraints,
            warmstart = warmstart,
            train_iterations = train_iterations,
            test_data_ratio = test_data_ratio,
            overwrite_saved_data = overwrite_saved_data,
            optimization_iterations = optimization_iterations,
            learning_rate = learning_rate,
            normalize_training_data = normalize_training_data,
            use_gpu = use_gpu, 
            gp_model_path = gp_model_path,
            prob = prob,
            initial_rollout_std = initial_rollout_std,
            input_mask = input_mask,
            target_mask = target_mask,
            gp_approx = gp_approx,
            sparse_gp = sparse_gp,
            n_ind_points = n_ind_points,
            inducing_point_selection_method = 'kmeans',
            recalc_inducing_points_at_every_step = False,
            online_learning = online_learning,
            prior_info = prior_info,
            # inertial_prop: list = [1.0],
            prior_param_coeff = prior_param_coeff,
            terminate_run_on_done = terminate_run_on_done,
            output_dir = output_dir,
            **kwargs)
        # self.prior_ctrl = LinearMPC(
        #     self.prior_env_func,
        #     horizon=horizon,
        #     q_mpc=q_mpc,
        #     r_mpc=r_mpc,
        #     warmstart=warmstart,
        #     soft_constraints=self.soft_constraints_params['prior_soft_constraints'],
        #     terminate_run_on_done=terminate_run_on_done,
        #     prior_info=prior_info,
        #     # runner args
        #     # shared/base args
        #     output_dir=output_dir,
        #     additional_constraints=additional_constraints,
        # )
        # self.prior_ctrl.reset()
        # # Setup environments.
        self.env_func = env_func
        self.env = env_func(randomized_init=False, seed=seed)
        self.env_training = env_func(randomized_init=True, seed=seed)
        # No training data accumulated yet so keep the dynamics function as linear prior.
        self.train_data = None
        self.data_inputs = None
        self.data_targets = None
        self.prior_dynamics_func = self.prior_ctrl.linear_dynamics_func
        # self.prior_dynamics_func = self.prior_ctrl.dynamics_func # nonlinear prior 
        self.X_EQ = self.prior_ctrl.X_EQ
        self.U_EQ = self.prior_ctrl.U_EQ
        print(f'X_EQ: {self.X_EQ}, U_EQ: {self.U_EQ}')
        # GP and training parameters.
        self.gaussian_process = None
        self.train_iterations = train_iterations
        self.test_data_ratio = test_data_ratio
        self.overwrite_saved_data = overwrite_saved_data
        self.optimization_iterations = optimization_iterations
        self.learning_rate = learning_rate
        self.gp_model_path = gp_model_path
        self.normalize_training_data = normalize_training_data
        self.prob = prob
        if input_mask is None:
            self.input_mask = np.arange(self.model.nx + self.model.nu).tolist()
        else:
            self.input_mask = input_mask
        if target_mask is None:
            self.target_mask = np.arange(self.model.nx).tolist()
        else:
            self.target_mask = target_mask
        Bd = np.eye(self.model.nx)
        self.Bd = Bd[:, self.target_mask]
        self.gp_approx = gp_approx
        self.online_learning = online_learning
        self.last_obs = None
        self.last_action = None
        self.initial_rollout_std = initial_rollout_std
        # MPC params
        self.gp_soft_constraints = self.soft_constraints_params['gp_soft_constraints']
        self.gp_soft_constraints_coeff = self.soft_constraints_params['gp_soft_constraints_coeff']

        self.init_solver = 'ipopt'
        self.compute_ipopt_initial_guess = compute_ipopt_initial_guess
        self.x_guess = None
        self.u_guess = None
        self.x_prev = None
        self.u_prev = None
        self.use_RTI = use_RTI

        self.setup_prior_dynamics()
        self.setup_acados_model()
        self.setup_acados_optimizer()
        self.acados_ocp_solver = AcadosOcpSolver(self.ocp)
    
    def setup_acados_model(self) -> AcadosModel:

        model_name = self.env.NAME
        
        acados_model = AcadosModel()
        acados_model.x = self.model.x_sym
        acados_model.u = self.model.u_sym
        acados_model.name = model_name

        A_lin = self.discrete_dfdx
        B_lin = self.discrete_dfdu

        if self.gaussian_process is None:
            f_disc = self.prior_dynamics_func(x0=acados_model.x- self.X_EQ, 
                                              p=acados_model.u- self.U_EQ)['xf'] \
                + self.prior_ctrl.X_EQ[:, None]
        else:
            z = cs.vertcat(acados_model.x, acados_model.u) # GP prediction point
            z = z[self.input_mask]
            if self.sparse_gp:
                raise NotImplementedError('Sparse GP not implemented for acados.')
            else:
                f_disc = self.prior_dynamics_func(x0=acados_model.x- self.X_EQ, 
                                                  p=acados_model.u- self.U_EQ)['xf'] \
                + self.prior_ctrl.X_EQ[:, None]
                + self.Bd @ self.gaussian_process.casadi_predict(z=z)['mean']

        acados_model.disc_dyn_expr = f_disc

        acados_model.x_labels = self.env.STATE_LABELS
        acados_model.u_labels = self.env.ACTION_LABELS
        acados_model.t_label = 'time'

        self.acados_model = acados_model

    def setup_acados_optimizer(self):
        print('=================Setting up acados optimizer=================')
        pass
        # before_optimizer_setup = time.time()
        nx, nu = self.model.nx, self.model.nu
        ny = nx + nu
        ny_e = nx

        # create ocp object to formulate the OCP
        ocp = AcadosOcp()
        ocp.model = self.acados_model

        # set dimensions
        ocp.dims.N = self.T # prediction horizon

        # set cost 
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W = scipy.linalg.block_diag(self.Q, self.R)
        ocp.cost.W_e = self.Q
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx:(nx+nu), :nu] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)
        # placeholder y_ref and y_ref_e (will be set in select_action)
        ocp.cost.yref = np.zeros((ny, ))
        ocp.cost.yref_e = np.zeros((ny_e, ))

        # Constraints
        # # bounded input constraints
        ocp.constraints.Jbu = np.eye(nu)
        ocp.constraints.lbu = self.env.constraints.input_constraints[0].lower_bounds
        ocp.constraints.ubu = self.env.constraints.input_constraints[0].upper_bounds
        ocp.constraints.idxbu = np.arange(nu)
        # bounded state constraints
        ocp.constraints.Jbx = np.eye(nx)
        ocp.constraints.lbx = self.env.constraints.state_constraints[0].lower_bounds
        ocp.constraints.ubx = self.env.constraints.state_constraints[0].upper_bounds
        ocp.constraints.idxbx = np.arange(nx)
        # bounded terminal state constraints
        ocp.constraints.Jbx_e = np.eye(nx)
        ocp.constraints.lbx_e = self.env.constraints.state_constraints[0].lower_bounds
        ocp.constraints.ubx_e = self.env.constraints.state_constraints[0].upper_bounds
        ocp.constraints.idxbx_e = np.arange(nx)

        # general constraint expressions
        state_constraint_expr_list = []
        input_constraint_expr_list = []
        state_tighten_list = []
        input_tighten_list = []
        for sc_i, state_constraint in enumerate(self.state_constraints_sym):
            state_constraint_expr_list.append(state_constraint(ocp.model.x))
            # chance state constraint tightening
            state_tighten_list.append(cs.MX.sym(f'state_tighten_{sc_i}', state_constraint(ocp.model.x).shape[0], 1))
        for ic_i, input_constraint in enumerate(self.input_constraints_sym):
            input_constraint_expr_list.append(input_constraint(ocp.model.u))
            # chance input constraint tightening
            input_tighten_list.append(cs.MX.sym(f'input_tighten_{ic_i}', input_constraint(ocp.model.u).shape[0], 1))
        
        h_expr_list = state_constraint_expr_list + input_constraint_expr_list
        h_expr = cs.vertcat(*h_expr_list)
        h0_expr = cs.vertcat(*h_expr_list)
        he_expr = cs.vertcat(*state_constraint_expr_list) # terminal constraints are only state constraints
        # pass the constraints to the ocp object
        ocp = self.processing_acados_constraints_expression(ocp, h0_expr, h_expr, he_expr, state_tighten_list, input_tighten_list)
        # pass the tightening variables to the ocp object as parameters
        tighten_var = cs.vertcat(*state_tighten_list, *input_tighten_list)
        ocp.model.p = tighten_var 
        ocp.parameter_values = np.zeros((tighten_var.shape[0], )) # dummy values

        # slack costs for nonlinear constraints
        if self.gp_soft_constraints:
            raise NotImplementedError('Soft constraints not implemented for acados.')
        

        # placeholder initial state constraint
        x_init = np.zeros((nx))
        ocp.constraints.x0 = x_init

        # set up solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.nlp_solver_type = 'SQP' if not self.use_RTI else 'SQP_RTI'
        ocp.solver_options.nlp_solver_max_iter = 5000
        # prediction horizon
        ocp.solver_options.tf = self.T * self.dt

        self.ocp = ocp

    def processing_acados_constraints_expression(self, ocp: AcadosOcp, h0_expr, h_expr, he_expr, \
                                                 state_tighten_list, input_tighten_list) -> AcadosOcp:
        '''Preprocess the constraints to be compatible with acados.
            Args: 
                h0_expr (casadi expression): initial state constraints
                h_expr (casadi expression): state and input constraints
                he_expr (casadi expression): terminal state constraints
                state_tighten_list (list): list of casadi SX variables for state constraint tightening
                input_tighten_list (list): list of casadi SX variables for input constraint tightening
            Returns:
                ocp (AcadosOcp): acados ocp object with constraints set
        
        Note:
        all constraints in safe-control-gym are defined as g(x, u) <= constraint_tol
        However, acados requires the constraints to be defined as lb <= g(x, u) <= ub
        Thus, a large negative number (-1e8) is used as the lower bound.
        See: https://github.com/acados/acados/issues/650 

        An alternative way to set the constraints is to use bounded constraints of acados:
        # bounded input constraints
        idxbu = np.where(np.sum(self.env.constraints.input_constraints[0].constraint_filter, axis=0) != 0)[0]
        ocp.constraints.Jbu = np.eye(nu)
        ocp.constraints.lbu = self.env.constraints.input_constraints[0].lower_bounds
        ocp.constraints.ubu = self.env.constraints.input_constraints[0].upper_bounds 
        ocp.constraints.idxbu = idxbu # active constraints dimension
        '''
        # NOTE: only the upper bound is tightened due to constraint are defined in the 
        # form of g(x, u) <= constraint_tol in safe-control-gym

        # lambda functions to set the upper and lower bounds of the chance constraints
        constraint_ub_chance = lambda constraint: -self.constraint_tol * np.ones(constraint.shape) 
        constraint_lb_chance = lambda constraint: -1e8 * np.ones(constraint.shape)
        state_tighten_var = cs.vertcat(*state_tighten_list)
        input_tighten_var = cs.vertcat(*input_tighten_list)
        
        ub = {'h': constraint_ub_chance(h_expr - cs.vertcat(state_tighten_var, input_tighten_var)), \
              'h0': constraint_ub_chance(h0_expr - cs.vertcat(state_tighten_var, input_tighten_var)),\
              'he': constraint_ub_chance(he_expr - state_tighten_var)}
        lb = {'h': constraint_lb_chance(h_expr), 'h0': constraint_lb_chance(h0_expr),\
              'he': constraint_lb_chance(he_expr)}

        # make sure all the ub and lb are 1D casaadi SX variables
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
        ocp.model.con_h_expr_0 = h0_expr - cs.vertcat(state_tighten_var, input_tighten_var)
        ocp.model.con_h_expr = h_expr - cs.vertcat(state_tighten_var, input_tighten_var)
        ocp.model.con_h_expr_e = he_expr - state_tighten_var
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

    def select_action(self, obs, info=None):
        print('current obs:', obs)
        time_before = time.time()
        if self.gaussian_process is None:
            action = self.prior_ctrl.select_action(obs)
        else:
            action = self.select_action_with_gp(obs)
        time_after = time.time()
        self.last_obs = obs
        self.last_action = action
        print('gpmpc acados action selection time:', time_after - time_before)
        return action
    
    def select_action_with_gp(self, obs):
        nx, nu = self.model.nx, self.model.nu
        ny = nx + nu
        ny_e = nx

        # set initial condition (0-th state)
        self.acados_ocp_solver.set(0, "lbx", obs)
        self.acados_ocp_solver.set(0, "ubx", obs)
        if self.warmstart:
            if self.x_guess is None or self.u_guess is None:
                if self.compute_ipopt_initial_guess:
                    # compute initial guess with IPOPT
                    self.compute_initial_guess(obs, self.get_references())
                else:
                    self.x_guess = np.zeros((nx, self.T + 1))
                    if nu == 1:
                        self.u_guess = np.zeros((self.T,))
                    else:
                        self.u_guess = np.zeros((nu, self.T))
            for idx in range(self.T + 1):
                init_x = self.x_guess[:, idx]
                self.acados_ocp_solver.set(idx, "x", init_x)
            for idx in range(self.T):
                if nu == 1:
                    init_u = np.array([self.u_guess[idx]])
                else:
                    init_u = self.u_guess[:, idx]
                self.acados_ocp_solver.set(idx, "u", init_u)
        else:
            for idx in range(self.T + 1):
                self.acados_ocp_solver.set(idx, "x", obs)
            for idx in range(self.T):
                self.acados_ocp_solver.set(idx, "u", np.zeros((nu,)))

        # Set the probabilistic state and input constraint set limits.
        # Tightening at the first step is possible if self.compute_initial_guess is used 
        time_before = time.time()
        state_constraint_set_prev, input_constraint_set_prev = self.precompute_probabilistic_limits()
        time_after = time.time()
        print('precompute_probabilistic_limits time:', time_after - time_before)
        
        # for si in range(len(self.constraints.state_constraints)):
        # tighten initial and path constraints
        for idx in range(self.T):
            state_constraint_set = state_constraint_set_prev[0][:, idx]
            input_constraint_set = input_constraint_set_prev[0][:, idx]
            tighten_value = np.concatenate((state_constraint_set, input_constraint_set))
            self.acados_ocp_solver.set(idx, "p", tighten_value)
        # set terminal state constraints
        tighten_value = np.concatenate((state_constraint_set_prev[0][:, self.T], np.zeros((2 * nu,))))
        self.acados_ocp_solver.set(self.T, "p", tighten_value)
        # print('tighten_value:', tighten_value)
        # print('state_constraint_set_prev[0][:, self.T]:', state_constraint_set_prev[0][:, self.T])

        # set reference for the control horizon
        goal_states = self.get_references()
        if self.mode == 'tracking':
            self.traj_step += 1
        for idx in range(self.T):
            y_ref = np.concatenate((goal_states[:, idx], np.zeros((nu,))))
            self.acados_ocp_solver.set(idx, "yref", y_ref)
        y_ref_e = goal_states[:, -1] 
        self.acados_ocp_solver.set(self.T, "yref", y_ref_e)

        # solve the optimization problem
        # try:
        if self.use_RTI:
            # preparation phase
            self.acados_ocp_solver.options_set('rti_phase', 1)
            status = self.acados_ocp_solver.solve()

            # feedback phase
            self.acados_ocp_solver.options_set('rti_phase', 2) 
            status = self.acados_ocp_solver.solve()
            
            if status not in [0, 2]:
                self.acados_ocp_solver.print_statistics()
                raise Exception(f'acados returned status {status}. Exiting.')
                # print(f"acados returned status {status}. ")
            if status == 2:
                print(f"acados returned status {status}. ")
            
            action = self.acados_ocp_solver.get(0, "u")

        else:
            status = self.acados_ocp_solver.solve()
            if status not in [0, 2]:
                self.acados_ocp_solver.print_statistics()
                raise Exception(f'acados returned status {status}. Exiting.')
                # print(f"acados returned status {status}. ")
            if status == 2:
                print(f"acados returned status {status}. ")
            action = self.acados_ocp_solver.get(0, "u")
        # except Exception as e:
        #     print(f"========== acados solver failed with error: {e} =============")
        #     print('using prior controller')
        #     action = self.prior_ctrl.select_action(obs)

        return action
 
    def reset(self):
        '''Reset the controller before running.'''
        # Setup reference input.
        if self.env.TASK == Task.STABILIZATION:
            self.mode = 'stabilization'
            self.x_goal = self.env.X_GOAL
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = 'tracking'
            self.traj = self.env.X_GOAL.T
            self.traj_step = 0
        # Dynamics model.
        
        if self.gaussian_process is not None:
            self.set_gp_dynamics_func(self.n_ind_points)
        self.setup_acados_model()
        self.setup_acados_optimizer()
            # n_ind_points = self.train_data['train_targets'].shape[0]
        print('=========== Resetting prior controller ===========')
        self.prior_ctrl.reset()
        self.setup_results_dict()
        # Previously solved states & inputs, useful for warm start.
        self.x_prev = None
        self.u_prev = None

        self.x_guess = None
        self.u_guess = None

