"""Model Predictive Safety Certification (MPSC)

Based on
K.P. Wabsersich and M.N. Zeilinger "Linear model predictive safety certification for learning-based control" 2019
https://arxiv.org/pdf/1803.08552.pdf

The general idea is that any learning controller input can be certefied as safe, and if not safe, can be corrected
using an MPC controller based on Tube MPC.

"""

from copy import deepcopy
from itertools import product
import numpy as np
import casadi as cs
from munch import munchify
from scipy.linalg import solve_discrete_are
import torch

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.mpc.mpc_utils import get_cost_weight_matrix, discretize_linear_system
from safe_control_gym.controllers.mpsc.mpsc_utils import compute_RPI_set, pontryagin_difference_AABB, ellipse_bounding_box
from safe_control_gym.envs.constraints import QuadraticContstraint, ConstraintList, ConstraintInputType
from safe_control_gym.envs.benchmark_env import Task


class MPSC(BaseController):
    """Model Predictive Safety Certification Class."""
    def __init__(self,
                 env_func,
                 rl_controller=None,
                 horizon: int = 10,
                 q_lin: list = None,
                 r_lin: list = None,
                 n_samples: int = 600,
                 tau: float = 0.95,
                 warmstart: bool = True,
                 run_length: int = 200,
                 additional_constraints: list = None,
                 **kwargs):
        """Initialize the MPSC

        Args:
            env_func (partial gym.Env): Environment for the task.
            rl_controller (BaseController): The RL controller to certify.
            q_lin, r_lin (list): Q and R gain matrices for linear controller.
            n_samples (int): Number of samples to create W set.
            tau (float): The constant use in eqn. 8.b. of the paper when finding the RPI.
            warmstart (bool): If the previous MPC soln should be used to warmstart the next mpc step.
            run_length (int): How many iterations to run for.
            additional_constraints (list): List of additional constraints to consider.

        """
        # Setup the Environments
        self.env_func = env_func
        self.env = env_func(randomized_init=False)
        self.training_env = env_func(randomized_init=True)

        # Setup attributes
        self.model = self.env.symbolic
        self.dt = self.model.dt
        self.Q = get_cost_weight_matrix(q_lin, self.model.nx)
        self.R = get_cost_weight_matrix(r_lin, self.model.nu)
        self.X_LIN = np.atleast_2d(self.env.X_GOAL)[0, :].T
        self.U_LIN = np.atleast_2d(self.env.U_GOAL)[0, :]
        self.linear_dynamics_func, self.discrete_dfdx, self.discrete_dfdu = self.set_linear_dynamics()
        self.compute_lqr_gain()
        self.n_samples = n_samples
        self.horizon = horizon
        self.warmstart = warmstart
        self.tau = tau
        self.run_length = run_length
        self.rl_controller = rl_controller

        self.omega_AABB_verts = None
        self.z_prev = None
        self.v_prev = None

        if additional_constraints is None:
            additional_constraints = []
        self.constraints, self.state_constraints_sym, self.input_constraints_sym = self.reset_constraints(
                                                                                    self.env.constraints.constraints +
                                                                                    additional_constraints
                                                                                    )

    def reset_constraints(self, constraints):
        """ Setup the constraints list.

        Args:
            constraints (list): List of constraints controller is subject too.
        """
        constraints_list = ConstraintList(constraints)
        state_constraints_sym = constraints_list.get_state_constraint_symbolic_models()
        input_constraints_sym = constraints_list.get_input_constraint_symbolic_models()
        if len(constraints_list.input_state_constraints) > 0:
            raise NotImplementedError('MPSC cannot handle combined state input constraints yet.')
        return constraints_list, state_constraints_sym, input_constraints_sym

    def set_linear_dynamics(self):
        """Compute the linear dynamics"""
        # original version, used in shooting
        dfdxdfdu = self.model.df_func(x=self.X_LIN, u=self.U_LIN)
        dfdx = dfdxdfdu['dfdx'].toarray()
        dfdu = dfdxdfdu['dfdu'].toarray()
        delta_x = cs.MX.sym('delta_x', self.model.nx,1)
        delta_u = cs.MX.sym('delta_u', self.model.nu,1)

        x_dot_lin_vec = dfdx @ delta_x + dfdu @ delta_u
        linear_dynamics_func = cs.integrator(
            'linear_discrete_dynamics', self.model.integration_algo,
            {
                'x': delta_x,
                'p': delta_u,
                'ode': x_dot_lin_vec
            }, {'tf': self.dt}
        )

        discrete_dfdx, discrete_dfdu = discretize_linear_system(dfdx, dfdu, self.dt)
        return linear_dynamics_func, discrete_dfdx, discrete_dfdu

    def compute_lqr_gain(self):
        """Compute LQR gain by solving the DARE."""
        P = solve_discrete_are(self.discrete_dfdx,
                               self.discrete_dfdu,
                               self.Q,
                               self.R)
        btp = np.dot(self.discrete_dfdu.T, P)
        self.lqr_gain = -np.dot(np.linalg.inv(self.R + np.dot(btp, self.discrete_dfdu)), np.dot(btp,
         self.discrete_dfdx))

    def learn(self,
              env=None):
        """Compute the Robust Positively Invariant (RPI) set that the MPSC used.

        Args:
            env (BenchmarkEnv): If a different environment is to be used for learning, can supply it here.

        """
        if env is None:
            env = self.training_env

        # create set of error residuals
        w = np.zeros((self.model.nx, self.n_samples))
        next_true_states = np.zeros((self.model.nx, self.n_samples))
        next_pred_states = np.zeros((self.model.nx, self.n_samples))
        actions = np.zeros((self.model.nu, self.n_samples))

        # Use uniform sampling of control inputs and states.
        for i in range(self.n_samples):
            init_state, info = env.reset()
            u = env.action_space.sample() # will give a random action within action space
            actions[:,i] = u
            x_next_obs, _, _, _ = env.step(u)
            x_next_linear = self.linear_dynamics_func(x0=init_state - self.X_LIN, p=u - self.U_LIN)['xf'].toarray()

            next_true_states[:,i] = x_next_obs
            next_pred_states[:,i] = x_next_linear[:,0]
            w[:,i] = x_next_obs - x_next_linear[:,0]

        A_cl = self.discrete_dfdx + self.discrete_dfdu @ self.lqr_gain

        P = compute_RPI_set(A_cl, w, self.tau)

        self.learn_actions = actions
        self.learn_next_true_states = next_true_states
        self.learn_next_pred_states = next_pred_states
        self.w = w
        self.A_cl = A_cl
        self.omega_P = P

        self.omega_AABB_verts = ellipse_bounding_box(P)
        self.tighten_state_and_input_constraints()
        self.omega_constraint = QuadraticContstraint(self.env,
                                                     P,
                                                     1.0,
                                                     constraint_input_type=ConstraintInputType.STATE)

        # Now that constraints are defined, setup the optimizer
        self.setup_optimizer()

    def tighten_state_and_input_constraints(self):
        """Tigthen the state and input constraints based on the RPI."""
        if self.omega_AABB_verts is not None:
            K_omega_AABB_verts_raw = (self.lqr_gain @ self.omega_AABB_verts.T).T
            # take the outermost values
            self.K_omega_AABB_verts = np.vstack((np.max(K_omega_AABB_verts_raw), np.min(K_omega_AABB_verts_raw)))

            # get the current input constraint vertices
            input_constraint = self.constraints.input_constraints
            if len(input_constraint) > 1:
                raise NotImplementedError("MPSC currently can't handle more than 1 constraint")
            input_constraint = input_constraint[0]
            self.U_vertices = np.array([[input_constraint.high, input_constraint.low]]).T


            self.tightened_input_constraint_verts, tightened_input_constr_func\
                = pontryagin_difference_AABB(
                self.U_vertices,
                self.K_omega_AABB_verts)
            self.tightened_input_constraint = tightened_input_constr_func(env=self.env,
                                                                         constraint_input_type=ConstraintInputType.INPUT)
            # get the state constraint vertices
            state_constraints = self.constraints.state_constraints
            if len(state_constraints) > 1:
                raise NotImplementedError("MPSC currently can't handle more than 1 constraint")
            state_constraints = state_constraints[0]
            X_vertices_raw = [(state_constraints.high[i],state_constraints.low[i]) for i in range(self.model.nx)]
            self.X_vertices = np.clip(np.vstack(list(product(*X_vertices_raw))), -100, 100)

            self.tightened_state_constraint_verts, tightened_state_constraint_func = pontryagin_difference_AABB(
                self.X_vertices,
                self.omega_AABB_verts)
            self.tightened_state_constraint = tightened_state_constraint_func(env=self.env,
                                                                              constraint_input_type=ConstraintInputType.STATE)
        else:
            raise ValueError("")

    def setup_optimizer(self):
        """Setup the certifying MPC problem."""
        # horizon is a parameter
        horizon = self.horizon
        nx, nu = self.model.nx, self.model.nu

        # define optimizer and variables
        opti = cs.Opti()
        # states
        z_var = opti.variable(nx, horizon + 1)
        # inputs
        v_var = opti.variable(nu, horizon)
        # Certified Input
        u_tilde = opti.variable(nu, 1)
        # Desired Input
        u_L = opti.parameter(nu, 1)
        # Current observed state
        x = opti.parameter(nx, 1)

        # CONSTRAINTS
        # Currently only handles a single constraint for state and input
        state_constraints = self.tightened_state_constraint.get_symbolic_model()
        input_constraints = self.tightened_input_constraint.get_symbolic_model()
        omega_constraint = self.omega_constraint.get_symbolic_model()

        # constraints
        for i in range(self.horizon):
            # dynamics constraints eqn 5.b
            next_state = self.linear_dynamics_func(x0=z_var[:,i], p=v_var[:,i])['xf']
            opti.subject_to(z_var[:, i + 1] == next_state)

            # Eqn 5.c
            # input constraints
            opti.subject_to( input_constraints(v_var[:,i]) <= 0)

            # State Constraints
            opti.subject_to(state_constraints(z_var[:,i]) <= 0)

        # final state constraints (5.d)
        opti.subject_to(z_var[:, -1] == 0 )

        # Initial state constraints (5.e)
        opti.subject_to(omega_constraint(x - z_var[:, 0]) <= 0)

        # Real input (5.f)
        opti.subject_to(u_tilde == v_var[:,0] + self.lqr_gain @ (x - z_var[:,0]))

        ## COST
        # Note: Using 2norm or sqrt makes this infeasible
        cost = (u_L - u_tilde).T @ (u_L - u_tilde)  # eqn 5.a
        opti.minimize(cost)

        # create solver (IPOPT solver for now)
        opts = {"ipopt.print_level": 4,
                "ipopt.sb": "yes",
                "ipopt.max_iter": 50,
                "print_time": 1}
        opti.solver('ipopt', opts)



        self.opti_dict = {
            "opti": opti,
            "z_var": z_var,
            "v_var": v_var,
            "u_tilde": u_tilde,
            "u_L": u_L,
            "x": x,
            "cost": cost
        }

    def solve_optimization(self, obs, uncertified_input):
        """Solve the MPC optimization problem for a given observation and uncertified input."""
        opti_dict = self.opti_dict
        opti = opti_dict["opti"]
        z_var = opti_dict["z_var"]
        v_var = opti_dict["v_var"]
        u_tilde = opti_dict["u_tilde"]
        u_L = opti_dict["u_L"]
        x = opti_dict["x"]
        cost = opti_dict["cost"]

        opti.set_value(x, obs)
        opti.set_value(u_L, uncertified_input)

        # initial guess for optim problem
        if (self.warmstart and
                self.z_prev is not None and
                self.v_prev is not None and
                self.u_tilde_prev is not None):
            # shift previous solutions by 1 step
            z_guess = deepcopy(self.x_prev)
            v_guess = deepcopy(self.u_prev)
            z_guess[:, :-1] = z_guess[:, 1:]
            v_guess[:-1] = v_guess[1:]

            opti.set_initial(z_var, z_guess)
            opti.set_initial(v_var, v_guess)
            opti.set_initial(u_tilde, deepcopy(self.u_tilde_prev))

        # solve the optimization problem
        try:
            sol = opti.solve()
            x_val, u_val, u_tilde_val = sol.value(z_var), sol.value(v_var), sol.value(u_tilde)
            self.z_prev = x_val
            self.v_prev = u_val
            self.u_tilde_prev = u_tilde_val
            # take the first one from solved action sequence
            if u_val.ndim > 1:
                action = u_tilde_val
            else:
                action = u_tilde_val
            self.prev_action = u_tilde_val
            feasible = True
        except RuntimeError:
            feasible = False
            action = None

        return action, feasible

    def certify_action(self, obs, u_L):
        """Algorithm 1 from [1]."""
        action, feasible = self.solve_optimization(obs, u_L)
        self.results_dict['feasible'].append(feasible)
        if feasible:
            self.kinf = 0
            self.results_dict['kinf'].append(self.kinf)
            return action
        else:
            self.kinf += 1
            self.results_dict['kinf'].append(self.kinf)
            if (self.kinf <= self.horizon-1 and
                self.z_prev is not None and
                self.v_prev is not None):
                action = self.v_prev[self.kinf] +\
                         self.lqr_gain @ (obs - self.z_prev[:, self.kinf, None])
                return action
            else:
                action = self.lqr_gain @ obs
                return action

    def select_action(self, obs):
        """Selection feedback action.

        Args:
            obs (np.array): Observation from the environment.

        Returns:
            action (np.array): Action to take based on the obs.
            u_L (np.array): The rl_controllers action based on the obs.

        """
        if self.rl_controller is not None:
            with torch.no_grad():
                u_L, v, logp = self.rl_controller.agent.ac.step(
                    torch.FloatTensor(obs).to(self.rl_controller.device))
        else:
            u_L = 2*np.sin(0.01*np.pi*self.time_step) + 0.5*np.sin(0.12*np.pi*self.time_step)
        self.results_dict['learning_actions'].append(u_L)
        action = self.certify_action(obs, u_L)
        action_diff = np.linalg.norm(u_L - action)
        self.results_dict['corrections'].append(action_diff)

        return action, u_L

    def run(self,
            env=None,
            uncertified_env=None,
            run_length=None,
            **kwargs):
        """Run the simulation.

        Args:
            env (BenchmarkEnv): Environment to for the controller to run.
            uncertified_env (BenchmarkEnv): Environement for the uncertified controller to run on for comparison.
            run_length (int): Number of steps to run the MPSC.

        Return:
            results_dict (dict): Dictionary of the run results.

        """
        if env is None:
            env = self.env
        if run_length is None:
            run_length = self.run_length
        if uncertified_env is None:
            uncertified_env = self.env_func(randomized_init=False)

        self.setup_results_dict()
        obs, _ = env.reset()
        self.results_dict['obs'].append(obs)

        self.kinf = self.horizon - 1
        self.time_step = 0
        for i in range(run_length):
            action, u_L = self.select_action(obs)
            obs, _, _, _ = env.step(action)
            self.results_dict['obs'].append(obs)
            self.results_dict['actions'].append(action)
            self.time_step += 1

        uncertified_obs, _ = uncertified_env.reset()
        self.results_dict['uncertified_obs'].append(uncertified_obs)
        for i in range(run_length):
            if self.rl_controller is not None:
                with torch.no_grad():
                    uncertified_action, _, _ = self.rl_controller.agent.ac.step(
                        torch.FloatTensor(uncertified_obs).to(self.rl_controller.device)
                    )
            uncertified_obs, _, _, _ = uncertified_env.step(uncertified_action)
            self.results_dict['uncertified_actions'].append(uncertified_action)
            self.results_dict['uncertified_obs'].append(uncertified_obs)

        self.close_results_dict()
        uncertified_env.close()
        return self.results_dict

    def setup_results_dict(self):
        """Setup the results dictionary to store run information."""
        self.results_dict = {}
        self.results_dict['obs'] = []
        self.results_dict['actions'] = []
        self.results_dict['uncertified_obs'] = []
        self.results_dict['uncertified_actions'] = []
        self.results_dict['cost'] = []
        self.results_dict['learning_actions'] = []
        self.results_dict['corrections'] = [0.0]
        self.results_dict['feasible'] = []
        self.results_dict['kinf'] = []

    def close_results_dict(self):
        """Cleanup the rtesults dict and munchify it."""
        self.results_dict['obs'] = np.vstack(self.results_dict['obs'])
        self.results_dict['uncertified_obs'] = np.vstack(self.results_dict['uncertified_obs'])
        self.results_dict['uncertified_actions'] = np.vstack(self.results_dict['uncertified_actions'])
        self.results_dict['actions'] = np.vstack(self.results_dict['actions'])
        self.results_dict['learning_actions'] = np.vstack(self.results_dict['learning_actions'])
        self.results_dict['corrections'] = np.hstack(self.results_dict['corrections'])
        self.results_dict['kinf'] = np.vstack(self.results_dict['kinf'])
        self.results_dict = munchify(self.results_dict)

    def close(self):
        """Cleans up resources."""
        self.env.close()
        #self.logger.close()

    def reset(self):
        """Prepares for training or evaluation."""
        # setup reference input
        if self.env.TASK == Task.STABILIZATION:
            self.mode = "stabilization"
        elif self.env.TASK == Task.TRAJ_TRACKING:
            raise NotImplementedError
