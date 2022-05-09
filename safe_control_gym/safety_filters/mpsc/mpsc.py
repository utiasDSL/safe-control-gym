"""Model Predictive Safety Certification (MPSC).

The core idea is that any learning controller input can be either certificated as safe or, if not safe, corrected
using an MPC controller based on Tube MPC.

Based on
    * K.P. Wabsersich and M.N. Zeilinger "Linear model predictive safety certification for learning-based control" 2019
      https://arxiv.org/pdf/1803.08552.pdf

"""
import numpy as np
import casadi as cs

from copy import deepcopy
from itertools import product
from munch import munchify
from scipy.linalg import solve_discrete_are
from pytope import Polytope

from safe_control_gym.safety_filters.base_safety_filter import BaseSafetyFilter
from safe_control_gym.controllers.mpc.mpc_utils import get_cost_weight_matrix, discretize_linear_system
from safe_control_gym.safety_filters.mpsc.mpsc_utils import compute_RPI_set, pontryagin_difference_AABB, ellipse_bounding_box
from safe_control_gym.envs.constraints import LinearConstraint, QuadraticContstraint, ConstraintList, ConstrainedVariableType
from safe_control_gym.envs.benchmark_env import Task


class MPSC(BaseSafetyFilter):
    """Model Predictive Safety Certification Class.

    """

    def __init__(self,
                 env_func,
                 horizon: int = 10,
                 q_lin: list = None,
                 r_lin: list = None,
                 n_samples: int = 600,
                 n_samples_terminal_set: int = 100,
                 tau: float = 0.95,
                 warmstart: bool = True,
                 additional_constraints: list = None,
                 use_terminal_set: bool = True,
                 learn_terminal_set: bool = False,
                 **kwargs
                 ):
        """Initialize the MPSC.

        Args:
            env_func (partial gym.Env): Environment for the task.
            q_lin, r_lin (list): Q and R gain matrices for linear controller.
            n_samples (int): Number of samples to create W set.
            tau (float): The constant use in eqn. 8.b. of the paper when finding the RPI.
            warmstart (bool): If the previous MPC soln should be used to warmstart the next mpc step.
            additional_constraints (list): List of additional constraints to consider.
            use_terminal_set (bool): Whether to use a terminal set constraint or not
        """

        # Setup the Environments.
        self.env_func = env_func
        self.env = env_func(randomized_init=False)
        self.training_env = env_func(randomized_init=True)
        
        # Setup attributes.
        self.model = self.env.symbolic
        self.dt = self.model.dt
        self.Q = get_cost_weight_matrix(q_lin, self.model.nx)
        self.R = get_cost_weight_matrix(r_lin, self.model.nu)
        self.X_LIN = np.atleast_2d(self.env.X_GOAL)[0, :].T
        self.U_LIN = np.atleast_2d(self.env.U_GOAL)[0, :]
        self.linear_dynamics_func, self.discrete_dfdx, self.discrete_dfdu = self.set_linear_dynamics()
        self.compute_lqr_gain()
        self.n_samples = n_samples
        self.n_samples_terminal_set = n_samples_terminal_set
        self.horizon = horizon
        self.warmstart = warmstart
        self.tau = tau
        self.use_terminal_set = use_terminal_set
        self.learn_terminal_set = learn_terminal_set

        self.omega_AABB_verts = None
        self.z_prev = None
        self.v_prev = None
        self.terminal_set = None

        self.additional_constraints = additional_constraints
        
        if self.additional_constraints is None:
            additional_constraints = []
        self.constraints, self.state_constraints_sym, self.input_constraints_sym = self.reset_constraints(
                                                                                    self.env.constraints.constraints +
                                                                                    additional_constraints)

        self.kinf = self.horizon - 1
        self.setup_results_dict()

    def reset_constraints(self,
                          constraints
                          ):
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
        """Compute the linear dynamics

        """
        # Original version, used in shooting.
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
        """Compute LQR gain by solving the DARE.

        """
        P = solve_discrete_are(self.discrete_dfdx,
                               self.discrete_dfdu,
                               self.Q,
                               self.R)
        btp = np.dot(self.discrete_dfdu.T, P)
        self.lqr_gain = -np.dot(np.linalg.inv(self.R + np.dot(btp, self.discrete_dfdu)), np.dot(btp,
         self.discrete_dfdx))

    def learn(self,
              env=None
              ):
        """Compute the Robust Positively Invariant (RPI) set that the MPSC used.

        Args:
            env (BenchmarkEnv): If a different environment is to be used for learning, can supply it here.

        """
        if env is None:
            env = self.training_env
        # Create set of error residuals.
        w = np.zeros((self.model.nx, self.n_samples))
        # Use uniform sampling of control inputs and states.
        for i in range(self.n_samples):
            init_state, _ = env.reset()
            if self.env.NAME == 'quadrotor':
                u = np.random.rand(2)/6 - 1/12 + self.U_LIN
            else:
                u = env.action_space.sample() # Will yield a random action within action space.
            x_next_obs, _, _, _ = env.step(u)
            x_next_linear = self.linear_dynamics_func(x0=init_state - self.X_LIN, p=u - self.U_LIN)['xf'].toarray() + self.X_LIN[:,None]
            w[:,i] = x_next_obs - x_next_linear[:,0]
        A_cl = self.discrete_dfdx + self.discrete_dfdu @ self.lqr_gain
        P = compute_RPI_set(A_cl, w, self.tau)
        self.omega_AABB_verts = ellipse_bounding_box(P)
        self.tighten_state_and_input_constraints()
        self.omega_constraint = QuadraticContstraint(self.env,
                                                     P,
                                                     1.0,
                                                     constrained_variable=ConstrainedVariableType.STATE)
        # Now that constraints are defined, setup the optimizer.
        self.setup_optimizer()

        if self.learn_terminal_set:
            if self.additional_constraints is not None:
                print("[WARNING] Terminal set calculation assumes convex constraints")
            
            if self.env.TASK == Task.TRAJ_TRACKING:
                self.terminal_set = Polytope(self.X_LIN)
                self.terminal_set.minimize_V_rep()
            elif self.env.TASK == Task.STABILIZATION:
                self.terminal_set = None
            
            for i in range(self.n_samples_terminal_set):
                if self.terminal_set is None:
                    init_state = self.X_LIN
                else:
                    init_state = self.terminal_set.V[np.random.choice(self.terminal_set.V.shape[0], 1)]

                init_state = init_state.reshape((self.model.nx, 1))                
                init_state += (np.random.rand(self.model.nx, 1) - np.ones((self.model.nx, 1))/2)/2
                
                if self.env.NAME == 'quadrotor':
                    u = np.random.rand(self.model.nu)/6 - 1/12 + self.U_LIN
                else:
                    u = env.action_space.sample() # Will yield a random action within action space.

                _, feasible = self.solve_optimization(obs=init_state, uncertified_input=u)
                if feasible: 
                    if self.terminal_set is None:
                        self.terminal_set = Polytope(self.z_prev.T)
                    else: 
                        points = np.vstack((self.z_prev.T, self.terminal_set.V))
                        self.terminal_set = Polytope(points)
                    self.terminal_set.minimize_V_rep() 
                    self.setup_optimizer()
    
    def load(self,
             path_to_P,
             path_to_terminal_set=None,
             ):
        P = np.load(path_to_P)
        self.omega_AABB_verts = ellipse_bounding_box(P)
        self.tighten_state_and_input_constraints()
        self.omega_constraint = QuadraticContstraint(self.env,
                                                     P,
                                                     1.0,
                                                     constrained_variable=ConstrainedVariableType.STATE)
        # Now that constraints are defined, setup the optimizer.
        self.setup_optimizer()

        if self.learn_terminal_set and path_to_terminal_set is not None:
            terminal_set_verts = np.load(path_to_terminal_set)
            self.terminal_set = Polytope(terminal_set_verts)

    def tighten_state_and_input_constraints(self):
        """Tigthen the state and input constraints based on the RPI.

        """
        if self.omega_AABB_verts is not None:
            K_omega_AABB_verts_raw = (self.lqr_gain @ self.omega_AABB_verts.T).T
            # Take the outermost values.
            K_omega_AABB_verts_raw_limits = np.array([np.amax(K_omega_AABB_verts_raw, axis=0), np.amin(K_omega_AABB_verts_raw, axis=0)])
            self.K_omega_AABB_verts = np.vstack(list(product(*(K_omega_AABB_verts_raw_limits.T))))
            # Get the current input constraint vertices.
            input_constraint = self.constraints.input_constraints
            if len(input_constraint) > 1:
                raise NotImplementedError("MPSC currently can't handle more than 1 constraint")
            
            input_constraint = input_constraint[0]
            if self.training_env.NAME != 'quadrotor':
                U_vertices_raw = [(input_constraint.upper_bounds[i], input_constraint.lower_bounds[i]) for i in range(self.model.nu)]
            else:
                U_vertices_raw = [(input_constraint.upper_bounds[i], -input_constraint.upper_bounds[i]) for i in range(self.model.nu)]
            self.U_vertices = np.clip(np.vstack(list(product(*U_vertices_raw))), -100, 100)
            self.tightened_input_constraint_verts, tightened_input_constr_func\
                = pontryagin_difference_AABB(
                self.U_vertices,
                self.K_omega_AABB_verts)

            if self.training_env.NAME == 'quadrotor':
                self.tightened_input_constraint_verts = np.clip(self.tightened_input_constraint_verts, 0, 100)
            self.tightened_input_constraint = tightened_input_constr_func(env=self.env,
                                                                          constrained_variable=ConstrainedVariableType.INPUT)
            # Get the state constraint vertices.
            state_constraints = self.constraints.state_constraints
            if len(state_constraints) > 1:
                raise NotImplementedError("MPSC currently can't handle more than 1 constraint")
            state_constraints = state_constraints[0]
            X_vertices_raw = [(state_constraints.upper_bounds[i],state_constraints.lower_bounds[i]) for i in range(self.model.nx)]
            self.X_vertices = np.clip(np.vstack(list(product(*X_vertices_raw))), -100, 100)
            self.tightened_state_constraint_verts, tightened_state_constraint_func = pontryagin_difference_AABB(self.X_vertices,
                                                                                                                self.omega_AABB_verts)
            self.tightened_state_constraint = tightened_state_constraint_func(env=self.env,
                                                                              constrained_variable=ConstrainedVariableType.STATE)
        
            self.simple_terminal_set = QuadraticContstraint(env=self.env, 
                                                            P=np.eye(self.model.nx), 
                                                            b=self.env.TASK_INFO['stabilization_goal_tolerance'],
                                                            constrained_variable=ConstrainedVariableType.STATE)
        else:
            raise ValueError("")

    def setup_optimizer(self):
        """Setup the certifying MPC problem.

        """
        # Horizon parameter.
        horizon = self.horizon
        nx, nu = self.model.nx, self.model.nu
        # Define optimizer and variables.
        opti = cs.Opti()
        # States.
        z_var = opti.variable(nx, horizon + 1)
        # Inputs.
        v_var = opti.variable(nu, horizon)
        # Certified input.
        u_tilde = opti.variable(nu, 1)
        # Desired input.
        u_L = opti.parameter(nu, 1)
        # Current observed state.
        x = opti.parameter(nx, 1)
        # Constraints (currently only handles a single constraint for state and input).
        state_constraints = self.tightened_state_constraint.get_symbolic_model()
        input_constraints = self.tightened_input_constraint.get_symbolic_model()
        omega_constraint = self.omega_constraint.get_symbolic_model()
        simple_terminal_constraint = self.simple_terminal_set.get_symbolic_model()
        for i in range(self.horizon):
            # Dynamics constraints (eqn 5.b).
            next_state = self.linear_dynamics_func(x0=z_var[:,i], p=v_var[:,i])['xf']
            opti.subject_to(z_var[:, i + 1] == next_state)
            # Input constraints (eqn 5.c).
            opti.subject_to(input_constraints(v_var[:,i]) <= 0)
            # State Constraints
            opti.subject_to(state_constraints(z_var[:,i]) <= 0)
        # Final state constraints (5.d).
        if self.use_terminal_set:
            if self.terminal_set is not None:
                terminal_constraint = LinearConstraint(env=self.env, A=self.terminal_set.A, b=self.terminal_set.b, constrained_variable=ConstrainedVariableType.STATE)
                terminal_constraint = terminal_constraint.get_symbolic_model()
                opti.subject_to(terminal_constraint(z_var[:, -1]) <= 0)
            else:
                opti.subject_to(simple_terminal_constraint(z_var[:, -1] - self.X_LIN[:,None]) <= 0)
        # Initial state constraints (5.e).
        opti.subject_to(omega_constraint(x - z_var[:, 0]) <= 0)
        # Real input (5.f).
        opti.subject_to(u_tilde == v_var[:,0] + self.lqr_gain @ (x - z_var[:,0]))
        # Cost (# eqn 5.a, note: using 2norm or sqrt makes this infeasible).
        cost = (u_L - u_tilde).T @ (u_L - u_tilde)  
        opti.minimize(cost)
        # Create solver (IPOPT solver as of this version).
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

    def solve_optimization(self,
                           obs,
                           uncertified_input
                           ):
        """Solve the MPC optimization problem for a given observation and uncertified input.

        """
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
        # Initial guess for optimization problem.
        if (self.warmstart and
                self.z_prev is not None and
                self.v_prev is not None and
                self.u_tilde_prev is not None):
            # Shift previous solutions by 1 step.
            z_guess = deepcopy(self.z_prev)
            v_guess = deepcopy(self.v_prev)
            z_guess[:, :-1] = z_guess[:, 1:]
            v_guess[:-1] = v_guess[1:]
            opti.set_initial(z_var, z_guess)
            opti.set_initial(v_var, v_guess)
            opti.set_initial(u_tilde, deepcopy(self.u_tilde_prev))
        # Solve the optimization problem.
        try:
            sol = opti.solve()
            x_val, u_val, u_tilde_val = sol.value(z_var), sol.value(v_var), sol.value(u_tilde)
            self.z_prev = x_val
            self.v_prev = u_val.reshape((self.model.nu), self.horizon)
            self.u_tilde_prev = u_tilde_val
            # Take the first one from solved action sequence.
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

    def certify_action(self,
                       current_state,
                       unsafe_action,
                       ):
        """Algorithm 1 from Wabsersich 2019.

        """

        self.results_dict['unsafe_action'].append(unsafe_action)        
        success = True
        
        action, feasible = self.solve_optimization(current_state, unsafe_action)
        self.results_dict['feasible'].append(feasible)
        if feasible:
            self.kinf = 0
            self.results_dict['kinf'].append(self.kinf)
            self.results_dict['action'].append(action)
            return action, success
        else:
            self.kinf += 1
            self.results_dict['kinf'].append(self.kinf)
            if (self.kinf <= self.horizon-1 and
                self.z_prev is not None and
                self.v_prev is not None):
                action = self.v_prev[:, self.kinf] +\
                         self.lqr_gain @ (current_state.reshape((self.model.nx, 1)) - self.z_prev[:, self.kinf].reshape((self.model.nx, 1)))
                action = action[0, 0]
                clipped_action = np.clip(action, self.constraints.input_constraints[0].lower_bounds, self.constraints.input_constraints[0].upper_bounds)
                
                if np.linalg.norm(clipped_action - action) >= 0.01:
                    success = False
                action = clipped_action
                
                self.results_dict['action'].append(action)
                return action, success
            else:
                action = self.lqr_gain @ current_state
                clipped_action = np.clip(action, self.constraints.input_constraints[0].lower_bounds, self.constraints.input_constraints[0].upper_bounds)
                
                if np.linalg.norm(clipped_action - action) >= 0.01:
                    success = False
                action = clipped_action
                
                self.results_dict['action'].append(action)
                return action, success
    
    def setup_results_dict(self):
        """Setup the results dictionary to store run information.

        """
        self.results_dict = {}
        self.results_dict['feasible'] = []
        self.results_dict['kinf'] = []
        self.results_dict['unsafe_action'] = []
        self.results_dict['action'] = []

    def close_results_dict(self):
        """Cleanup the results dict and munchify it.

        """
        self.results_dict['feasible'] = np.hstack(self.results_dict['feasible'])
        self.results_dict['kinf'] = np.vstack(self.results_dict['kinf'])
        self.results_dict['unsafe_action'] = np.vstack(self.results_dict['unsafe_action'])
        self.results_dict['action'] = np.vstack(self.results_dict['action'])
        self.results_dict = munchify(self.results_dict)

    def close(self):
        """Cleans up resources.

        """
        self.env.close()

    def reset(self):
        """Prepares for training or evaluation.

        """
        self.env.reset()
        self.setup_results_dict()

        # setup reference input
        if self.env.TASK == Task.STABILIZATION:
            self.mode = "stabilization"
        elif self.env.TASK == Task.TRAJ_TRACKING:
            raise NotImplementedError
