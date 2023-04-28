'''Model Predictive Safety Certification (MPSC).

The core idea is that any learning controller input can be either certificated as safe or, if not safe, corrected
using an MPC controller based on Tube MPC.

Based on
    * K.P. Wabsersich and M.N. Zeilinger 'Linear model predictive safety certification for learning-based control' 2019
      https://arxiv.org/pdf/1803.08552.pdf
'''

import pickle
from itertools import product

import numpy as np
import casadi as cs
from pytope import Polytope

from safe_control_gym.safety_filters.mpsc.mpsc import MPSC
from safe_control_gym.controllers.mpc.mpc_utils import discretize_linear_system, rk_discrete
from safe_control_gym.safety_filters.mpsc.mpsc_utils import compute_RPI_set, pontryagin_difference_AABB, ellipse_bounding_box, Cost_Function
from safe_control_gym.envs.constraints import LinearConstraint, QuadraticContstraint, ConstrainedVariableType
from safe_control_gym.envs.benchmark_env import Task, Environment


class LINEAR_MPSC(MPSC):
    '''Model Predictive Safety Certification Class.'''

    def __init__(self,
                 env_func,
                 horizon: int = 10,
                 q_lin: list = None,
                 r_lin: list = None,
                 integration_algo: str = 'rk4',
                 n_samples: int = 600,
                 n_samples_terminal_set: int = 100,
                 tau: float = 0.95,
                 warmstart: bool = True,
                 additional_constraints: list = None,
                 use_terminal_set: bool = True,
                 learn_terminal_set: bool = False,
                 cost_function: str = Cost_Function.ONE_STEP_COST,
                 **kwargs
                 ):
        '''Initialize the MPSC.

        Args:
            env_func (partial gym.Env): Environment for the task.
            horizon (int): The MPC horizon.
            q_lin, r_lin (list): Q and R gain matrices for linear controller.
            integration_algo (str): The algorithm used for integrating the dynamics,
                either 'LTI', 'rk4', 'rk', or 'cvodes'.
            n_samples (int): Number of samples to create W set.
            n_samples_terminal_set (int): Number of samples to grow the terminal set.
            tau (float): The constant use in eqn. 8.b. of the paper when finding the RPI.
            warmstart (bool): If the previous MPC soln should be used to warmstart the next mpc step.
            additional_constraints (list): List of additional constraints to consider.
            use_terminal_set (bool): Whether to use a terminal set constraint or not.
            learn_terminal_set (bool): Whether to learn a terminal set or not.
            cost_function (str): A string (from Cost_Function) representing the cost function to be used.
        '''

        # Store all params/args.
        for key, value in locals().items():
            if key != 'self' and key != 'kwargs' and '__' not in key:
                self.__dict__[key] = value

        super().__init__(env_func, horizon, q_lin, r_lin, integration_algo, warmstart, additional_constraints, use_terminal_set, cost_function, **kwargs)

        self.terminal_set_verts = None

    def set_dynamics(self):
        '''Compute the linear dynamics.'''

        # Original version, used in shooting.
        dfdxdfdu = self.model.df_func(x=self.X_EQ, u=self.U_EQ)
        dfdx = dfdxdfdu['dfdx'].toarray()
        dfdu = dfdxdfdu['dfdu'].toarray()
        delta_x = cs.MX.sym('delta_x', self.model.nx, 1)
        delta_u = cs.MX.sym('delta_u', self.model.nu, 1)
        self.discrete_dfdx, self.discrete_dfdu = discretize_linear_system(dfdx, dfdu, self.dt)

        if self.integration_algo == 'LTI':
            x_dot_lin_vec = self.discrete_dfdx @ delta_x + self.discrete_dfdu @ delta_u
            dynamics_func = cs.Function('fd',
                                        [delta_x, delta_u],
                                        [x_dot_lin_vec],
                                        ['x0', 'p'],
                                        ['xf'])
        elif self.integration_algo == 'rk4':
            dynamics_func = rk_discrete(self.model.fc_func,
                                        self.model.nx,
                                        self.model.nu,
                                        self.dt)
        else:
            x_dot_lin_vec = dfdx @ delta_x + dfdu @ delta_u
            dynamics_func = cs.integrator(
                'linear_discrete_dynamics', self.integration_algo,
                {
                    'x': delta_x,
                    'p': delta_u,
                    'ode': x_dot_lin_vec
                }, {'tf': self.dt}
            )

        self.dynamics_func = dynamics_func

    def learn(self,
              env=None
              ):
        '''Compute the Robust Positively Invariant (RPI) set that the MPSC used.

        Args:
            env (BenchmarkEnv): If a different environment is to be used for learning, can supply it here.
        '''

        if env is None:
            env = self.training_env
        # Create set of error residuals.
        w = np.zeros((self.model.nx, self.n_samples))
        # Use uniform sampling of control inputs and states.
        for i in range(self.n_samples):
            init_state, _ = env.reset()
            if self.env.NAME == Environment.QUADROTOR:
                u = np.random.rand(self.model.nu) / 20 - 1 / 40 + self.U_EQ
            else:
                u = env.action_space.sample()  # Will yield a random action within action space.
            x_next_obs, _, _, _ = env.step(u)
            x_next_linear = np.squeeze(self.dynamics_func(x0=init_state - self.X_EQ, p=u - self.U_EQ)['xf'].toarray()) + self.X_EQ
            w[:, i] = x_next_obs - x_next_linear
        A_cl = self.discrete_dfdx + self.discrete_dfdu @ self.lqr_gain
        self.P = compute_RPI_set(A_cl, w, self.tau)
        self.omega_AABB_verts = ellipse_bounding_box(self.P)
        self.tighten_state_and_input_constraints()
        self.omega_constraint = QuadraticContstraint(self.env,
                                                     self.P,
                                                     1.0,
                                                     constrained_variable=ConstrainedVariableType.STATE)
        # Now that constraints are defined, setup the optimizer.
        self.setup_optimizer()

        if self.learn_terminal_set:
            if self.additional_constraints is not None:
                print('[WARNING] Terminal set calculation assumes convex constraints')

            if self.env.TASK == Task.TRAJ_TRACKING:
                self.terminal_set = Polytope(self.env.X_GOAL.T)
                self.terminal_set.minimize_V_rep()
            elif self.env.TASK == Task.STABILIZATION:
                self.terminal_set = None

            points = None
            for i in range(self.n_samples_terminal_set):
                if self.terminal_set is None:
                    init_state = self.X_EQ
                else:
                    init_state = self.terminal_set.V[np.random.choice(self.terminal_set.V.shape[0], 1)]

                init_state = init_state.reshape((self.model.nx, 1))
                init_state += (np.random.rand(self.model.nx, 1) - np.ones((self.model.nx, 1)) / 2) / 2

                if self.env.NAME == Environment.QUADROTOR:
                    u = np.random.rand(self.model.nu) / 6 - 1 / 12 + self.U_EQ
                else:
                    u = env.action_space.sample()  # Will yield a random action within action space.

                _, feasible = self.solve_optimization(obs=init_state, uncertified_action=u)
                if feasible:
                    if self.terminal_set is None:
                        self.terminal_set = Polytope(self.z_prev.T)
                    else:
                        points = np.vstack((self.z_prev.T, self.terminal_set.V))
                        self.terminal_set = Polytope(points)
                    self.terminal_set.minimize_V_rep()
                    self.setup_optimizer()

            self.terminal_set_verts = points

    def load(self,
             path,
             ):
        '''Load values used by the MPC.

        Args:
            path (str): Path to the file containing the P matrix and terminal set.
        '''

        with open(path, 'rb') as file:
            parameters = pickle.load(file)

        self.P = parameters['P']
        self.omega_AABB_verts = ellipse_bounding_box(self.P)
        self.tighten_state_and_input_constraints()
        self.omega_constraint = QuadraticContstraint(self.env,
                                                     self.P,
                                                     1.0,
                                                     constrained_variable=ConstrainedVariableType.STATE)

        if self.learn_terminal_set and 'terminal_set' in parameters:
            self.terminal_set_verts = parameters['terminal_set']
            self.terminal_set = Polytope(self.terminal_set_verts)

        # Now that constraints are defined, setup the optimizer.
        self.setup_optimizer()

    def save(self,
             path,
             ):
        '''Saves the values used by the MPC.

        Args:
            path (str): Path to where to save the P matrix and terminal set.
        '''

        parameters = {}
        parameters['P'] = self.P
        if self.learn_terminal_set and self.terminal_set_verts is not None:
            parameters['terminal_set'] = self.terminal_set_verts

        with open(path, 'wb') as file:
            pickle.dump(parameters, file)

    def tighten_state_and_input_constraints(self):
        '''Tigthen the state and input constraints based on the RPI.'''

        K_omega_AABB_verts_raw = (self.lqr_gain @ self.omega_AABB_verts.T).T
        # Take the outermost values.
        K_omega_AABB_verts_raw_limits = np.array([np.amax(K_omega_AABB_verts_raw, axis=0), np.amin(K_omega_AABB_verts_raw, axis=0)])
        self.K_omega_AABB_verts = np.vstack(list(product(*(K_omega_AABB_verts_raw_limits.T))))
        # Get the current input constraint vertices.
        input_constraint = self.constraints.input_constraints
        if len(input_constraint) > 1:
            raise NotImplementedError('MPSC currently can\'t handle more than 1 constraint')

        input_constraint = input_constraint[0]
        if self.training_env.NAME != Environment.QUADROTOR:
            U_vertices_raw = [(input_constraint.upper_bounds[i], input_constraint.lower_bounds[i]) for i in range(self.model.nu)]
        else:
            U_vertices_raw = [(input_constraint.upper_bounds[i], -input_constraint.upper_bounds[i]) for i in range(self.model.nu)]
        self.U_vertices = np.clip(np.vstack(list(product(*U_vertices_raw))), -100, 100)
        self.tightened_input_constraint_verts, tightened_input_constr_func\
            = pontryagin_difference_AABB(
                self.U_vertices,
                self.K_omega_AABB_verts)

        if self.training_env.NAME == Environment.QUADROTOR:
            min_input = input_constraint.lower_bounds[0] + np.max(self.U_vertices) - np.max(self.tightened_input_constraint_verts)
            self.tightened_input_constraint_verts = np.clip(self.tightened_input_constraint_verts, min_input, 100)
        self.tightened_input_constraint = tightened_input_constr_func(env=self.env,
                                                                      constrained_variable=ConstrainedVariableType.INPUT)
        # Get the state constraint vertices.
        state_constraints = self.constraints.state_constraints
        if len(state_constraints) > 1:
            raise NotImplementedError('MPSC currently can\'t handle more than 1 constraint')
        state_constraints = state_constraints[0]
        X_vertices_raw = [(state_constraints.upper_bounds[i], state_constraints.lower_bounds[i]) for i in range(self.model.nx)]
        self.X_vertices = np.clip(np.vstack(list(product(*X_vertices_raw))), -100, 100)
        self.tightened_state_constraint_verts, tightened_state_constraint_func = pontryagin_difference_AABB(self.X_vertices,
                                                                                                            self.omega_AABB_verts)
        self.tightened_state_constraint = tightened_state_constraint_func(env=self.env,
                                                                          constrained_variable=ConstrainedVariableType.STATE)

        self.simple_terminal_set = QuadraticContstraint(env=self.env,
                                                        P=np.eye(self.model.nx),
                                                        b=self.env.TASK_INFO['stabilization_goal_tolerance'],
                                                        constrained_variable=ConstrainedVariableType.STATE)

    def setup_optimizer(self):
        '''Setup the certifying MPC problem.'''

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
        next_u = opti.variable(nu, 1)
        # Desired input.
        u_L = opti.parameter(nu, 1)
        # Current observed state.
        x_init = opti.parameter(nx, 1)
        # Linearization point
        X_EQ = opti.parameter(nx, 1)
        # Reference trajectory and predicted LQR gains
        if self.env.TASK == Task.STABILIZATION:
            X_GOAL = opti.parameter(1, nx)
        elif self.env.TASK == Task.TRAJ_TRACKING:
            X_GOAL = opti.parameter(self.horizon, nx)

        # Constraints (currently only handles a single constraint for state and input).
        state_constraints = self.tightened_state_constraint.get_symbolic_model()
        input_constraints = self.tightened_input_constraint.get_symbolic_model()
        omega_constraint = self.omega_constraint.get_symbolic_model()
        simple_terminal_constraint = self.simple_terminal_set.get_symbolic_model()
        for i in range(self.horizon):
            # Dynamics constraints (eqn 5.b).
            next_state = self.dynamics_func(x0=z_var[:, i], p=v_var[:, i])['xf']
            opti.subject_to(z_var[:, i + 1] == next_state)
            # Input constraints (eqn 5.c).
            opti.subject_to(input_constraints(v_var[:, i] + self.U_EQ) <= 0)
            # State Constraints
            opti.subject_to(state_constraints(z_var[:, i] + X_EQ) <= 0)

        # Final state constraints (5.d).
        if self.use_terminal_set:
            if self.terminal_set is not None:
                terminal_constraint = LinearConstraint(env=self.env, A=self.terminal_set.A, b=self.terminal_set.b, constrained_variable=ConstrainedVariableType.STATE)
                terminal_constraint = terminal_constraint.get_symbolic_model()
                opti.subject_to(terminal_constraint(z_var[:, -1]) <= 0)
            else:
                opti.subject_to(simple_terminal_constraint(z_var[:, -1]) <= 0)

        # Initial state constraints (5.e).
        opti.subject_to(omega_constraint(x_init - z_var[:, 0]) <= 0)
        # Real input (5.f).
        opti.subject_to(next_u == (v_var[:, 0] + self.U_EQ) + self.lqr_gain @ (x_init - z_var[:, 0]))

        # Create solver (IPOPT solver as of this version).
        opts = {'expand': False,
                'ipopt.print_level': 4,
                'ipopt.sb': 'yes',
                'ipopt.max_iter': 50,
                'print_time': 1}
        if self.integration_algo in ['LTI', 'rk4']:
            opts['expand'] = True
        opti.solver('ipopt', opts)
        self.opti_dict = {
            'opti': opti,
            'z_var': z_var,
            'v_var': v_var,
            'next_u': next_u,
            'u_L': u_L,
            'x_init': x_init,
            'X_EQ': X_EQ,
            'X_GOAL': X_GOAL,
        }

        # Cost (# eqn 5.a, note: using 2norm or sqrt makes this infeasible).
        cost = self.cost_function.get_cost(self.opti_dict)
        opti.minimize(cost)

    def before_optimization(self, obs):
        '''Setup the optimization.

        Args:
            obs (ndarray): Current state/observation.
        '''

        if self.env.NAME == Environment.CARTPOLE:
            self.X_EQ = np.array([obs[0], 0, 0, 0])
        elif self.env.NAME == Environment.QUADROTOR:
            self.X_EQ = np.array([obs[0], 0, obs[2], 0, 0, 0])

        opti_dict = self.opti_dict
        opti = opti_dict['opti']
        X_EQ = opti_dict['X_EQ']
        opti.set_value(X_EQ, self.X_EQ)
