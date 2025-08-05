'''Abstract Model Predictive Safety Certification (MPSC).

The core idea is that any learning controller input can be either certificated as safe or, if not safe, corrected
using an MPC controller based on Tube MPC.

Based on
    * K.P. Wabsersich and M.N. Zeilinger 'Linear model predictive safety certification for learning-based control' 2019
      https://arxiv.org/pdf/1803.08552.pdf
'''

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

from safe_control_gym.controllers.lqr.lqr_utils import compute_lqr_gain, get_cost_weight_matrix
from safe_control_gym.controllers.mpc.mpc_utils import reset_constraints
from safe_control_gym.safety_filters.base_safety_filter import BaseSafetyFilter
from safe_control_gym.safety_filters.mpsc.mpsc_cost_function.one_step_cost import ONE_STEP_COST
from safe_control_gym.safety_filters.mpsc.mpsc_utils import Cost_Function, get_trajectory_on_horizon


class MPSC(BaseSafetyFilter, ABC):
    '''Abstract Model Predictive Safety Certification Class.'''

    def __init__(self,
                 env_func,
                 horizon: int = 10,
                 q_lin: list = None,
                 r_lin: list = None,
                 integration_algo: str = 'rk4',
                 warmstart: bool = True,
                 additional_constraints: list = None,
                 use_terminal_set: bool = True,
                 cost_function: Cost_Function = Cost_Function.ONE_STEP_COST,
                 **kwargs
                 ):
        '''Initialize the MPSC.

        Args:
            env_func (partial BenchmarkEnv): Environment for the task.
            horizon (int): The MPC horizon.
            q_lin, r_lin (list): Q and R gain matrices for linear controller.
            integration_algo (str): The algorithm used for integrating the dynamics,
                either 'LTI', 'rk4', 'rk', or 'cvodes'.
            warmstart (bool): If the previous MPC soln should be used to warmstart the next mpc step.
            additional_constraints (list): List of additional constraints to consider.
            use_terminal_set (bool): Whether to use a terminal set constraint or not.
            cost_function (Cost_Function): A string (from Cost_Function) representing the cost function to be used.
        '''

        # Store all params/args.
        for k, v in locals().items():
            if k != 'self' and k != 'kwargs' and '__' not in k:
                self.__dict__[k] = v

        super().__init__(env_func, **kwargs)

        np.random.seed(self.seed)

        # Setup the Environments.
        self.env = env_func(normalized_rl_action_space=False)
        self.training_env = env_func(randomized_init=True,
                                     init_state=None,
                                     cost='quadratic',
                                     normalized_rl_action_space=False,
                                     )

        # Setup attributes.
        self.reset()
        self.dt = self.model.dt
        self.Q = get_cost_weight_matrix(q_lin, self.model.nx)
        self.R = get_cost_weight_matrix(r_lin, self.model.nu)

        self.X_EQ = np.zeros(self.model.nx)
        self.U_EQ = self.model.U_EQ

        self.set_dynamics()
        self.lqr_gain = -compute_lqr_gain(self.model, self.X_EQ, self.U_EQ, self.Q, self.R, discrete_dynamics=True)

        self.terminal_set = None

        if self.additional_constraints is None:
            additional_constraints = []
        self.constraints, self.state_constraints_sym, self.input_constraints_sym = reset_constraints(
            self.env.constraints.constraints + additional_constraints)

        if cost_function == Cost_Function.ONE_STEP_COST:
            self.cost_function = ONE_STEP_COST()
        else:
            raise NotImplementedError(f'The MPSC cost function {cost_function} has not been implemented')

    @abstractmethod
    def set_dynamics(self):
        '''Compute the dynamics.'''
        raise NotImplementedError

    @abstractmethod
    def setup_optimizer(self):
        '''Setup the certifying MPC problem.'''
        raise NotImplementedError

    def before_optimization(self, obs):
        '''Setup the optimization.

        Args:
            obs (ndarray): Current state/observation.
        '''
        return

    def solve_optimization(self,
                           obs,
                           uncertified_action,
                           iteration=None,
                           ):
        '''Solve the MPC optimization problem for a given observation and uncertified input.

        Args:
            obs (ndarray): Current state/observation.
            uncertified_action (ndarray): The uncertified_controller's action.
            iteration (int): The current iteration, used for trajectory tracking.

        Returns:
            action (ndarray): The certified action.
            feasible (bool): Whether the safety filtering was feasible or not.
        '''

        opti_dict = self.opti_dict
        opti = opti_dict['opti']
        z_var = opti_dict['z_var']
        v_var = opti_dict['v_var']
        next_u = opti_dict['next_u']
        u_L = opti_dict['u_L']
        x_init = opti_dict['x_init']
        X_GOAL = opti_dict['X_GOAL']
        opti.set_value(x_init, obs - self.X_EQ)
        opti.set_value(u_L, uncertified_action)
        clipped_X_GOAL = get_trajectory_on_horizon(self.env, iteration, self.horizon)
        opti.set_value(X_GOAL, clipped_X_GOAL)

        self.cost_function.prepare_cost_variables(opti_dict, obs, iteration)

        # Initial guess for optimization problem.
        if (self.warmstart and self.z_prev is not None and self.v_prev is not None):
            # Shift previous solutions by 1 step.
            z_guess = deepcopy(self.z_prev)
            v_guess = deepcopy(self.v_prev)
            z_guess[:, :-1] = z_guess[:, 1:]
            v_guess[:-1] = v_guess[1:]
            opti.set_initial(z_var, z_guess)
            opti.set_initial(v_var, v_guess)

        # Solve the optimization problem.
        try:
            sol = opti.solve()
            x_val, u_val, next_u_val = sol.value(z_var), sol.value(v_var), sol.value(next_u)
            self.z_prev = x_val
            self.v_prev = u_val.reshape((self.model.nu), self.horizon)
            self.next_u_prev = next_u_val
            # Take the first one from solved action sequence.
            action = next_u_val
            self.prev_action = next_u_val
            feasible = True
        except Exception as e:
            print('Error Return Status:', opti.debug.return_status())
            print(e)
            feasible = False
            action = None
        return action, feasible

    def certify_action(self,
                       current_state,
                       uncertified_action,
                       info=None,
                       ):
        '''Algorithm 1 from Wabsersich 2019.

        Args:
            current_state (ndarray): Current state/observation.
            uncertified_action (ndarray): The uncertified_controller's action.
            info (dict): The info at this timestep.

        Returns:
            certified_action (ndarray): The certified action
            success (bool): Whether the safety filtering was successful or not.
        '''
        uncertified_action = np.clip(uncertified_action, self.env.physical_action_bounds[0], self.env.physical_action_bounds[1])
        self.results_dict['uncertified_action'].append(uncertified_action)
        success = True

        self.before_optimization(current_state)
        iteration = self.extract_step(info)
        action, feasible = self.solve_optimization(current_state, uncertified_action, iteration)
        self.results_dict['feasible'].append(feasible)

        if feasible:
            self.kinf = 0
            certified_action = action
        else:
            self.kinf += 1
            if (self.kinf <= self.horizon - 1 and self.z_prev is not None and self.v_prev is not None):
                action = np.squeeze(self.v_prev[:, self.kinf]) + \
                    np.squeeze(self.lqr_gain @ (current_state.reshape((self.model.nx, 1)) - self.z_prev[:, self.kinf].reshape((self.model.nx, 1))))
                if self.integration_algo == 'LTI':
                    action = np.squeeze(action) + np.squeeze(self.U_EQ)
                action = np.squeeze(action)
                clipped_action = np.clip(action, self.constraints.input_constraints[0].lower_bounds, self.constraints.input_constraints[0].upper_bounds)

                if np.linalg.norm(clipped_action - action) >= 0.01:
                    success = False
                certified_action = clipped_action
            else:
                action = np.squeeze(self.lqr_gain @ (current_state - self.X_EQ))
                if self.integration_algo == 'LTI':
                    action += np.squeeze(self.U_EQ)
                clipped_action = np.clip(action, self.constraints.input_constraints[0].lower_bounds, self.constraints.input_constraints[0].upper_bounds)

                success = False
                certified_action = clipped_action

        certified_action = np.squeeze(np.array(certified_action))
        self.results_dict['kinf'].append(self.kinf)
        self.results_dict['certified_action'].append(certified_action)
        self.results_dict['correction'].append(np.linalg.norm(certified_action - uncertified_action))

        return certified_action, success

    def setup_results_dict(self):
        '''Setup the results dictionary to store run information.'''
        self.results_dict = {}
        self.results_dict['feasible'] = []
        self.results_dict['kinf'] = []
        self.results_dict['uncertified_action'] = []
        self.results_dict['certified_action'] = []
        self.results_dict['correction'] = []

    def close(self):
        '''Cleans up resources.'''
        self.env.close()
        self.training_env.close()

    def reset(self):
        '''Prepares for training or evaluation.'''
        self.model = self.get_prior(self.env, self.prior_info)
        self.env.reset()
        self.training_env.reset()
        self.reset_before_run()

    def reset_before_run(self, env=None):
        '''Reinitialize just the controller before a new run.

        Args:
            env (BenchmarkEnv): The environment to be used for the new run.
        '''
        self.z_prev = None
        self.v_prev = None
        self.kinf = self.horizon - 1
        self.setup_results_dict()
