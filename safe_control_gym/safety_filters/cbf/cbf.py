'''Control barrier function (CBF) quadratic programming (QP) safety filter.

Reference:
    * [Control Barrier Functions: Theory and Applications](https://arxiv.org/abs/1903.11199)
'''

import numpy as np
import casadi as cs

from safe_control_gym.safety_filters.base_safety_filter import BaseSafetyFilter
from safe_control_gym.safety_filters.cbf.cbf_utils import linear_function, cbf_cartpole, cartesian_product


class CBF(BaseSafetyFilter):
    '''Control Barrier Function Class. '''

    def __init__(self,
                 env_func,
                 slope: float=0.1,
                 soft_constrained: bool=True,
                 slack_weight: float=10000.0,
                 slack_tolerance: float=1.0E-3,
                 **kwargs):
        '''
        CBF-QP Safety Filter: The CBF's superlevel set defines a positively control invariant set.
        A QP based on the CBF's Lie derivative with respect to the dynamics allows to filter arbitrary control
        inputs to keep the system inside the CBF's superlevel set.

        Args:
            env_func (partial BenchmarkEnv): Functionalized initialization of the environment.
            slope (float): The slope of the linear function in the CBF.
            soft_constrainted (bool): Whether to use soft or hard constraints.
            slack_weight (float): The weight of the slack in the optimization.
            slack_tolerance (float): How high the slack can be in the optimization.
        '''

        # TODO: Combine with CLF approach for stabilization
        # TODO: Currently specific for cartpole! Make more general for other systems, e.g., extend to quadrotor
        #  environment

        super().__init__(env_func=env_func, **kwargs)
        self.env = self.env_func()

        self.slope = slope
        self.soft_constrained = soft_constrained
        self.slack_weight = slack_weight
        self.slack_tolerance = slack_tolerance

        self.input_constraints_sym = self.env.constraints.get_input_constraint_symbolic_models()
        self.state_constraints_sym = self.env.constraints.get_state_constraint_symbolic_models()
        input_constraints = self.env.constraints.input_constraints
        state_constraints = self.env.constraints.state_constraints

        if len(input_constraints) > 1:
            raise NotImplementedError('CBF currently can\'t handle more than 1 constraint')
        if len(input_constraints) == 0:
            raise Exception('CBF requires at least 1 input constraint')
        self.input_constraint = input_constraints[0]

        if len(state_constraints) > 1:
            raise NotImplementedError('CBF currently can\'t handle more than 1 constraint')
        if len(state_constraints) == 0:
            raise Exception('CBF requires at least 1 state constraint')
        self.state_constraint = state_constraints[0]

        self.model = self.env.symbolic
        self.X = self.model.x_sym
        self.u = self.model.u_sym

        if self.env.NAME == 'cartpole':
            self.state_limits = [min(abs(self.state_constraint.upper_bounds[i]), abs(self.state_constraint.lower_bounds[i])) for i in range(self.model.nx)]
            self.cbf = cbf_cartpole(self.X, self.state_limits)
        else:
            raise NotImplementedError('[Error] Currently CBF is only implemented for the cartpole system.')

        # TODO: Extend this to systems that are not control affine. Then we would need to move away from a QP solver
        # Check if the dynamics are control affine
        assert self.is_control_affine()

        # Lie derivative with respect to the dynamics
        self.lie_derivative = self.get_lie_derivative()

        self.linear_func = linear_function(self.slope)

        # Setup Optimizer
        self.setup_optimizer()

        self.reset()

    def get_lie_derivative(self):
        '''Determines the Lie derivative of the CBF with respect to the known dynamics. '''
        dVdx = cs.gradient(self.cbf(X=self.X)['cbf'], self.X)
        LfV = cs.dot(dVdx, self.model.x_dot)
        LfV_func = cs.Function('LfV', [self.X, self.u], [LfV], ['X', 'u'], ['LfV'])
        return LfV_func

    def is_control_affine(self):
        '''Check if the system is control affine. '''
        dfdu = cs.jacobian(self.model.x_dot, self.u)
        return not cs.depends_on(dfdu, self.u)

    def setup_optimizer(self):
        '''Setup the certifying CBF problem. '''
        nx, nu = self.model.nx, self.model.nu

        # Define optimizer
        opti = cs.Opti('conic')  # Tell casadi that it's a conic problem

        # Optimization variable: control input
        u_var = opti.variable(nu, 1)

        # States
        current_state_var = opti.parameter(nx, 1)
        uncertified_action_var = opti.parameter(nu, 1)

        # Evaluate at Lie derivative and CBF at the current state
        lie_derivative_at_x = self.lie_derivative(X=current_state_var, u=u_var)['LfV']
        barrier_at_x = self.cbf(X=current_state_var)['cbf']

        right_hand_side = 0.0
        slack_var = opti.variable(1, 1)

        if self.soft_constrained:
            # Quadratic objective
            cost = 0.5 * cs.norm_2(uncertified_action_var - u_var) ** 2 + self.slack_weight * slack_var**2

            # Soften CBF constraint
            right_hand_side = slack_var

            # Non-negativity constraint on slack variable
            opti.subject_to(slack_var >= 0.0)
        else:
            opti.subject_to(slack_var == 0.0)
            # Quadratic objective
            cost = 0.5 * cs.norm_2(uncertified_action_var - u_var) ** 2

        # CBF constraint
        opti.subject_to(-self.linear_func(x=barrier_at_x)['y'] - lie_derivative_at_x <=
                        right_hand_side)

        # Input constraints
        for input_constraint in self.input_constraints_sym:
            opti.subject_to(input_constraint(u_var) <= 0)

        opti.minimize(cost)

        # Set verbosity option of optimizer
        opts = {'printLevel': 'low', 'error_on_fail': False}

        # Select QP solver
        opti.solver('qpoases', opts)

        self.opti_dict = {
            'opti': opti,
            'u_var': u_var,
            'slack_var': slack_var,
            'current_state_var': current_state_var,
            'uncertified_action_var': uncertified_action_var,
            'cost': cost
        }

    def solve_optimization(self,
                           current_state,
                           uncertified_action,
                           ):
        '''Solve the CBF optimization problem for a given observation and uncertified input.

        Args:
            current_state (ndarray): Current state/observation.
            uncertified_action (ndarray): The uncertified_controller's action.

        Returns:
            certified_action (ndarray): The certified action.
            feasible (bool): Whether the safety filtering was feasible or not.
        '''

        opti_dict = self.opti_dict
        opti = opti_dict['opti']
        u_var = opti_dict['u_var']
        slack_var = opti_dict['slack_var']
        current_state_var = opti_dict['current_state_var']
        uncertified_action_var = opti_dict['uncertified_action_var']
        cost = opti_dict['cost']

        opti.set_value(current_state_var, current_state)
        opti.set_value(uncertified_action_var, uncertified_action)

        try:
            # Solve optimization problem
            sol = opti.solve()
            feasible = True

            certified_action = sol.value(u_var)
            if self.soft_constrained:
                slack_val = sol.value(slack_var)
                if slack_val > self.slack_tolerance:
                    print('\nFailed: Slack greater than tolerance')
                    print('Slack:', slack_val)
                    print('------------------------------------------------')
                    feasible = False
            cost = sol.value(cost)
        except RuntimeError as e:
            print(e)
            feasible = False
            certified_action = opti.debug.value(u_var)
            print('Certified_action:', certified_action)
            if self.soft_constrained:
                slack_val = opti.debug.value(slack_var)
                print('Slack:', slack_val)
            print('Lie Derivative:', self.lie_derivative(X=current_state, u=certified_action)['LfV'])
            print('Linear Function:', self.linear_func(x=self.cbf(X=current_state)['cbf'])['y'])
            print('------------------------------------------------')
        return certified_action, feasible

    def certify_action(self,
                       current_state,
                       uncertified_action,
                       info=None,
                       ):
        '''Determines a safe action from the current state and proposed action.

        Args:
            current_state (ndarray): Current state/observation.
            uncertified_action (ndarray): The uncertified_controller's action.
            info (dict): The info at this timestep.

        Returns:
            certified_action (ndarray): The certified action.
            success (bool): Whether the safety filtering was successful or not.
        '''

        self.results_dict['uncertified_action'].append(uncertified_action)
        certified_action, success = self.solve_optimization(current_state, uncertified_action)
        self.results_dict['feasible'].append(success)
        self.results_dict['certified_action'].append(certified_action)
        self.results_dict['correction'].append(np.linalg.norm(certified_action-uncertified_action))

        return certified_action, success

    def is_cbf(self, num_points=100, tolerance=0.01):
        '''
        Check if the provided CBF candidate is actually a CBF for the system using a gridded approach.

        Args:
            num_points (int): The number of points in each dimension to check.
            tolerance (float): The tolerance of the condition outside the superlevel set.

        Returns:
            valid_cbf (bool): Whether the provided CBF candidate is valid.
            infeasible_states (list): List of all states for which the QP is infeasible.
        '''
        valid_cbf = False
        epsilon = 1e-6

        # Select the states to check the CBF condition
        max_bounds = np.array(self.state_limits)
        # Add some tolerance to the bounds to also check the condition outside of the superlevel set
        max_bounds += tolerance
        min_bounds = -max_bounds

        # state dimension and input dimension
        nx, nu = self.model.nx, self.model.nu

        # Make sure that every vertex is checked
        num_points = max(2 * nx, num_points + num_points % (2 * nx))
        num_points_per_dim = num_points // nx

        # Create the lists of states to check
        states_to_sample = [np.linspace(min_bounds[i], max_bounds[i], num_points_per_dim) for i in range(nx)]
        states_to_check = cartesian_product(*states_to_sample)

        # Set dummy control input
        control_input = np.ones((nu, 1))

        num_infeasible = 0
        num_infeasible_states_inside_set = 0
        infeasible_states = []

        # Check if the optimization problem is feasible for every considered state
        for state in states_to_check:
            # Certify action
            _, success = self.certify_action(state, control_input)

            if not success:
                infeasible_states.append(state)
                num_infeasible += 1
                barrier_at_x = self.cbf(X=state)['cbf']

                # Check if the infeasible point is inside or outside the superlevel set. Note that the sampled region makes up a
                # box, but the superlevel set is not. The superlevel set only needs to be contained inside the box.
                if barrier_at_x > 0.0 + epsilon:
                    num_infeasible_states_inside_set += 1

        print('Number of infeasible states:', num_infeasible)
        print('Number of infeasible states inside superlevel set:', num_infeasible_states_inside_set)

        if num_infeasible_states_inside_set > 0:
            valid_cbf = False
            print('The provided CBF candidate is not a valid CBF.')
        elif num_infeasible > 0:
            valid_cbf = True
            print('The provided CBF candidate is a valid CBF inside its superlevel set for the checked states. '
                  'Consider increasing the sampling resolution to get a more precise evaluation. '
                  'The CBF is not valid on the entire provided domain. Consider softening the CBF constraint by '
                  'setting \'soft_constraint: True\' inside the config.')
        else:
            valid_cbf = True
            print('The provided CBF candidate is a valid CBF for the checked states. '
                  'Consider increasing the sampling resolution to get a more precise evaluation.')

        return valid_cbf, infeasible_states

    def setup_results_dict(self):
        '''Setup the results dictionary to store run information. '''
        self.results_dict = {}
        self.results_dict['feasible'] = []
        self.results_dict['uncertified_action'] = []
        self.results_dict['certified_action'] = []
        self.results_dict['correction'] = []

    def reset(self):
        '''Resets the environment. '''
        self.env.reset()
        self.setup_results_dict()

    def close(self):
        '''Cleans up resources. '''
        self.env.close()
