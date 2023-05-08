'''Standard cost function for MPSC, minimizing change in the next step's input.'''

from safe_control_gym.safety_filters.mpsc.mpsc_cost_function.abstract_cost import MPSC_COST


class ONE_STEP_COST(MPSC_COST):
    '''Standard one step MPSC Cost Function.'''

    def get_cost(self, opti_dict):
        '''Returns the cost function for the MPSC optimization in symbolic form.

        Args:
            opti_dict (dict): The dictionary of optimization variables.

        Returns:
            cost (casadi symbolic expression): The symbolic cost function using casadi.
        '''

        next_u = opti_dict['next_u']
        u_L = opti_dict['u_L']

        cost = (u_L - next_u).T @ (u_L - next_u)
        return cost
