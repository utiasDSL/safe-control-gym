'''Base class for safety filter. '''

from abc import abstractmethod

from safe_control_gym.controllers.base_controller import BaseController


class BaseSafetyFilter(BaseController):
    '''Template for safety filter, implement the following methods as needed. '''

    @abstractmethod
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
        raise NotImplementedError

    def select_action(self, obs, info=None):
        raise NotImplementedError('[ERROR] select_action is not and will not be implemented for safety filters.')
