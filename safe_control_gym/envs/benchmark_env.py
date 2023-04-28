'''Base environment class module.

This module also contains enumerations for cost functions, tasks, disturbances, and quadrotor types.
'''

import os
import copy
from enum import Enum
from abc import ABC, abstractmethod

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from matplotlib import pyplot as plt

from safe_control_gym.envs.constraints import create_constraint_list
from safe_control_gym.envs.disturbances import create_disturbance_list


class Cost(str, Enum):
    '''Reward/cost functions enumeration class. '''

    RL_REWARD = 'rl_reward'  # Default RL reward function.
    QUADRATIC = 'quadratic'  # Quadratic cost.


class Task(str, Enum):
    '''Environment tasks enumeration class. '''

    STABILIZATION = 'stabilization'  # Stabilization task.
    TRAJ_TRACKING = 'traj_tracking'  # Trajectory tracking task.


class Environment(str, Enum):
    '''Environment enumeration class. '''

    CARTPOLE = 'cartpole'  # Cartpole system
    QUADROTOR = 'quadrotor'  # Quadrotor, both 1D and 2D


class BenchmarkEnv(gym.Env, ABC):
    '''Benchmark environment base class.

    Attributes:
        id (int): Unique identifier of the current env instance (among other instances).
    '''

    _count = 0  # Class variable, count env instance in current process.
    NAME = 'base'  # Environment name.
    URDF_PATH = None  # Path to urdf file that defines base parameters of the robot.
    AVAILABLE_CONSTRAINTS = None  # Dict of constraint names & classes.
    DISTURBANCE_MODES = None  # Dict of disturbance mode names & shared args, e.g. dim of the affected variable.
    INERTIAL_PROP_RAND_INFO = None  # Dict of parameters & distributions for domain randomization.
    INIT_STATE_RAND_INFO = None  # Dict of state name & distribution info to randomize at episode reset
    TASK_INFO = None  # Dict of task related info, e.g. goal state or trajectory args.

    def __init__(self,
                 output_dir=None,
                 seed=None,
                 info_in_reset: bool = False,
                 gui: bool = False,
                 verbose: bool = False,
                 normalized_rl_action_space: bool = False,
                 # Task.
                 task: Task = Task.STABILIZATION,
                 task_info=None,
                 cost: Cost = Cost.RL_REWARD,
                 pyb_freq: int = 50,
                 ctrl_freq: int = 50,
                 episode_len_sec: int = 5,
                 # Initialization.
                 init_state=None,
                 randomized_init: bool = True,
                 init_state_randomization_info=None,
                 # Domain randomization.
                 prior_prop=None,
                 inertial_prop=None,
                 randomized_inertial_prop: bool = False,
                 inertial_prop_randomization_info=None,
                 # Constraint.
                 constraints=None,
                 done_on_violation: bool = False,
                 use_constraint_penalty=False,
                 constraint_penalty=-1,
                 # Disturbance.
                 disturbances=None,
                 adversary_disturbance=None,
                 adversary_disturbance_offset=0.0,
                 adversary_disturbance_scale=0.01,
                 **kwargs
                 ):
        '''Initialization method for BenchmarkEnv.

        Args:
            output_dir (str, optional): Path to directory to save any env outputs.
            seed (int, optional): Seed for the random number generator.
            info_in_reset (bool, optional): Whether .reset() returns a dictionary with the
                                            environment's symbolic model.
            gui (bool, optional): Whether to show PyBullet's GUI.
            verbose (bool, optional): If to suppress environment print statetments.
            normalized_rl_action_space (bool, optional): Whether to normalize the action space.
            task: (Task, optional): The environment's task (stabilization or traj. tracking).
            task_info (dict, optional): A dictionary with the information used to generate the
                task X and U references.
            cost (Cost, optional): Cost function choice used to compute the reward in .step().
            pyb_freq (int, optional): The frequency at which PyBullet steps (a multiple of ctrl_freq).
            ctrl_freq (int, optional): The frequency at which the environment steps.
            episode_len_sec (int, optional): Maximum episode duration in seconds.
            init_state (ndarray/dict, optional): The initial state of the environment
            randomized_init (bool, optional): Whether to randomize the initial state.
            init_state_randomization_info (dict, optional): A dictionary with information used to
                randomize the initial state.
            prior_prop (dict, optional): The prior inertial properties of the environment.
            inertial_prop (dict, optional): The ground truth inertial properties of the environment.
            randomized_inertial_prop (bool, optional): Whether to randomize the inertial properties.
            inertial_prop_randomization_info (dict, optional): A dictionary with information used
                to randomize the inert. properties.
            constraints (Dict, optional): Dictionary to specify the constraints being used.
            done_on_violation (bool, optional): Whether to return done==True on a constraint violation.
            use_constraint_penalty (bool, optional): If to use shaped reward to penalize potential
                constraint violation.
            constraint_penalty (float, optional): Constraint penalty cost for reward shaping.
            disturbances (dict, optional): Dictionary to specify disturbances being used.
            adversary_disturbance (str, optional): If to use adversary/external disturbance.
            adversary_disturbance_offset (float, optional): Parameterizes the offset of the adversary disturbance.
            adversary_disturbance_scale (float, optional): Parameterizes magnitude of adversary disturbance.

        Attributes:
            id (int): Unique identifier of the current env instance (among other instances).
        '''
        # Assign unique ID based on env instance count.
        self.id = self.__class__._count
        self.__class__._count += 1
        # Directory to save any env output.
        if output_dir is None:
            output_dir = os.getcwd()
        self.output_dir = output_dir
        self.GUI = gui
        self.VERBOSE = verbose
        # Task.
        self.TASK = Task(task)
        if task_info is not None:
            self.TASK_INFO = task_info
        # Set timing constants.
        self.CTRL_FREQ = ctrl_freq
        self.PYB_FREQ = pyb_freq
        if self.PYB_FREQ % self.CTRL_FREQ != 0:
            raise ValueError('[ERROR] in BenchmarkEnv.__init__(), pyb_freq is not divisible by env_freq.')
        self.PYB_STEPS_PER_CTRL = int(self.PYB_FREQ / self.CTRL_FREQ)
        self.CTRL_TIMESTEP = 1. / self.CTRL_FREQ
        self.PYB_TIMESTEP = 1. / self.PYB_FREQ
        # Maximum episode length in seconds.
        self.EPISODE_LEN_SEC = episode_len_sec
        self.CTRL_STEPS = self.EPISODE_LEN_SEC * self.CTRL_FREQ
        # Initialization of state.
        self.INIT_STATE = init_state
        self.RANDOMIZED_INIT = randomized_init
        if init_state_randomization_info is not None:
            self.INIT_STATE_RAND_INFO = init_state_randomization_info
        # Domain randomization on parameters.
        self.PRIOR_PROP = prior_prop
        self.INERTIAL_PROP = inertial_prop
        self.RANDOMIZED_INERTIAL_PROP = randomized_inertial_prop
        if inertial_prop_randomization_info is not None:
            self.INERTIAL_PROP_RAND_INFO = inertial_prop_randomization_info
        # Set up action and observation space.
        self.NORMALIZED_RL_ACTION_SPACE = normalized_rl_action_space
        # Define cost-related quantities.
        self.COST = Cost(cost)
        # Create action and observation spaces.
        self._set_action_space()
        self._set_observation_space()
        # Store action (input) and observation spaces dimensions.
        # if observation is not the same as state, env should also have a `state_space`
        # and `state_dim` is queried from it.
        self.action_dim = self.action_space.shape[0]
        self.obs_dim = self.observation_space.shape[0]
        if hasattr(self, 'state_space'):
            self.state_dim = self.state_space.shape[0]
        else:
            self.state_dim = self.obs_dim
        # Default Q and R matrices for quadratic cost.
        if self.COST == Cost.QUADRATIC:
            self.Q = np.eye(self.observation_space.shape[0])
            self.R = np.eye(self.action_space.shape[0])
        # Set constraint info.
        self.CONSTRAINTS = constraints
        self.DONE_ON_VIOLATION = done_on_violation
        self.use_constraint_penalty = use_constraint_penalty
        self.constraint_penalty = constraint_penalty
        self._setup_constraints()
        # Set disturbance info.
        self.DISTURBANCES = disturbances
        self.adversary_disturbance = adversary_disturbance
        self.adversary_disturbance_offset = adversary_disturbance_offset
        self.adversary_disturbance_scale = adversary_disturbance_scale
        self._setup_disturbances()
        # Default seed None means pure randomness/no seeding.
        self.seed(seed)
        self.initial_reset = False
        self.at_reset = False
        self.INFO_IN_RESET = info_in_reset

    def seed(self,
             seed=None
             ):
        '''Sets up a random number generator for a given seed.

        Remember to seed all random generators, currently in
        - env
        - action_space
        - disturbances

        Args:
            seed (int): The seed used to seed the random number generator.

        Returns:
            seeds (list): A list containing the seed.
        '''

        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        for _, disturbs in self.disturbances.items():
            disturbs.seed(self)
        return [seed]

    def set_cost_function_param(self,
                                Q,
                                R
                                ):
        '''Set the cost function parameters.

        Args:
            Q (ndarray): State weight matrix (nx by nx).
            R (ndarray): Input weight matrix (nu by nu).
        '''

        if not self.initial_reset:
            self.Q = Q
            self.R = R
        else:
            raise RuntimeError(
                '[ERROR] env.set_cost_function_param() cannot be called after the first reset of the environment.'
            )

    def set_adversary_control(self, action):
        '''Sets disturbance by an adversary controller, called before (each) step().

        Args:
            action (ndarray): The action.
        '''
        if self.adversary_disturbance is not None:
            clipped_adv_action = np.clip(action, self.adversary_action_space.low, self.adversary_action_space.high)
            self.adv_action = clipped_adv_action * self.adversary_disturbance_scale + self.adversary_disturbance_offset
        else:
            raise RuntimeError(
                '[ERROR] adversary_disturbance does not exist, env.set_adversary_control() cannot be called.'
            )

    def _check_initial_reset(self):
        '''Makes sure that .reset() is called at least once before .step(). '''
        if not self.initial_reset:
            raise RuntimeError(
                '[ERROR] You must call env.reset() at least once before using env.step().'
            )

    def _randomize_values_by_info(self,
                                  original_values,
                                  randomization_info
                                  ):
        '''Randomizes a list of values according to desired distributions.

        Args:
            original_values (dict): A dict of orginal values.
            randomization_info (dict): A dictionary containing information about the distributions
                                       used to randomize original_values.

        Returns:
            randomized_values (dict): A dict of randomized values.
        '''

        # Start from a copy of the original values.
        randomized_values = copy.deepcopy(original_values)
        # Copy the info dict to parse it with 'pop'.
        rand_info_copy = copy.deepcopy(randomization_info)
        # Randomized and replace values for which randomization info are given.
        for key in original_values:
            if key in rand_info_copy:
                # Get distribution removing it from info dict.
                distrib = getattr(self.np_random,
                                  rand_info_copy[key].pop('distrib'))
                # Pop positional args.
                d_args = rand_info_copy[key].pop('args', [])
                # Keyword args are just anything left.
                d_kwargs = rand_info_copy[key]
                # Randomize (adding to the original values).
                randomized_values[key] += distrib(*d_args, **d_kwargs)
        return randomized_values

    @abstractmethod
    def _setup_symbolic(self, prior_prop={}, **kwargs):
        '''Creates a symbolic (CasADi) model for dynamics and cost.

        Args:
            prior_prop (dict): specify the prior inertial prop to use in the symbolic model.
        '''
        raise NotImplementedError

    def _setup_disturbances(self):
        '''Creates attributes and action spaces for the disturbances. '''
        # Default: no passive disturbances.
        self.disturbances = {}
        if self.DISTURBANCES is not None:
            for mode, disturb_specs in self.DISTURBANCES.items():
                assert mode in self.DISTURBANCE_MODES, '[ERROR] in BenchmarkEnv._setup_disturbances(), disturbance mode not available.'
                mode_shared_args = self.DISTURBANCE_MODES[mode]
                self.disturbances[mode] = create_disturbance_list(disturb_specs, mode_shared_args, self)
        # Adversary disturbance (set from outside of env, active/non-passive).
        if self.adversary_disturbance is not None:
            assert self.adversary_disturbance in self.DISTURBANCE_MODES, '[ERROR] in Cartpole._setup_disturbances()'
            shared_args = self.DISTURBANCE_MODES[self.adversary_disturbance]
            dim = shared_args['dim']
            self.adversary_action_space = spaces.Box(low=-1, high=1, shape=(dim,))
            # Adversary obs are the same as those of the protagonist.
            self.adversary_observation_space = self.observation_space

    def _setup_constraints(self):
        '''Creates a list of constraints as an attribute. '''
        self.constraints = None
        self.num_constraints = 0
        if self.CONSTRAINTS is not None:
            self.constraints = create_constraint_list(self.CONSTRAINTS, self.AVAILABLE_CONSTRAINTS, self)
            self.num_constraints = self.constraints.num_constraints

    @abstractmethod
    def _set_action_space(self):
        '''Defines the action space of the environment. '''
        raise NotImplementedError

    @abstractmethod
    def _set_observation_space(self):
        '''Defines the observation space of the environment.

        Sets `self.observation_space`, if observation is not identical to state,
        e.g. in RL where obs is [state, goal] or angle is converted to sine & cosine,
        additionally sets a `self.state_space`.
        '''
        raise NotImplementedError

    def before_reset(self, seed=None):
        '''Pre-processing before calling `.reset()`.

        Args:
            seed (int): Number to reset the env with a new random seed.
        '''
        # Housekeeping variables.
        self.initial_reset = True
        self.at_reset = True
        self.pyb_step_counter = 0
        self.ctrl_step_counter = 0
        self.current_raw_action = None  # Action sent by controller, possibly normalized and unclipped
        self.current_physical_action = None  # current_raw_action unnormalized if it was normalized
        self.current_noisy_physical_action = None  # current_physical_action with noise added
        self.current_clipped_action = None  # current_noisy_physical_action clipped to physical action bounds
        # Reset the disturbances.
        for mode in self.disturbances.keys():
            self.disturbances[mode].reset(self)
        if self.adversary_disturbance is not None:
            self.adv_action = None
        if seed is not None:
            self.seed(seed)

    def after_reset(self, obs, info):
        '''Post-processing after calling `.reset()`.

        Args:
            obs (ndarray): The first observation.
            info (dict): The first info.

        Returns:
            obs (ndarray): The updated first observation.
            info (dict): The updated first info.
        '''
        # Add initial constraint info (no action/input yet, so only state-based constraints)
        info['current_step'] = 0
        if self.constraints is not None and not(self.constraints.state_constraints == []):
            info['constraint_values'] = self.constraints.get_values(self, only_state=True)
        self.at_reset = False
        return obs, info

    @abstractmethod
    def _preprocess_control(self, action):
        '''Pre-processes the action passed to `.step()`, default is identity.

        It's suggested that you set `self.current_clipped_action` here,
        if you ever need to use it later on (e.g. to compute reward/cost).

        Args:
            action (ndarray): The raw action returned by the controller.

        Returns:
            action (ndarray): The processed action to be executed.
        '''
        raise NotImplementedError

    @abstractmethod
    def normalize_action(self, action):
        '''Converts a physical action into an normalized action if necessary.

        Args:
            action (ndarray): The action to be converted.

        Returns:
            normalized_action (ndarray): The action in the correct action space.
        '''
        raise NotImplementedError

    @abstractmethod
    def denormalize_action(self, action):
        '''Converts a normalized action into a physical action if necessary.

        Args:
            action (ndarray): The action to be converted.

        Returns:
            physical_action (ndarray): The physical action.
        '''
        raise NotImplementedError

    def before_step(self, action):
        '''Pre-processing before calling `.step()`.

        Args:
            action (ndarray): The raw action returned by the controller.

        Returns:
            action (ndarray): The processed action to be executed.
        '''
        # Sanity check (reset at least once).
        self._check_initial_reset()
        # Save the raw input action.
        action = np.atleast_1d(action)

        if action.ndim != 1:
            raise ValueError('[ERROR]: The action returned by the controller must be 1 dimensional.')

        self.current_raw_action = action
        # Pre-process/clip the action
        processed_action = self._preprocess_control(action)
        return processed_action

    def extend_obs(self, obs, next_step):
        '''Extends an observation with the next self.obs_goal_horizon reference points.

        Args:
            obs (ndarray): The observation to be extended.
            next_step (int): The iteration for which to extend it.

        Returns:
            extended_obs (ndarray): The extended observation.
        '''
        if self.COST == Cost.RL_REWARD and self.TASK == Task.TRAJ_TRACKING and self.obs_goal_horizon > 0:
            wp_idx = [
                min(next_step + i, self.X_GOAL.shape[0]-1)
                for i in range(self.obs_goal_horizon)
            ]
            goal_state = self.X_GOAL[wp_idx].flatten()
            extended_obs = np.concatenate([obs, goal_state])
        elif self.COST == Cost.RL_REWARD and self.TASK == Task.STABILIZATION and self.obs_goal_horizon > 0:
            goal_state = self.X_GOAL.flatten()
            extended_obs = np.concatenate([obs, goal_state])
        else:
            extended_obs = obs

        return extended_obs

    def after_step(self, obs, rew, done, info):
        '''Post-processing after calling `.step()`.

        Args:
            obs (ndarray): The observation after this step.
            rew (float): The reward after this step.
            done (bool): Whether the evaluation is done.
            info (dict): The info after this step.

        Returns:
            obs (ndarray): The udpdated observation after this step.
            rew (float): The udpdated reward after this step.
            done (bool): Whether the evaluation is done.
            info (dict): The udpdated info after this step.
        '''
        # Increment counters
        self.pyb_step_counter += self.PYB_STEPS_PER_CTRL
        self.ctrl_step_counter += 1

        info['current_step'] = self.ctrl_step_counter

        # Terminate when (any) constraint is violated.
        # here we cache the constraint values `c_value`, so we only evaluate the constraints once,
        # but use it in 1) info dict; 2) check constraint violation; 3) check near constraint violation/almost active.
        # it also allows stateful constraint, where evaluation/update should only be done once per time step.
        c_value = None
        if self.constraints is not None:
            c_value = self.constraints.get_values(self)
            info['constraint_values'] = c_value
            if self.constraints.is_violated(self, c_value=c_value):
                info['constraint_violation'] = 1
                if self.DONE_ON_VIOLATION:
                    done = True
            else:
                info['constraint_violation'] = 0
        else:
            info['constraint_violation'] = 0

        # Apply penalized reward when close to constraint violation
        if self.COST == Cost.RL_REWARD:
            if self.constraints is not None and self.use_constraint_penalty and self.constraints.is_almost_active(self, c_value=c_value):
                rew += self.constraint_penalty

        # Terminate when reaching time limit,
        # but distinguish between done due to true termination or time limit reached
        if self.ctrl_step_counter >= self.CTRL_STEPS:
            info['TimeLimit.truncated'] = not done
            done = True
        return obs, rew, done, info

    def _generate_trajectory(self,
                             traj_type='figure8',
                             traj_length=10.0,
                             num_cycles=1,
                             traj_plane='xy',
                             position_offset=np.array([0, 0]),
                             scaling=1.0,
                             sample_time=0.01):
        '''Generates a 2D trajectory.

        Args:
            traj_type (str, optional): The type of trajectory (circle, square, figure8).
            traj_length (float, optional): The length of the trajectory in seconds.
            num_cycles (int, optional): The number of cycles within the length.
            traj_plane (str, optional): The plane of the trajectory (e.g. 'xz').
            position_offset (ndarray, optional): An initial position offset in the plane.
            scaling (float, optional): Scaling factor for the trajectory.
            sample_time (float, optional): The sampling timestep of the trajectory.

        Returns:
            ndarray: The positions in x, y, z of the trajectory sampled for its entire duration.
            ndarray: The velocities in x, y, z of the trajectory sampled for its entire duration.
            ndarray: The scalar speed of the trajectory sampled for its entire duration.
        '''

        # Get trajectory type.
        valid_traj_type = ['circle', 'square', 'figure8']
        if traj_type not in valid_traj_type:
            raise ValueError('Trajectory type should be one of [circle, square, figure8].')
        traj_period = traj_length / num_cycles
        direction_list = ['x', 'y', 'z']
        # Get coordinates indexes.
        if traj_plane[0] in direction_list and traj_plane[
                1] in direction_list and traj_plane[0] != traj_plane[1]:
            coord_index_a = direction_list.index(traj_plane[0])
            coord_index_b = direction_list.index(traj_plane[1])
        else:
            raise ValueError('Trajectory plane should be in form of ab, where a and b can be {x, y, z}.')
        # Generate time stamps.
        times = np.arange(0, traj_length+sample_time, sample_time) # sample time added to make reference one step longer than traj_length
        pos_ref_traj = np.zeros((len(times), 3))
        vel_ref_traj = np.zeros((len(times), 3))
        speed_traj = np.zeros((len(times), 1))
        # Compute trajectory points.
        for t in enumerate(times):
            pos_ref_traj[t[0]], vel_ref_traj[t[0]] = self._get_coordinates(t[1],
                                                                           traj_type,
                                                                           traj_period,
                                                                           coord_index_a,
                                                                           coord_index_b,
                                                                           position_offset[0],
                                                                           position_offset[1],
                                                                           scaling)
            speed_traj[t[0]] = np.linalg.norm(vel_ref_traj[t[0]])
        return pos_ref_traj, vel_ref_traj, speed_traj

    def _get_coordinates(self,
                         t,
                         traj_type,
                         traj_period,
                         coord_index_a,
                         coord_index_b,
                         position_offset_a,
                         position_offset_b,
                         scaling
                         ):
        '''Computes the coordinates of a specified trajectory at time t.

        Args:
            t (float): The time at which we want to sample one trajectory point.
            traj_type (str, optional): The type of trajectory (circle, square, figure8).
            traj_period (float): The period of the trajectory in seconds.
            coord_index_a (int): The index of the first coordinate of the trajectory plane.
            coord_index_b (int): The index of the second coordinate of the trajectory plane.
            position_offset_a (float): The offset in the first coordinate of the trajectory plane.
            position_offset_b (float): The offset in the second coordinate of the trajectory plane.
            scaling (float, optional): Scaling factor for the trajectory.

        Returns:
            pos_ref (ndarray): The position in x, y, z, at time t.
            vel_ref (ndarray): The velocity in x, y, z, at time t.
        '''

        # Get coordinates for the trajectory chosen.
        if traj_type == 'figure8':
            coords_a, coords_b, coords_a_dot, coords_b_dot = self._figure8(
                t, traj_period, scaling)
        elif traj_type == 'circle':
            coords_a, coords_b, coords_a_dot, coords_b_dot = self._circle(
                t, traj_period, scaling)
        elif traj_type == 'square':
            coords_a, coords_b, coords_a_dot, coords_b_dot = self._square(
                t, traj_period, scaling)
        # Initialize position and velocity references.
        pos_ref = np.zeros((3,))
        vel_ref = np.zeros((3,))
        # Set position and velocity references based on the plane of the trajectory chosen.
        pos_ref[coord_index_a] = coords_a + position_offset_a
        vel_ref[coord_index_a] = coords_a_dot
        pos_ref[coord_index_b] = coords_b + position_offset_b
        vel_ref[coord_index_b] = coords_b_dot
        return pos_ref, vel_ref

    def _figure8(self,
                 t,
                 traj_period,
                 scaling
                 ):
        '''Computes the coordinates of a figure8 trajectory at time t.

        Args:
            t (float): The time at which we want to sample one trajectory point.
            traj_period (float): The period of the trajectory in seconds.
            scaling (float, optional): Scaling factor for the trajectory.

        Returns:
            coords_a (float): The position in the first coordinate.
            coords_b (float): The position in the second coordinate.
            coords_a_dot (float): The velocity in the first coordinate.
            coords_b_dot (float): The velocity in the second coordinate.
        '''

        traj_freq = 2.0 * np.pi / traj_period
        coords_a = scaling * np.sin(traj_freq * t)
        coords_b = scaling * np.sin(traj_freq * t) * np.cos(traj_freq * t)
        coords_a_dot = scaling * traj_freq * np.cos(traj_freq * t)
        coords_b_dot = scaling * traj_freq * (np.cos(traj_freq * t)**2 - np.sin(traj_freq * t)**2)
        return coords_a, coords_b, coords_a_dot, coords_b_dot

    def _circle(self,
                t,
                traj_period,
                scaling
                ):
        '''Computes the coordinates of a circle trajectory at time t.

        Args:
            t (float): The time at which we want to sample one trajectory point.
            traj_period (float): The period of the trajectory in seconds.
            scaling (float, optional): Scaling factor for the trajectory.

        Returns:
            coords_a (float): The position in the first coordinate.
            coords_b (float): The position in the second coordinate.
            coords_a_dot (float): The velocity in the first coordinate.
            coords_b_dot (float): The velocity in the second coordinate.
        '''

        traj_freq = 2.0 * np.pi / traj_period
        coords_a = scaling * np.cos(traj_freq * t)
        coords_b = scaling * np.sin(traj_freq * t)
        coords_a_dot = -scaling * traj_freq * np.sin(traj_freq * t)
        coords_b_dot = scaling * traj_freq * np.cos(traj_freq * t)
        return coords_a, coords_b, coords_a_dot, coords_b_dot

    def _square(self,
                t,
                traj_period,
                scaling
                ):
        '''Computes the coordinates of a square trajectory at time t.

        Args:
            t (float): The time at which we want to sample one trajectory point.
            traj_period (float): The period of the trajectory in seconds.
            scaling (float, optional): Scaling factor for the trajectory.

        Returns:
            coords_a (float): The position in the first coordinate.
            coords_b (float): The position in the second coordinate.
            coords_a_dot (float): The velocity in the first coordinate.
            coords_b_dot (float): The velocity in the second coordinate.
        '''

        # Compute time for each segment to complete.
        segment_period = traj_period / 4.0
        traverse_speed = scaling / segment_period
        # Compute time for the cycle.
        cycle_time = t % traj_period
        # Check time along the current segment and ratio of completion.
        segment_time = cycle_time % segment_period
        # Check current segment index.
        segment_index = int(np.floor(cycle_time / segment_period))
        # Position along segment
        segment_position = traverse_speed * segment_time
        if segment_index == 0:
            # Moving up along second axis from (0, 0).
            coords_a = 0.0
            coords_b = segment_position
            coords_a_dot = 0.0
            coords_b_dot = traverse_speed
        elif segment_index == 1:
            # Moving left along first axis from (0, 1).
            coords_a = -segment_position
            coords_b = scaling
            coords_a_dot = -traverse_speed
            coords_b_dot = 0.0
        elif segment_index == 2:
            # Moving down along second axis from (-1, 1).
            coords_a = -scaling
            coords_b = scaling - segment_position
            coords_a_dot = 0.0
            coords_b_dot = -traverse_speed
        elif segment_index == 3:
            # Moving right along second axis from (-1, 0).
            coords_a = -scaling + segment_position
            coords_b = 0.0
            coords_a_dot = traverse_speed
            coords_b_dot = 0.0
        return coords_a, coords_b, coords_a_dot, coords_b_dot

    def _plot_trajectory(self,
                         traj_type,
                         traj_plane,
                         traj_length,
                         num_cycles,
                         pos_ref_traj,
                         vel_ref_traj,
                         speed_traj
                         ):
        '''Plots a trajectory along x, y, z, and in a 3D projection.

        Args:
            traj_type (str, optional): The type of trajectory (circle, square, figure8).
            traj_plane (str, optional): The plane of the trajectory (e.g. 'xz').
            traj_length (float, optional): The length of the trajectory in seconds.
            num_cycles (int, optional): The number of cycles within the length.
            pos_ref_traj (ndarray): The positions in x, y, z of the trajectory sampled for its entire duration.
            vel_ref_traj (ndarray): The velocities in x, y, z of the trajectory sampled for its entire duration.
            speed_traj (ndarray): The scalar speed of the trajectory sampled for its entire duration.
        '''

        # Print basic properties.
        print('Trajectory type: %s' % traj_type)
        print('Trajectory plane: %s' % traj_plane)
        print('Trajectory length: %s sec' % traj_length)
        print('Number of cycles: %d' % num_cycles)
        print('Trajectory period: %.2f sec' % (traj_length / num_cycles))
        print('Angular speed: %.2f rad/sec' % (2.0 * np.pi / (traj_length / num_cycles)))
        print(
            'Position bounds: x [%.2f, %.2f] m, y [%.2f, %.2f] m, z [%.2f, %.2f] m'
            % (min(pos_ref_traj[:, 0]), max(pos_ref_traj[:, 0]),
               min(pos_ref_traj[:, 1]), max(pos_ref_traj[:, 1]),
               min(pos_ref_traj[:, 2]), max(pos_ref_traj[:, 2])))
        print(
            'Velocity bounds: vx [%.2f, %.2f] m/s, vy [%.2f, %.2f] m/s, vz [%.2f, %.2f] m/s'
            % (min(vel_ref_traj[:, 0]), max(vel_ref_traj[:, 0]),
               min(vel_ref_traj[:, 1]), max(vel_ref_traj[:, 1]),
               min(vel_ref_traj[:, 2]), max(vel_ref_traj[:, 2])))
        print('Speed: min %.2f m/s max %.2f m/s mean %.2f' %
              (min(speed_traj), max(speed_traj), np.mean(speed_traj)))
        # Plot in x, y, z.
        fig, axs = plt.subplots(3, 2)
        t = np.arange(0, traj_length, traj_length / pos_ref_traj.shape[0])
        axs[0, 0].plot(t, pos_ref_traj[:, 0])
        axs[0, 0].set_ylabel('pos x (m)')
        axs[1, 0].plot(t, pos_ref_traj[:, 1])
        axs[1, 0].set_ylabel('pos y (m)')
        axs[2, 0].plot(t, pos_ref_traj[:, 2])
        axs[2, 0].set_ylabel('pos z (m)')
        axs[2, 0].set_xlabel('time (s)')
        axs[0, 1].plot(t, vel_ref_traj[:, 0])
        axs[0, 1].set_ylabel('vel x (m)')
        axs[1, 1].plot(t, vel_ref_traj[:, 1])
        axs[1, 1].set_ylabel('vel y (m)')
        axs[2, 1].plot(t, vel_ref_traj[:, 2])
        axs[2, 1].set_ylabel('vel z (m)')
        axs[2, 1].set_xlabel('time (s)')
        plt.show()
        # Plot in 3D.
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(pos_ref_traj[:, 0], pos_ref_traj[:, 1], pos_ref_traj[:, 2])
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        plt.show()
