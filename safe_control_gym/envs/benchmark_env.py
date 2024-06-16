"""Base environment class module.

This module also contains enumerations for cost functions, tasks, disturbances, and quadrotor types.

"""

import os
from enum import Enum
import numpy as np
import gym
from gym.utils import seeding
import copy

from safe_control_gym.envs.constraints import create_constraint_list
from safe_control_gym.envs.disturbances import create_disturbance_list


class Cost(str, Enum):
    """Reward/cost functions enumeration class."""

    RL_REWARD = "rl_reward"  # Default RL reward function.
    QUADRATIC = "quadratic"  # Quadratic cost.
    COMPETITION = "competition"  # IROS 2022 competition sparse reward.


class Task(str, Enum):
    """Environment tasks enumeration class."""

    STABILIZATION = "stabilization"  # Stabilization task.
    TRAJ_TRACKING = "traj_tracking"  # Trajectory tracking task.


class BenchmarkEnv(gym.Env):
    """Benchmark environment base class.

    Attributes:
        id (int): unique identifier of the current env instance (among other instances).

    """

    _count = 0  # Class variable, count env instance in current process.
    NAME = "base"  # Environment name.
    URDF_PATH = None  # Path to urdf file that defines base parameters of the robot.
    AVAILABLE_CONSTRAINTS = None  # Dict of constraint names & classes.
    DISTURBANCE_MODES = (
        None  # Dict of disturbance mode names & shared args, e.g. dim of the affected variable.
    )
    INERTIAL_PROP_RAND_INFO = None  # Dict of parameters & distributions for domain randomization.
    INIT_STATE_RAND_INFO = (
        None  # Dict of state name & distribution info to randomize at episode reset
    )
    TASK_INFO = None  # Dict of task related info, e.g. goal state or trajectory args.

    def __init__(
        self,
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
        **kwargs,
    ):
        """Initialization method for BenchmarkEnv.

        Args:
            output_dir (str, optional): path to directory to save any env outputs.
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
            use_constraint_penalty (bool, optional): if to use shaped reward to penalize potential
                constraint violation.
            constraint_penalty (float, optional): constraint penalty cost for reward shaping.
            disturbances (dict, optional): Dictionary to specify disturbances being used.
            adversary_disturbance (str, optional): if to use adversary/external disturbance.
            adversary_disturbance_offset (float, optional): parameterizes the offset of the adversary disturbance.
            adversary_disturbance_scale (float, optional): parameterizes magnitude of adversary disturbance.

        Attributes:
            id (int): unique identifier of the current env instance (among other instances).

        """
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
        print(task)
        self.TASK = Task(task)
        if task_info is not None:
            self.TASK_INFO = task_info
        # Set timing constants.
        self.CTRL_FREQ = ctrl_freq
        self.PYB_FREQ = pyb_freq
        if self.PYB_FREQ % self.CTRL_FREQ != 0:
            raise ValueError(
                "[ERROR] in BenchmarkEnv.__init__(), pyb_freq is not divisible by env_freq."
            )
        self.PYB_STEPS_PER_CTRL = int(self.PYB_FREQ / self.CTRL_FREQ)
        self.CTRL_TIMESTEP = 1.0 / self.CTRL_FREQ
        self.PYB_TIMESTEP = 1.0 / self.PYB_FREQ
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
        if hasattr(self, "state_space"):
            self.state_dim = self.state_space.shape[0]
        else:
            self.state_dim = self.obs_dim
        # Default Q and R matrices for quadratic cost.
        if self.COST == Cost.QUADRATIC or self.COST == Cost.COMPETITION:
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
        self.INFO_IN_RESET = info_in_reset

        # IROS 2022 - Save random seed for re-seeding.
        self.RND_SEED = seed
        if "reseed_on_reset" in kwargs:
            self.RESEED_ON_RESET = kwargs["reseed_on_reset"]
        else:
            self.RESEED_ON_RESET = False

        # IROS 2022 - Constrain violation flag for reward.
        self.cnstr_violation = False

    def seed(self, seed=None):
        """Sets up a random number generator for a given seed.

        Remember to seed all random generators, currently in
        - env
        - action_space
        - disturbances

        """
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        for _, disturbs in self.disturbances.items():
            disturbs.seed(self)
        return [seed]

    def _check_initial_reset(self):
        """Makes sure that .reset() is called at least once before .step()."""
        if not self.initial_reset:
            raise RuntimeError(
                "[ERROR] You must call env.reset() at least once before using env.step()."
            )

    def _randomize_values_by_info(self, original_values, randomization_info):
        """Randomizes a list of values according to desired distributions.

        Args:
            original_values (dict): a dict of orginal values.
            randomization_info (dict): A dictionary containing information about the distributions
                                       used to randomize original_values.

        Returns:
            dict: A dict of randomized values.

        """
        # Start from a copy of the original values.
        randomized_values = copy.deepcopy(original_values)
        # Copy the info dict to parse it with "pop".
        rand_info_copy = copy.deepcopy(randomization_info)
        # Randomized and replace values for which randomization info are given.
        for key in original_values:
            if key in rand_info_copy:
                # Get distribution removing it from info dict.
                distrib = getattr(self.np_random, rand_info_copy[key].pop("distrib"))
                # Pop positional args.
                d_args = rand_info_copy[key].pop("args", [])
                # Keyword args are just anything left.
                d_kwargs = rand_info_copy[key]
                # Randomize (adding to the original values).
                randomized_values[key] += distrib(*d_args, **d_kwargs)
        return randomized_values

    def _setup_disturbances(self):
        """Creates attributes and action spaces for the disturbances."""
        # Default: no passive disturbances.
        self.disturbances = {}
        if self.DISTURBANCES is not None:
            print(self.DISTURBANCES)
            for mode, disturb_specs in self.DISTURBANCES.items():
                assert (
                    mode in self.DISTURBANCE_MODES
                ), "[ERROR] in BenchmarkEnv._setup_disturbances(), disturbance mode not available."
                mode_shared_args = self.DISTURBANCE_MODES[mode]
                self.disturbances[mode] = create_disturbance_list(
                    disturb_specs, mode_shared_args, self
                )

    def _setup_constraints(self):
        """Creates a list of constraints as an attribute."""
        self.constraints = None
        self.num_constraints = 0
        if self.CONSTRAINTS is not None:
            self.constraints = create_constraint_list(
                self.CONSTRAINTS, self.AVAILABLE_CONSTRAINTS, self
            )
            self.num_constraints = self.constraints.num_constraints

    def before_reset(self):
        """Pre-processing before calling `.reset()`."""
        # IROS 2022 - Re-seed on reset.
        if self.RESEED_ON_RESET:
            self.seed(self.RND_SEED)

        # Housekeeping variables.
        self.initial_reset = True
        self.pyb_step_counter = 0
        self.ctrl_step_counter = 0
        self.current_raw_input_action = None
        self.current_preprocessed_action = None
        # Reset the disturbances.
        for mode in self.disturbances.keys():
            self.disturbances[mode].reset(self)
        if self.adversary_disturbance is not None:
            self.adv_action = None

    def after_reset(self, obs, info):
        """Post-processing after calling `.reset()`."""
        # Add initial constraint info (no action/input yet, so only state-based constraints)
        if self.constraints is not None:
            info["constraint_values"] = self.constraints.get_values(self, only_state=True)
        return obs, info

    def before_step(self, action):
        """Pre-processing before calling `.step()`."""
        # Sanity check (reset at least once).
        self._check_initial_reset()
        # Save the raw input action.
        self.current_raw_input_action = action
        # Pre-process/clip the action
        processed_action = self._preprocess_control(action)
        return processed_action

    def after_step(self, obs, rew, done, info):
        """Post-processing after calling `.step()`."""
        # Increment counters
        self.pyb_step_counter += self.PYB_STEPS_PER_CTRL
        self.ctrl_step_counter += 1

        # Terminate when (any) constraint is violated.
        # here we cache the constraint values `c_value`, so we only evaluate the constraints once,
        # but use it in 1) info dict; 2) check constraint violation; 3) check near constraint violation/almost active.
        # it also allows stateful constraint, where evaluation/update should only be done once per time step.
        c_value = None
        if self.constraints is not None:
            c_value = self.constraints.get_values(self)
            info["constraint_values"] = c_value
            if self.constraints.is_violated(self, c_value=c_value):
                # IROS 2022 - Constrain violation flag for reward.
                self.cnstr_violation = True

                info["constraint_violation"] = 1
                if self.DONE_ON_VIOLATION:
                    done = True
            else:
                # IROS 2022 - Constrain violation flag for reward.
                self.cnstr_violation = False

                info["constraint_violation"] = 0

        # Apply penalized reward when close to constraint violation
        if self.COST == Cost.RL_REWARD:
            if (
                self.constraints is not None
                and self.use_constraint_penalty
                and self.constraints.is_almost_active(self, c_value=c_value)
            ):
                rew += self.constraint_penalty

        # Terminate when reaching time limit,
        # but distinguish between done due to true termination or time limit reached
        if self.ctrl_step_counter >= self.CTRL_STEPS:
            info["TimeLimit.truncated"] = not done
            done = True
        return obs, rew, done, info
