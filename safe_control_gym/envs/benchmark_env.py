"""Base environment class module.

This module also contains enumerations for cost functions, tasks, disturbances, and quadrotor types.

"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import gymnasium
from gymnasium.utils import seeding
import copy
import math

from safe_control_gym.envs.constraints import GENERAL_CONSTRAINTS
from safe_control_gym.envs.constraints import create_constraint_list
from safe_control_gym.envs.disturbances import create_disturbance_list


@dataclass
class SimSettings:
    """Simulation settings dataclass."""

    sim_freq: int = 500
    ctrl_freq: int = 500
    gui: bool = False

    def __post_init__(self):
        assert self.sim_freq % self.ctrl_freq == 0, "sim_freq must be divisible by ctrl_freq."


class BenchmarkEnv(gymnasium.Env):
    """Benchmark environment base class."""

    def __init__(
        self,
        gui: bool = False,
        sim_freq: int = 500,
        ctrl_freq: int = 500,
        episode_len_sec: int = 5,
        init_state=None,
        randomized_init: bool = True,
        init_state_randomization_info=None,
        inertial_prop=None,
        randomized_inertial_prop: bool = False,
        inertial_prop_randomization_info=None,
        constraints=None,
        disturbances=None,
        seed: int | None = None,
        reseed: bool = False,
    ):
        """Initialization method for BenchmarkEnv.

        Args:
            gui: Option to show PyBullet's GUI.
            sim_freq: The frequency at which PyBullet steps (a multiple of ctrl_freq).
            ctrl_freq: The frequency at which the environment steps.
            init_state: The initial state of the environment
            randomized_init: Flag to randomize the initial state.
            init_state_randomization_info: A dictionary with information used to randomize the
                initial state.
            inertial_prop: The ground truth inertial properties of the environment.
            randomized_inertial_prop: Flag to randomize the inertial properties.
            inertial_prop_randomization_info: A dictionary with information used to randomize the
                inert. properties.
            constraints: Dictionary to specify the constraints being used.
            disturbances: Dictionary to specify disturbances being used.
            seed: Seed for the random number generator.
            reseed: Option to reseed the environment on each reset with the same seed.
        """
        self.sim_settings = SimSettings(sim_freq, ctrl_freq, gui)

        self._init_state = init_state
        self.EPISODE_LEN_SEC = episode_len_sec

        self._init_state = init_state
        self._init_state_randomization = init_state_randomization_info
        self._enable_init_state_randomization = randomized_init

        self._inertial = inertial_prop
        self._inertial_randomization = inertial_prop_randomization_info
        self._enable_inertial_randomization = randomized_inertial_prop

        # Create action and observation spaces.
        self._set_action_space()
        self._set_observation_space()
        self.state_space = self.observation_space  # TODO: Remove this, not always correct
        # Store action (input) and observation spaces dimensions.
        self.Q = np.eye(self.observation_space.shape[0])
        self.R = np.eye(self.action_space.shape[0])
        # Set constraint info.
        self.constraints = None
        if constraints is not None:
            self.constraints = create_constraint_list(constraints, GENERAL_CONSTRAINTS, self)
        # Set disturbance info.
        self._disturbance_config = disturbances
        self.disturbances = self._setup_disturbances(disturbances)
        # Default seed None means pure randomness/no seeding.
        self.seed(seed)
        # IROS 2022 - Save random seed for re-seeding.
        self._seed = seed
        self._reseed_on_reset = reseed

    def _set_action_space(self):
        a_low = self.KF * (self.PWM2RPM_SCALE * self.MIN_PWM + self.PWM2RPM_CONST) ** 2
        a_high = self.KF * (self.PWM2RPM_SCALE * self.MAX_PWM + self.PWM2RPM_CONST) ** 2
        self.action_space = gymnasium.spaces.Box(low=a_low, high=a_high, shape=(4,))

    def _set_observation_space(self):
        self.x_threshold = 5
        self.y_threshold = 5
        self.z_threshold = 2.5
        self.phi_threshold_radians = 85 * math.pi / 180
        self.theta_threshold_radians = 85 * math.pi / 180
        self.psi_threshold_radians = 180 * math.pi / 180  # Do not bound yaw.

        # Define obs/state bounds, labels and units.
        # obs/state = {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r}.
        low = np.array(
            [
                -self.x_threshold,
                -np.finfo(np.float32).max,
                -self.y_threshold,
                -np.finfo(np.float32).max,
                self.GROUND_PLANE_Z,
                -np.finfo(np.float32).max,
                -self.phi_threshold_radians,
                -self.theta_threshold_radians,
                -self.psi_threshold_radians,
                -np.finfo(np.float32).max,
                -np.finfo(np.float32).max,
                -np.finfo(np.float32).max,
            ],
            np.float32,
        )
        high = np.array(
            [
                self.x_threshold,
                np.finfo(np.float32).max,
                self.y_threshold,
                np.finfo(np.float32).max,
                self.z_threshold,
                np.finfo(np.float32).max,
                self.phi_threshold_radians,
                self.theta_threshold_radians,
                self.psi_threshold_radians,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
            ],
            np.float32,
        )
        self.STATE_LABELS = [
            "x",
            "x_dot",
            "y",
            "y_dot",
            "z",
            "z_dot",
            "phi",
            "theta",
            "psi",
            "p",
            "q",
            "r",
        ]
        self.STATE_UNITS = [
            "m",
            "m/s",
            "m",
            "m/s",
            "m",
            "m/s",
            "rad",
            "rad",
            "rad",
            "rad/s",
            "rad/s",
            "rad/s",
        ]

        # Define the state space for the dynamics.
        self.state_space = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)

        # Define obs space exposed to the controller.
        # Note how the obs space can differ from state space (i.e. augmented with the next reference states for RL)
        self.observation_space = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)

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

    def _setup_disturbances(self, disturbances: dict | None = None):
        """Creates attributes and action spaces for the disturbances."""
        # Default: no passive disturbances.
        if disturbances is None:
            return
        dist = {}
        modes = {
            "observation": {"dim": self.observation_space.shape[0]},
            "action": {"dim": self.action_space.shape[0]},
            "dynamics": {"dim": 3},
        }
        for mode, spec in disturbances.items():
            assert mode in modes, "Disturbance mode not available."
            dist[mode] = create_disturbance_list(spec, modes[mode], self)
        return dist

    @property
    def n_constraints(self):
        return 0 if self.constraints is None else self.constraints.num_constraints

    def before_reset(self):
        """Pre-processing before calling `.reset()`."""
        # IROS 2022 - Re-seed on reset.
        if self._reseed_on_reset:
            self.seed(self._seed)

        # Housekeeping variables.
        self.pyb_step_counter = 0
        self.ctrl_step_counter = 0
        self.current_raw_input_action = None
        self.current_preprocessed_action = None
        # Reset the disturbances.
        for mode in self.disturbances.keys():
            self.disturbances[mode].reset(self)

    def after_reset(self, obs, info):
        """Post-processing after calling `.reset()`."""
        # Add initial constraint info (no action/input yet, so only state-based constraints)
        if self.constraints is not None:
            info["constraint_values"] = self.constraints.get_values(self, only_state=True)
        return obs, info

    def before_step(self, action):
        """Pre-processing before calling `.step()`."""
        # Save the raw input action.
        self.current_raw_input_action = action
        # Pre-process/clip the action
        processed_action = self._preprocess_control(action)
        return processed_action

    def after_step(self, obs, rew, done, info):
        """Post-processing after calling `.step()`."""
        # Increment counters
        self.pyb_step_counter += self.sim_settings.sim_freq // self.sim_settings.ctrl_freq
        self.ctrl_step_counter += 1

        # Terminate when (any) constraint is violated.
        # here we cache the constraint values `c_value`, so we only evaluate the constraints once,
        # but use it in 1) info dict; 2) check constraint violation; 3) check near constraint violation/almost active.
        # it also allows stateful constraint, where evaluation/update should only be done once per time step.
        c_value = None
        if self.constraints is not None:
            c_value = self.constraints.get_values(self)
            info["constraint_values"] = c_value
            info["constraint_violation"] = self.constraints.is_violated(self, c_value=c_value)
        # Terminate when reaching time limit,
        # but distinguish between done due to true termination or time limit reached
        if self.ctrl_step_counter >= self.EPISODE_LEN_SEC * self.sim_settings.ctrl_freq:
            info["TimeLimit.truncated"] = not done
            done = True
        return obs, rew, done, info
