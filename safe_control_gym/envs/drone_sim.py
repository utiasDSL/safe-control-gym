"""Base environment class module.

This module also contains enumerations for cost functions, tasks, disturbances, and quadrotor types.

"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
import copy
import math

from safe_control_gym.envs.constraints import ConstraintList
from safe_control_gym.envs.disturbances import create_disturbance_list
from safe_control_gym.envs.drone import Drone


@dataclass
class SimSettings:
    """Simulation settings dataclass."""

    sim_freq: int = 500
    ctrl_freq: int = 500
    gui: bool = False

    def __post_init__(self):
        assert self.sim_freq % self.ctrl_freq == 0, "sim_freq must be divisible by ctrl_freq."


class DroneSim(gymnasium.Env):
    """Drone simulation based on gym-pybullet-drones."""

    def __init__(
        self,
        sim_freq: int = 500,
        ctrl_freq: int = 500,
        constraints=None,
        disturbances=None,
        gui: bool = False,
        seed: int | None = None,
        reseed: bool = False,
    ):
        """Initialization method for BenchmarkEnv.

        Args:
            gui: Option to show PyBullet's GUI.
            sim_freq: The frequency at which PyBullet steps (a multiple of ctrl_freq).
            ctrl_freq: The frequency at which the environment steps.
            constraints: Dictionary to specify the constraints being used.
            disturbances: Dictionary to specify disturbances being used.
            seed: Seed for the random number generator.
            reseed: Option to reseed the environment on each reset with the same seed.
        """
        self.sim_settings = SimSettings(sim_freq, ctrl_freq, gui)
        self.drone = Drone(controller="mellinger")

        # Create action, observation and state spaces.
        min_thrust, max_thrust = self.drone.params.min_thrust, self.drone.params.max_thrust
        self.action_space = spaces.Box(low=min_thrust, high=max_thrust, shape=(4,))

        # pos: x, y, z in meters
        # rpy: roll, pitch, yaw in radians
        # vel: vx, vy, vz in m/s
        # ang_vel: p, q, r in rad/s
        rpy_max = np.array([85 / 180 * np.pi, 85 / 180 * np.pi, np.pi], np.float32)  # Yaw unbounded
        max_flt = np.full(3, np.finfo(np.float32).max, np.float32)
        self.observation_space = spaces.Dict(
            {
                "pos": spaces.Box(low=np.array([-5, -5, 0]), high=np.array([5, 5, 2.5])),
                "rpy": spaces.Box(low=-rpy_max, high=rpy_max),
                "vel": spaces.Box(low=-max_flt, high=max_flt),
                "ang_vel": spaces.Box(low=-max_flt, high=max_flt),
            }
        )
        self.state_space = self.observation_space
        self.constraints = ConstraintList.from_specs(
            self.state_space, self.action_space, constraints
        )
        # Set disturbance info.
        self._disturbance_config = disturbances
        self.disturbances = self._setup_disturbances(disturbances)
        # Default seed None means pure randomness/no seeding.
        self.seed(seed)
        # IROS 2022 - Save random seed for re-seeding.
        self._seed = seed
        self._reseed_on_reset = reseed

    def seed(self, seed=None):
        """Set up a random number generator for a given seed.

        Remember to seed all random generators, currently in
        - env
        - action_space
        - disturbances

        """
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        for d in self.disturbances.values():
            d.seed(self)
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
            "observation": {"dim": spaces.flatdim(self.observation_space)},
            "action": {"dim": spaces.flatdim(self.action_space)},
            "dynamics": {"dim": 3},
        }
        for mode, spec in disturbances.items():
            assert mode in modes, "Disturbance mode not available."
            dist[mode] = create_disturbance_list(spec, modes[mode], self)
        return dist

    @property
    def n_constraints(self):
        return 0 if self.constraints is None else self.constraints.n_constraints

    def before_reset(self):
        """Pre-processing before calling `.reset()`."""
        if self._reseed_on_reset:  # IROS 2022 - Re-seed on reset.
            self.seed(self._seed)

        # Housekeeping variables.
        self.pyb_step_counter = 0
        self.ctrl_step_counter = 0
        self.current_raw_input_action = None
        self.current_preprocessed_action = None
        # Reset the disturbances.
        for mode in self.disturbances.keys():
            self.disturbances[mode].reset(self)

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
            c_value = self.constraints.value(self.state, self.current_raw_input_action)
            info["constraint_values"] = c_value
            info["constraint_violation"] = self.constraints.is_violated(
                self.state, self.current_raw_input_action, c_value=c_value
            )
        # Terminate when reaching time limit,
        # but distinguish between done due to true termination or time limit reached
        return obs, rew, done, info
