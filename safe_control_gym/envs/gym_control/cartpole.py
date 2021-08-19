"""Cartpole environment using PyBullet physics.

Classic cart-pole system implemented by Rich Sutton et al.
    * http://incompleteideas.net/sutton/book/code/pole.c

Reference implementations:
    * https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    * https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/cartpole_bullet.py

Todo:
    * Should environment record all past values up to the last reset?

"""
import os
import math
import xml.etree.ElementTree as etxml
from copy import deepcopy
import casadi as cs
import numpy as np
import pybullet as p
from gym import spaces

from safe_control_gym.envs.benchmark_env import BenchmarkEnv, Cost, Task
from safe_control_gym.envs.disturbances import DisturbanceList, DISTURBANCE_TYPES
from safe_control_gym.envs.constraints import create_ConstraintList_from_dict, GENERAL_CONSTRAINTS
from safe_control_gym.envs.gym_control.cartpole_constraints import CartPoleStateConstraint, CartPoleInputConstraint, \
    CartPoleSymmetricStateConstraint, CartPoleBoundConstraint
from safe_control_gym.math_and_models.symbolic_systems import SymbolicModel


class CartPole(BenchmarkEnv):
    """Cartpole environment task.

    Including symbolic model, constraints, randomization, adversarial disturbances,
    multiple cost functions, stabilization and trajectory tracking references.

    task_config:
        info_in_reset: True
        randomized_inertial_prop: True
        inertial_prop_randomization_info:
            pole_length:
                distrib: choice
                args: [[1,5,10]]
            pole_mass:
                distrib: uniform
                low: 1
                high: 5
        constraints:
            cartpole_state: {} # to use default state constraints
            cartpole_input:
                low: -0.2 # to modify the default values from environment
                high: 0.2
        done_on_violation: True
        disturbances:
            observation:
                white_noise:
                    sigma: 1.0
            action:
                white_noise:
                    sigma: 1.0
            dynamics: 
                white_noise:
                    sigma: 1.0
        adversary_disturbance: dynamics
        adversary_disturbance_scale: 0.01

    """
    URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "cartpole_template.urdf")

    AVAILABLE_CONSTRAINTS = {
        "cartpole_state": CartPoleStateConstraint,
        "cartpole_input": CartPoleInputConstraint,
        "abs_bound": CartPoleSymmetricStateConstraint,
        "cartpole_bound": CartPoleBoundConstraint,
    }
    AVAILABLE_CONSTRAINTS.update(deepcopy(GENERAL_CONSTRAINTS))

    DISTURBANCE_MODES = {"observation": {"dim": 4}, "action": {"dim": 1}, "dynamics": {"dim": 2}}

    INERTIAL_PROP_RAND_INFO = {
        "pole_length": {  # 1
            "distrib": "choice",
            "args": [[1, 5, 10]]
        },
        "cart_mass": {  # 1
            "distrib": "uniform",
            "low": 0.5,
            "high": 1.5
        },
        "pole_mass": {  # 0.1
            "distrib": "uniform",
            "low": 0.05,
            "high": 0.15
        }
    }

    INIT_STATE_RAND_INFO = {
        "init_x": {
            "distrib": "uniform",
            "low": -0.05,
            "high": 0.05
        },
        "init_x_dot": {
            "distrib": "uniform",
            "low": -0.05,
            "high": 0.05
        },
        "init_theta": {
            "distrib": "uniform",
            "low": -0.05,
            "high": 0.05
        },
        "init_theta_dot": {
            "distrib": "uniform",
            "low": -0.05,
            "high": 0.05
        }
    }

    TASK_INFO = {
        "stabilization_goal": [0],
        "trajectory_type": "circle",
        "num_cycles": 1,
        "trajectory_plane": "zx",
        "trajectory_position_offset": [0, 0],
        "trajectory_scale": 0.2
    }

    def __init__(self,
                 seed=None,
                 output_dir=None,
                 info_in_reset: bool = False,
                 ctrl_freq: int = 50,
                 pyb_freq: int = 50,
                 gui: bool = False,
                 normalized_rl_action_space: bool = False,
                 init_state=None,
                 randomized_init: bool = True,
                 init_state_randomization_info=None,
                 prior_prop=None,
                 inertial_prop=None,
                 randomized_inertial_prop: bool = False,
                 inertial_prop_randomization_info=None,
                 task: Task = Task.STABILIZATION,
                 task_info=None,
                 episode_len_sec: int = 5,
                 cost: Cost = Cost.RL_REWARD,
                 disturbances=None,
                 adversary_disturbance=None,
                 adversary_disturbance_scale=0.01,
                 constraints=None,
                 done_on_violation: bool = False,
                 use_constraint_penalty=False,
                 constraint_penalty=-1,
                 verbose: bool = False):
        """Initialize a cartpole environment.

        Args:
            seed (int, optional): Seed for the random number generator.
            output_dir (str, optional): path to directory to save any env outputs.
            info_in_reset (bool, optional): Whether .reset() returns a dictionary with the 
                environment's symbolic model.
            ctrl_freq (int, optional): The frequency at which the environment steps.
            pyb_freq (int, optional): The frequency at which PyBullet steps (a multiple of ctrl_freq).
            gui (bool, optional): Whether to show PyBullet's GUI.
            normalized_rl_action_space (bool, optional): Whether to normalize the action space.
            init_state  (ndarray/dict, optional): The initial state of the environment 
                (x, x_dot, theta, theta_dot).
            randomized_init (bool, optional): Whether to randomize the initial state.
            init_state_randomization_info (dict, optional): A dictionary with information used to 
                randomize the initial state.
            prior_prop (dict, optional): The prior inertial properties of the environment.
            inertial_prop (dict, optional): The ground truth inertial properties of the environment.
            randomized_inertial_prop (bool, optional): Whether to randomize the inert. properties.
            inertial_prop_randomization_info (dict, optional): A dictionary with information used 
                to randomize the inert. properties.
            task: (Task, optional): The environment's task (stabilization or traj. tracking).
            task_info (dict, optional): A dictionary with the information used to generate the 
                task X and U references.
            episode_len_sec (int, optional): Maximum episode duration in seconds.
            cost: (Cost, optional): Cost function choice used to compute the reward in .step().
            disturbances (dict, optional): Dictionary to specify disturbances being used. 
            adversary_disturbance (str, optional): if to use adversary/external disturbance.
            adversary_disturbance_scale (float, optional): parameterizes magnitude of adversary disturbance.
            constraints (Dict, optional): Dictionary to specify the constraints being used.
            done_on_violation (bool, optional): Whether to return done==True on a constraint violation.
            use_constraint_penalty (bool, optional): if to use shaped reward to penalize potential 
                constraint violation.
            constraint_penalty (float, optional): constraint penalty cost for reward shaping. 
            verbose (bool, optional): If to suppress environment print statetments.

        """
        # Set timing constants.
        self.NORMALIZED_RL_ACTION_SPACE = normalized_rl_action_space
        self.CTRL_FREQ = ctrl_freq
        self.PYB_FREQ = pyb_freq
        if self.PYB_FREQ % self.CTRL_FREQ != 0:
            raise ValueError("[ERROR] in CartPole.__init__(), pyb_freq is not divisible by env_freq.")
        self.CTRL_TIMESTEP = 1. / self.CTRL_FREQ
        self.PYB_TIMESTEP = 1. / self.PYB_FREQ

        # Set GUI and rendering constants.
        self.GUI = gui
        self.RENDER_HEIGHT = int(200)
        self.RENDER_WIDTH = int(320)

        # Set the initial state.
        if init_state is None:
            self.INIT_X, self.INIT_X_DOT, self.INIT_THETA, self.INIT_THETA_DOT = np.zeros(4)
        elif isinstance(init_state, np.ndarray):
            self.INIT_X, self.INIT_X_DOT, self.INIT_THETA, self.INIT_THETA_DOT = init_state
        elif isinstance(init_state, dict):
            self.INIT_X = init_state.get("init_x", 0)
            self.INIT_X_DOT = init_state.get("init_x_dot", 0)
            self.INIT_THETA = init_state.get("init_theta", 0)
            self.INIT_THETA_DOT = init_state.get("init_theta_dot", 0)
        else:
            raise ValueError("[ERROR] in CartPole.__init__(), init_state incorrect format.")
        # Decide whether to randomize the initial state and how (see info dictionary).
        self.RANDOMIZED_INIT = randomized_init
        if init_state_randomization_info is not None:
            self.INIT_STATE_RAND_INFO = init_state_randomization_info

        # Get physical properties from URDF (as default parameters).
        self.GRAVITY_ACC = 9.8
        EFFECTIVE_POLE_LENGTH, POLE_MASS, CART_MASS = self._parse_urdf_parameters(self.URDF_PATH)

        # Store prior parameters.
        if prior_prop is None:
            self.PRIOR_EFFECTIVE_POLE_LENGTH = EFFECTIVE_POLE_LENGTH
            self.PRIOR_POLE_MASS = POLE_MASS
            self.PRIOR_CART_MASS = CART_MASS
        elif isinstance(prior_prop, dict):
            self.PRIOR_EFFECTIVE_POLE_LENGTH = prior_prop.get("pole_length", EFFECTIVE_POLE_LENGTH)
            self.PRIOR_POLE_MASS = prior_prop.get("pole_mass", POLE_MASS)
            self.PRIOR_CART_MASS = prior_prop.get("cart_mass", CART_MASS)
        else:
            raise ValueError("[ERROR] in CartPole.__init__(), prior_prop incorrect format.")

        # Store ground truth parameters.
        if inertial_prop is None:
            self.EFFECTIVE_POLE_LENGTH = EFFECTIVE_POLE_LENGTH
            self.POLE_MASS = POLE_MASS
            self.CART_MASS = CART_MASS
        elif isinstance(inertial_prop, dict):
            self.EFFECTIVE_POLE_LENGTH = inertial_prop.get("pole_length", EFFECTIVE_POLE_LENGTH)
            self.POLE_MASS = inertial_prop.get("pole_mass", POLE_MASS)
            self.CART_MASS = inertial_prop.get("cart_mass", CART_MASS)
        else:
            raise ValueError("[ERROR] in CartPole.__init__(), inertial_prop incorrect format.")
        # Domain randomization on parameters.
        self.RANDOMIZED_INERTIAL_PROP = randomized_inertial_prop
        if inertial_prop_randomization_info is not None:
            self.INERTIAL_PROP_RAND_INFO = inertial_prop_randomization_info

        # Store disturbance info.
        self.DISTURBANCES = disturbances
        self.adversary_disturbance = adversary_disturbance
        self.adversary_disturbance_scale = adversary_disturbance_scale

        # Store constraint info
        self.CONSTRAINTS = constraints
        self.DONE_ON_VIOLATION = done_on_violation
        self.use_constraint_penalty = use_constraint_penalty
        self.constraint_penalty = constraint_penalty

        self.VERBOSE = verbose

        # Set up action and observation space.
        self._set_action_space()
        self._set_observation_space()
        # Create PyBullet client connection.
        self.PYB_CLIENT = -1
        if self.GUI:
            self.PYB_CLIENT = p.connect(p.GUI)
        else:
            self.PYB_CLIENT = p.connect(p.DIRECT)
        # disable urdf caching for randomization via reloading urdf
        p.setPhysicsEngineParameter(enableFileCaching=0)

        # Call BenchmarkEnv constructor.
        super().__init__(seed=seed, output_dir=output_dir, info_in_reset=info_in_reset, episode_len_sec=episode_len_sec, cost=cost)

        # Create X_GOAL and U_GOAL references for the assigned task.
        self.TASK = Task(task)
        if task_info is not None:
            self.TASK_INFO = task_info
        self.U_GOAL = np.zeros(1)
        if self.TASK == Task.STABILIZATION:
            self.X_GOAL = np.hstack([self.TASK_INFO["stabilization_goal"][0], 0., 0., 0.])  # x = {x, x_dot, theta, theta_dot}.
        elif self.TASK == Task.TRAJ_TRACKING:
            POS_REF, \
            VEL_REF, \
            SPEED = self._generate_trajectory(traj_type=self.TASK_INFO["trajectory_type"],
                                              traj_length=self.EPISODE_LEN_SEC,
                                              num_cycles=self.TASK_INFO["num_cycles"],
                                              traj_plane=self.TASK_INFO["trajectory_plane"],
                                              position_offset=np.array(self.TASK_INFO["trajectory_position_offset"]) + np.array([0, self.INIT_X]),
                                              scaling=self.TASK_INFO["trajectory_scale"],
                                              sample_time=self.CTRL_TIMESTEP
                                              )

            self.X_GOAL = np.vstack([
                POS_REF[:, 0],  # TODO Add offset?
                VEL_REF[:, 0],
                np.zeros(POS_REF.shape[0]),
                np.zeros(VEL_REF.shape[0])
            ]).transpose()

    @property
    def control_step_counter(self):
        """Smiliar to `step_counter` but on control frequency."""
        return self.step_counter // (self.PYB_FREQ // self.CTRL_FREQ)

    def step(self, action):
        """Advances the environment by one control step.

        Args:
            action (ndarray): the action applied to the environment for the step.

        Returns:
            ndarray: The state of the environment after the step.
            float: The scalar reward/cost of the step.
            bool: Whether the conditions for the end of an episode are met in the step.
            dict: A dictionary with information about the constraints evaluations and violations.

        """
        # Sanity check (reset at least once).
        self._check_initial_reset()
        # Save the raw input action.
        self.current_raw_input_action = action
        # Pre-process/clip the action
        force = self._preprocess_control(action)
        # Save the actual input.
        self.current_preprocessed_action = force
        # Advance the simulation.
        self._advance_simulation(force)
        # Update the state.
        self.state = np.hstack(
            (p.getJointState(self.CARTPOLE_ID, jointIndex=0,
                             physicsClientId=self.PYB_CLIENT)[0:2], p.getJointState(self.CARTPOLE_ID, jointIndex=1, physicsClientId=self.PYB_CLIENT)[0:2]))
        # Standard Gym return.
        obs = self._get_observation()
        rew = self._get_reward()
        done = self._get_done()
        info = self._get_info()
        return obs, rew, done, info

    def reset(self):
        """(Re-)initializes the environment to start an episode.

        Mandatory to call at least once after __init__().

        Returns:
            ndarray: The initial state of the environment.
            dict: A dictionary with information about the dynamics and constraints symbolic models.

        """
        # Housekeeping variables.
        self.initial_reset = True
        self.step_counter = 0
        self.current_raw_input_action = None
        self.current_preprocessed_action = None
        if self.adversary_disturbance is not None:
            self.adv_action = None

        # PyBullet simulation reset.
        p.resetSimulation(physicsClientId=self.PYB_CLIENT)
        p.setGravity(0, 0, -self.GRAVITY_ACC, physicsClientId=self.PYB_CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.PYB_CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.PYB_CLIENT)

        # Choose randomized or deterministic inertial properties.
        prop_values = {"pole_length": self.EFFECTIVE_POLE_LENGTH, "cart_mass": self.CART_MASS, "pole_mass": self.POLE_MASS}
        if self.RANDOMIZED_INERTIAL_PROP:
            prop_values = self._randomize_values_by_info(prop_values, self.INERTIAL_PROP_RAND_INFO)
            if any(phy_quantity < 0 for phy_quantity in prop_values.values()):
                raise ValueError("[ERROR] in CartPole.reset(), negative randomized inertial properties.")

        self.OVERRIDDEN_EFFECTIVE_POLE_LENGTH = prop_values["pole_length"]
        self.OVERRIDDEN_CART_MASS = prop_values["cart_mass"]
        self.OVERRIDDEN_POLE_MASS = prop_values["pole_mass"]

        # refer to `slender rod` in https://en.wikipedia.org/wiki/List_of_moments_of_inertia
        OVERRIDDEN_POLE_INERTIA = (1 / 12) * self.OVERRIDDEN_POLE_MASS * (2 * self.OVERRIDDEN_EFFECTIVE_POLE_LENGTH)**2

        # Load the cartpole with new urdf.
        override_urdf_tree = self._create_urdf(self.URDF_PATH, length=self.OVERRIDDEN_EFFECTIVE_POLE_LENGTH, inertia=OVERRIDDEN_POLE_INERTIA)
        self.override_path = os.path.join(self.output_dir, "pid-{}_id-{}_cartpole.urdf".format(os.getpid(), self.id))
        override_urdf_tree.write(self.override_path)

        self.CARTPOLE_ID = p.loadURDF(
            self.override_path,
            basePosition=[0, 0, 0],
            # flags = p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.PYB_CLIENT)
        # remove cache file after loading into pybullet
        os.remove(self.override_path)

        # Cartpole settings.
        for i in [-1, 0, 1]:  # Slider, cart, and pole.
            p.changeDynamics(self.CARTPOLE_ID, linkIndex=i, linearDamping=0, angularDamping=0, physicsClientId=self.PYB_CLIENT)
        for i in [0, 1]:  # Slider-to-cart and cart-to-pole joints.
            p.setJointMotorControl2(self.CARTPOLE_ID, jointIndex=i, controlMode=p.VELOCITY_CONTROL, force=0, physicsClientId=self.PYB_CLIENT)
        # Override inertial properties.
        p.changeDynamics(
            self.CARTPOLE_ID,
            linkIndex=0,  # Cart.
            mass=self.OVERRIDDEN_CART_MASS,
            physicsClientId=self.PYB_CLIENT)
        p.changeDynamics(
            self.CARTPOLE_ID,
            linkIndex=1,  # Pole.
            mass=self.OVERRIDDEN_POLE_MASS,
            physicsClientId=self.PYB_CLIENT)

        # Randomize initial state.
        init_values = {"init_x": self.INIT_X, "init_x_dot": self.INIT_X_DOT, "init_theta": self.INIT_THETA, "init_theta_dot": self.INIT_THETA_DOT}
        if self.RANDOMIZED_INIT:
            init_values = self._randomize_values_by_info(init_values, self.INIT_STATE_RAND_INFO)

        OVERRIDDEN_INIT_X = init_values["init_x"]
        OVERRIDDEN_INIT_X_DOT = init_values["init_x_dot"]
        OVERRIDDEN_INIT_THETA = init_values["init_theta"]
        OVERRIDDEN_INIT_THETA_DOT = init_values["init_theta_dot"]

        p.resetJointState(
            self.CARTPOLE_ID,
            jointIndex=0,  # Slider-to-cart joint.
            targetValue=OVERRIDDEN_INIT_X,
            targetVelocity=OVERRIDDEN_INIT_X_DOT,
            physicsClientId=self.PYB_CLIENT)
        p.resetJointState(
            self.CARTPOLE_ID,
            jointIndex=1,  # Cart-to-pole joints.
            targetValue=OVERRIDDEN_INIT_THETA,
            targetVelocity=OVERRIDDEN_INIT_THETA_DOT,
            physicsClientId=self.PYB_CLIENT)

        # Compute state (x, x_dot, theta, theta_dot).
        self.state = np.hstack(
            (p.getJointState(self.CARTPOLE_ID, jointIndex=0,
                             physicsClientId=self.PYB_CLIENT)[0:2], p.getJointState(self.CARTPOLE_ID, jointIndex=1, physicsClientId=self.PYB_CLIENT)[0:2]))

        # Debug visualization if GUI enabled
        self.line = None

        # Return either an observation and dictionary or just the observation.
        if self.INFO_IN_RESET:
            return self._get_observation(), self._get_reset_info()
        return self._get_observation()

    def render(self, mode='human'):
        """Retrieves a frame from PyBullet rendering.

        Args:
            mode (str): Unused.

        Returns:
            ndarray: A multidimensional array with the RGB frame captured by PyBullet's camera.

        """
        if self.PYB_CLIENT >= 0:
            VIEW_MATRIX = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0],
                                                              distance=2,
                                                              yaw=0,
                                                              pitch=0.3,
                                                              roll=0,
                                                              upAxisIndex=2,
                                                              physicsClientId=self.PYB_CLIENT)
            PROJ_MATRIX = p.computeProjectionMatrixFOV(fov=60,
                                                       aspect=float(self.RENDER_WIDTH) / self.RENDER_HEIGHT,
                                                       nearVal=0.1,
                                                       farVal=100.0,
                                                       physicsClientId=self.PYB_CLIENT)
            (w, h, rgb, _, _) = p.getCameraImage(width=self.RENDER_WIDTH,
                                                 height=self.RENDER_HEIGHT,
                                                 renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                                 viewMatrix=VIEW_MATRIX,
                                                 projectionMatrix=PROJ_MATRIX,
                                                 physicsClientId=self.PYB_CLIENT)
        # Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA').show()
        return np.reshape(rgb, (h, w, 4))

    def close(self):
        """Clean up the environment and PyBullet connection."""
        if self.PYB_CLIENT >= 0:
            p.disconnect(physicsClientId=self.PYB_CLIENT)
        self.PYB_CLIENT = -1

    def set_adversary_control(self, action):
        """Sets disturbance by an adversary controller, called before (each) step()."""
        if self.adversary_disturbance is not None:
            clipped_adv_action = np.clip(action, self.adversary_action_space.low, self.adversary_action_space.high)
            self.adv_action = clipped_adv_action * self.adversary_disturbance_scale

    def _setup_disturbances(self):
        """Sets up scaling and actions space for an adversarial disturbance."""
        # default no passive disturbance
        self.disturbances = {}

        if self.DISTURBANCES is not None:
            for mode, disturbs in self.DISTURBANCES.items():
                assert mode in self.DISTURBANCE_MODES, "[ERROR] in Cartpole._setup_disturbances(), disturbance mode not available."
                disturb_list = []
                shared_args = self.DISTURBANCE_MODES[mode]

                # each disturbance for this mode
                for name, cfg in disturbs.items():
                    assert name in DISTURBANCE_TYPES, "[ERROR] in Cartpole._setup_disturbances(), disturbance type not available."
                    disturb_cls = DISTURBANCE_TYPES[name]
                    disturb = disturb_cls(self, **shared_args, **cfg)
                    disturb_list.append(disturb)

                # combine as one for this mode
                self.disturbances[mode] = DisturbanceList(disturb_list)

        # adversary disturbance (set from outside of env, non-passive)
        if self.adversary_disturbance is not None:
            assert self.adversary_disturbance in self.DISTURBANCE_MODES, "[ERROR] in Cartpole._setup_disturbances()"

            shared_args = self.DISTURBANCE_MODES[self.adversary_disturbance]
            dim = shared_args["dim"]
            # TODO: symmetric space can be problematic for obs ?
            self.adversary_action_space = spaces.Box(low=-1, high=1, shape=(dim,))
            # adversary obs same as protagonist
            self.adversary_observation_space = self.observation_space

    def _setup_constraints(self):
        """Sets up a list (ConstraintList) of constraints."""
        self.constraints = None
        self.num_constraints = 0

        if self.CONSTRAINTS is not None:
            self.constraints = create_ConstraintList_from_dict(self.CONSTRAINTS, self.AVAILABLE_CONSTRAINTS, self)
            self.num_constraints = self.constraints.num_constraints

    def _setup_symbolic(self):
        """Creates symbolic (CasADi) models for dynamics, observation, and cost."""
        l, m, M = self.PRIOR_EFFECTIVE_POLE_LENGTH, self.PRIOR_POLE_MASS, self.PRIOR_CART_MASS
        Mm, ml = m + M, m * l
        g = self.GRAVITY_ACC
        dt = self.CTRL_TIMESTEP

        # Input variables.
        x = cs.MX.sym('x')
        x_dot = cs.MX.sym('x_dot')
        theta = cs.MX.sym('theta')
        theta_dot = cs.MX.sym('theta_dot')
        X = cs.vertcat(x, x_dot, theta, theta_dot)
        U = cs.MX.sym('U')
        nx = 4
        nu = 1

        # Dynamics.
        temp_factor = (U + ml * theta_dot**2 * cs.sin(theta)) / Mm
        theta_dot_dot = ((g * cs.sin(theta) - cs.cos(theta) * temp_factor) / (l * (4.0 / 3.0 - m * cs.cos(theta)**2 / Mm)))
        X_dot = cs.vertcat(x_dot, temp_factor - ml * theta_dot_dot * cs.cos(theta) / Mm, theta_dot, theta_dot_dot)

        # Observation.
        Y = cs.vertcat(x, x_dot, theta, theta_dot)

        # Define cost (quadratic form).
        Q = cs.MX.sym('Q', nx, nx)
        R = cs.MX.sym('R', nu, nu)
        Xr = cs.MX.sym('Xr', nx, 1)
        Ur = cs.MX.sym('Ur', nu, 1)
        cost_func = 0.5 * (X - Xr).T @ Q @ (X - Xr) + 0.5 * (U - Ur).T @ R @ (U - Ur)

        # Define dynamics and cost dictionaries.
        dynamics = {"dyn_eqn": X_dot, "obs_eqn": Y, "vars": {"X": X, "U": U}}
        cost = {"cost_func": cost_func, "vars": {"X": X, "U": U, "Xr": Xr, "Ur": Ur, "Q": Q, "R": R}}

        # Setup symbolic model.
        self.symbolic = SymbolicModel(dynamics=dynamics, cost=cost, dt=dt)

    def _set_action_space(self):
        """Returns the action space of the environment."""
        self.action_scale = 10
        self.action_threshold = 1 if self.NORMALIZED_RL_ACTION_SPACE else self.action_scale
        self.action_space = spaces.Box(low=-self.action_threshold, high=self.action_threshold, shape=(1,))

    def _set_observation_space(self):
        """Returns the observation space of the environment."""
        # Angle at which to fail the episode (0.20943951023931953).
        self.theta_threshold_radians = 12 * (2 * math.pi / 360)
        # NOTE: different value in PyBullet gym (0.4) and OpenAI gym (2.4).
        self.x_threshold = 2.4

        # Limit set to 2x: i.e. a failing observation is still within bounds.
        OBS_BOUND = np.array([self.x_threshold * 2, np.finfo(np.float32).max, self.theta_threshold_radians * 2, np.finfo(np.float32).max])
        self.observation_space = spaces.Box(-OBS_BOUND, OBS_BOUND, dtype=np.float32)

    def _preprocess_control(self, action):
        """Converts the raw action input into the one used by .step().

        Args:
            action (ndarray): The raw action input.

        Returns:
            float: The scalar, clipped force to apply to the cart.

        """
        force = np.clip(action, self.action_space.low, self.action_space.high)
        if not np.array_equal(force, np.array(action)) and self.VERBOSE:
            print("[WARNING]: action was clipped in CartPole._preprocess_control().")
        if self.NORMALIZED_RL_ACTION_SPACE:
            force = self.action_scale * force

        # apply disturbances
        if "action" in self.disturbances:
            force = self.disturbances["action"].apply(force, self)
        if self.adversary_disturbance == "action":
            force = force + self.adv_action

        # only use the scalar value.
        force = force[0]
        return force

    def _advance_simulation(self, force):
        """Apply the commanded forces and adversarial actions to the cartpole.

        The PyBullet simulation is stepped PYB_FREQ/CTRL_FREQ times.

        Args:
            force (float): The force to apply to the slider-to-cart joint.

        """
        tab_force = None
        # determine disturbance force
        passive_disturb = "dynamics" in self.disturbances
        adv_disturb = self.adversary_disturbance == "dynamics"
        if passive_disturb or adv_disturb:
            tab_force = np.zeros(2)
        if passive_disturb:
            tab_force = self.disturbances["dynamics"].apply(tab_force, self)
        if adv_disturb and self.adv_action is not None:
            tab_force = tab_force + self.adv_action
            # clear adversary action, wait for next one
            self.adv_action = None

        for _ in range(int(self.PYB_FREQ / self.CTRL_FREQ)):
            # apply disturbance (by tabbing pole on x-z plane)
            if tab_force is not None:
                # convert 2D force to 3D for pybullet
                tab_force_3d = [float(tab_force[0]), 0.0, float(tab_force[1])]
                p.applyExternalForce(
                    self.CARTPOLE_ID,
                    linkIndex=1,  # Pole link.
                    forceObj=tab_force_3d,
                    posObj=p.getLinkState(
                        self.CARTPOLE_ID,
                        linkIndex=1,  # Pole link.
                        physicsClientId=self.PYB_CLIENT)[0],  # exert force on pole center
                    flags=p.WORLD_FRAME,
                    physicsClientId=self.PYB_CLIENT)

                # Debug visualization
                if self.GUI:
                    center = np.asarray(p.getLinkState(self.CARTPOLE_ID, linkIndex=1, physicsClientId=self.PYB_CLIENT)[0])
                    ff = np.asarray(tab_force_3d) * 10
                    if self.line is None:
                        self.line = p.addUserDebugLine(center.tolist(), (center - ff).tolist(), lineColorRGB=[0, 0, 0], lineWidth=1)
                    else:
                        p.addUserDebugLine(center.tolist(), (center - ff).tolist(), lineColorRGB=[0, 0, 0], lineWidth=1, replaceItemUniqueId=self.line)

            # apply main control
            p.setJointMotorControl2(
                self.CARTPOLE_ID,
                jointIndex=0,  # Slider-to-cart joint.
                controlMode=p.TORQUE_CONTROL,
                force=force,
                physicsClientId=self.PYB_CLIENT)

            # step simulation and counter.
            p.stepSimulation(physicsClientId=self.PYB_CLIENT)
            self.step_counter += 1

    def _get_observation(self):
        """Returns the current observation (state) of the environment.

        Returns:
            ndarray: The state (x, x_dot, theta, theta_dot) of the cartpole.

        """
        if not np.array_equal(self.state, np.clip(self.state, self.observation_space.low, self.observation_space.high)) and self.VERBOSE:
            print("[WARNING]: observation was clipped in CartPole._get_observation().")
        # apply observation disturbance
        obs = deepcopy(self.state)
        if "observation" in self.disturbances:
            obs = self.disturbances["observation"].apply(obs, self)
        return obs

    def _get_reward(self):
        """Computes the current step's reward value.

        Returns:
            float: The evaluated reward/cost.

        """
        if self.COST == Cost.RL_REWARD:
            if self.constraints is not None and self.use_constraint_penalty and self.constraints.is_almost_active(self):
                return self.constraint_penalty

            # Constant reward if episode not done (pole stays upright).
            return 1.0

            # # a more control-oriented rl rewarc
            # state = self.state
            # x, theta = state[0], state[2]
            # length = self.OVERRIDDEN_EFFECTIVE_POLE_LENGTH
            # # shape (*, 2)
            # ee_pos = np.stack([x + length * np.sin(theta), length * np.cos(theta)], -1)
            # goal_pos = np.asarray([0.0, length])
            # # shape (*,)
            # reward = np.exp(-np.sum(np.square(ee_pos - goal_pos) * np.asarray([1.0, 1.0]), -1) / length**2)
            # return reward

        if self.COST == Cost.QUADRATIC:
            if self.TASK == Task.STABILIZATION:
                return float(-1 * self.symbolic.loss(x=self.state, Xr=self.X_GOAL, u=self.current_preprocessed_action, Ur=self.U_GOAL, Q=self.Q, R=self.R)["l"])
            if self.TASK == Task.TRAJ_TRACKING:
                return -1

    def _get_done(self):
        """Computes the conditions for termination of an episode.

        Returns:
            bool: Whether an episode is over.

        """
        # Done if the episode length is exceeded.
        if self.step_counter / self.PYB_FREQ >= self.EPISODE_LEN_SEC:
            return True
        # Done if a constraint is violated.
        if self.constraints is not None:
            if self.DONE_ON_VIOLATION and self.constraints.is_violated(self):
                return True
        # Done if state is out-of-bounds.
        x, _, theta, _ = self.state
        return bool(x < -self.x_threshold or x > self.x_threshold or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians)

    def _get_info(self):
        """Generates the info dictionary returned by every call to .step().

        Returns:
            dict: A dictionary with information about the constraints evaluations and violations.

        """
        info = {}
        if self.constraints is not None:
            info["constraint_values"] = self.constraints.get_values(self)
            violation = np.any(np.greater(info["constraint_values"], 0.))
            info["constraint_violation"] = int(violation)

        # HACK: adding Timilimit flag
        if self.step_counter / self.PYB_FREQ >= self.EPISODE_LEN_SEC:
            # should be `not done` instead of True, for hack just True
            x, _, theta, _ = self.state
            done = bool(x < -self.x_threshold or x > self.x_threshold or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians)
            if self.constraints is not None:
                if self.DONE_ON_VIOLATION and self.constraints.is_violated(self):
                    done = True
            info['TimeLimit.truncated'] = not done
        return info

    def _get_reset_info(self):
        """Generates the info dictionary returned by every call to .reset().

        Returns:
            dict: A dictionary with information about the dynamics and constraints symbolic models.

        """
        info = {}
        info["symbolic_model"] = self.symbolic
        info["x_reference"] = self.X_GOAL
        info["u_reference"] = self.U_GOAL
        if self.constraints is not None:
            # NOTE: Cannot evaluate constraints on reset because there are no inputs. How to handle this?
            info["constraint_values"] = self.constraints.get_values(self, only_state=True)
        return info

    def _parse_urdf_parameters(self, file_name):
        """Parses an URDF file for the robot's properties.

        Args:
            file_name (str, optional): The .urdf file from which the properties should be pased.

        Returns:
            float: The effective pole length.
            float: The pole mass.
            float: The cart mass.

        """
        URDF_TREE = (etxml.parse(file_name)).getroot()

        EFFECTIVE_POLE_LENGTH = 0.5 * float(URDF_TREE[3][0][0][0].attrib["size"].split(" ")[-1])  # Note: HALF length of pole.
        POLE_MASS = float(URDF_TREE[3][1][1].attrib["value"])
        CART_MASS = float(URDF_TREE[1][2][0].attrib["value"])

        return EFFECTIVE_POLE_LENGTH, POLE_MASS, CART_MASS

    def _create_urdf(self, file_name, length=None, inertia=None):
        """For domain randomization.
        
        Args:
            file_name (str): path to the base URDF with attributes to modify.
            length (float): overriden effective pole length. 
            inertia (float): pole inertia (symmetric, Ixx & Iyy).
            
        Returns:
            xml tree object.
            
        """
        tree = etxml.parse(file_name)
        root = tree.getroot()

        # overwrite pod length
        if length is not None:
            # pole visual geometry box
            out = root[3][0][0][0].attrib["size"]
            out = " ".join(out.split(" ")[:-1] + [str(2 * length)])
            root[3][0][0][0].attrib["size"] = out
            # pole visual origin
            out = root[3][0][1].attrib["xyz"]
            out = " ".join(out.split(" ")[:-1] + [str(length)])
            root[3][0][1].attrib["xyz"] = out
            # pole inertial origin
            out = root[3][1][0].attrib["xyz"]
            out = " ".join(out.split(" ")[:-1] + [str(length)])
            root[3][1][0].attrib["xyz"] = out
            # pole inertia
            root[3][1][2].attrib["ixx"] = str(inertia)
            root[3][1][2].attrib["iyy"] = str(inertia)
            root[3][1][2].attrib["izz"] = str(0.0)
            # pole collision geometry box
            out = root[3][2][0][0].attrib["size"]
            out = " ".join(out.split(" ")[:-1] + [str(2 * length)])
            root[3][2][0][0].attrib["size"] = out
            # pole collision origin
            out = root[3][2][1].attrib["xyz"]
            out = " ".join(out.split(" ")[:-1] + [str(length)])
            root[3][2][1].attrib["xyz"] = out

        return tree
