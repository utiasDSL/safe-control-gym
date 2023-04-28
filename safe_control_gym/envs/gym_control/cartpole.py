'''Cartpole environment using PyBullet physics.

Classic cart-pole system implemented by Rich Sutton et al.
    * http://incompleteideas.net/sutton/book/code/pole.c

Also see:
    * github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    * github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/cartpole_bullet.py
'''

import os
import math
import xml.etree.ElementTree as etxml
from copy import deepcopy

import casadi as cs
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces

from safe_control_gym.envs.benchmark_env import BenchmarkEnv, Cost, Task
from safe_control_gym.envs.constraints import SymmetricStateConstraint, GENERAL_CONSTRAINTS
from safe_control_gym.math_and_models.symbolic_systems import SymbolicModel
from safe_control_gym.math_and_models.normalization import normalize_angle


class CartPole(BenchmarkEnv):
    '''Cartpole environment task.

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
        - constraint_form: bounded_constraint
          constrained_variable: STATE
          active_dims: [2, 3]
          lower_bounds: [-0.2, -0.2]
          upper_bounds: [0.2, 0.2]
          tolerance: [0.05, 0.05, 0.05, 0.05]
        done_on_violation: True
        disturbances:
            observation:
            - disturbance_func: white_noise
              std: 4.0
            action:
            - disturbance_func: white_noise
              std: 4.0
        adversary_disturbance: dynamics
        adversary_disturbance_scale: 0.01
    '''

    NAME = 'cartpole'

    URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'cartpole_template.urdf')

    AVAILABLE_CONSTRAINTS = {
        'abs_bound': SymmetricStateConstraint
    }
    AVAILABLE_CONSTRAINTS.update(deepcopy(GENERAL_CONSTRAINTS))

    DISTURBANCE_MODES = {'observation': {'dim': 4}, 'action': {'dim': 1}, 'dynamics': {'dim': 2}}

    INERTIAL_PROP_RAND_INFO = {
        'pole_length': {  # Nominal: 1
            'distrib': 'choice',
            'args': [[1, 5, 10]]
        },
        'cart_mass': {  # Nominal: 1
            'distrib': 'uniform',
            'low': 0.5,
            'high': 1.5
        },
        'pole_mass': {  # Nominal: 0.1
            'distrib': 'uniform',
            'low': 0.05,
            'high': 0.15
        }
    }

    INIT_STATE_RAND_INFO = {
        'init_x': {
            'distrib': 'uniform',
            'low': -0.05,
            'high': 0.05
        },
        'init_x_dot': {
            'distrib': 'uniform',
            'low': -0.05,
            'high': 0.05
        },
        'init_theta': {
            'distrib': 'uniform',
            'low': -0.05,
            'high': 0.05
        },
        'init_theta_dot': {
            'distrib': 'uniform',
            'low': -0.05,
            'high': 0.05
        }
    }

    TASK_INFO = {
        'stabilization_goal': [0],
        'stabilization_goal_tolerance': 0.05,
        'trajectory_type': 'circle',
        'num_cycles': 1,
        'trajectory_plane': 'zx',
        'trajectory_position_offset': [0, 0],
        'trajectory_scale': 0.2
    }

    def __init__(self,
                 init_state=None,
                 inertial_prop=None,
                 # custom args
                 obs_goal_horizon=0,
                 obs_wrap_angle=False,
                 rew_state_weight=1.0,
                 rew_act_weight=0.0001,
                 rew_exponential=True,
                 done_on_out_of_bound=True,
                 **kwargs
                 ):
        '''Initialize a cartpole environment.

        Args:
            init_state  (ndarray/dict, optional): The initial state of the environment.
                (x, x_dot, theta, theta_dot).
            inertial_prop (dict, optional): The ground truth inertial properties of the environment.
            obs_goal_horizon (int): How many future goal states to append to obervation.
            obs_wrap_angle (bool): If to wrap angle to [-pi, pi] when used in observation.
            rew_state_weight (list/ndarray): Quadratic weights for state in rl reward.
            rew_act_weight (list/ndarray): Quadratic weights for action in rl reward.
            rew_exponential (bool): If to exponentiate negative quadratic cost to positive, bounded [0,1] reward.
            done_on_out_of_bound (bool): If to termiante when state is out of bound.
        '''
        self.obs_goal_horizon = obs_goal_horizon
        self.obs_wrap_angle = obs_wrap_angle
        self.rew_state_weight = np.array(rew_state_weight, ndmin=1, dtype=float)
        self.rew_act_weight = np.array(rew_act_weight, ndmin=1, dtype=float)
        self.rew_exponential = rew_exponential
        self.done_on_out_of_bound = done_on_out_of_bound
        # BenchmarkEnv constructor, called after defining the custom args,
        # since some BenchmarkEnv init setup can be task(custom args)-dependent.
        super().__init__(init_state=init_state, inertial_prop=inertial_prop, **kwargs)

        # Create PyBullet client connection.
        self.PYB_CLIENT = -1
        if self.GUI:
            self.PYB_CLIENT = p.connect(p.GUI)
        else:
            self.PYB_CLIENT = p.connect(p.DIRECT)
        # disable urdf caching for randomization via reloading urdf
        p.setPhysicsEngineParameter(enableFileCaching=0)

        # Set GUI and rendering constants.
        self.RENDER_HEIGHT = int(200)
        self.RENDER_WIDTH = int(320)

        # Set the initial state.
        if init_state is None:
            self.INIT_X, self.INIT_X_DOT, self.INIT_THETA, self.INIT_THETA_DOT = np.zeros(4)
        elif isinstance(init_state, np.ndarray):
            self.INIT_X, self.INIT_X_DOT, self.INIT_THETA, self.INIT_THETA_DOT = init_state
        elif isinstance(init_state, dict):
            self.INIT_X = init_state.get('init_x', 0)
            self.INIT_X_DOT = init_state.get('init_x_dot', 0)
            self.INIT_THETA = init_state.get('init_theta', 0)
            self.INIT_THETA_DOT = init_state.get('init_theta_dot', 0)
        else:
            raise ValueError('[ERROR] in CartPole.__init__(), init_state incorrect format.')

        # Get physical properties from URDF (as default parameters).
        self.GRAVITY_ACC = 9.8
        EFFECTIVE_POLE_LENGTH, POLE_MASS, CART_MASS = self._parse_urdf_parameters(self.URDF_PATH)

        # Store ground truth parameters.
        if inertial_prop is None:
            self.EFFECTIVE_POLE_LENGTH = EFFECTIVE_POLE_LENGTH
            self.POLE_MASS = POLE_MASS
            self.CART_MASS = CART_MASS
        elif isinstance(inertial_prop, dict):
            self.EFFECTIVE_POLE_LENGTH = inertial_prop.get('pole_length', EFFECTIVE_POLE_LENGTH)
            self.POLE_MASS = inertial_prop.get('pole_mass', POLE_MASS)
            self.CART_MASS = inertial_prop.get('cart_mass', CART_MASS)
        else:
            raise ValueError('[ERROR] in CartPole.__init__(), inertial_prop incorrect format.')

        # Create X_GOAL and U_GOAL references for the assigned task.
        self.U_GOAL = np.zeros(1)
        if self.TASK == Task.STABILIZATION:
            self.X_GOAL = np.hstack([self.TASK_INFO['stabilization_goal'][0], 0., 0., 0.])  # x = {x, x_dot, theta, theta_dot}.
        elif self.TASK == Task.TRAJ_TRACKING:
            POS_REF, VEL_REF, _ = self._generate_trajectory(traj_type=self.TASK_INFO['trajectory_type'],
                                                            traj_length=self.EPISODE_LEN_SEC,
                                                            num_cycles=self.TASK_INFO['num_cycles'],
                                                            traj_plane=self.TASK_INFO['trajectory_plane'],
                                                            position_offset=np.array(self.TASK_INFO['trajectory_position_offset']),
                                                            scaling=self.TASK_INFO['trajectory_scale'],
                                                            sample_time=self.CTRL_TIMESTEP
                                                            )
            self.X_GOAL = np.vstack([
                POS_REF[:, 0],  # Possible feature: add initial position.
                VEL_REF[:, 0],
                np.zeros(POS_REF.shape[0]),
                np.zeros(VEL_REF.shape[0])
            ]).transpose()

        # Set prior/symbolic info.
        self._setup_symbolic()

    def step(self, action):
        '''Advances the environment by one control step.

        Args:
            action (ndarray): The action applied to the environment for the step.

        Returns:
            obs (ndarray): The state of the environment after the step.
            reward (float): The scalar reward/cost of the step.
            done (bool): Whether the conditions for the end of an episode are met in the step.
            info (dict): A dictionary with information about the constraints evaluations and violations.
        '''

        force = super().before_step(action)
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
        obs, rew, done, info = super().after_step(obs, rew, done, info)
        return obs, rew, done, info

    def reset(self, seed=None):
        '''(Re-)initializes the environment to start an episode.

        Mandatory to call at least once after __init__().

        Args:
            seed (int): An optional seed to reseed the environment.

        Returns:
            obs (ndarray): The initial state of the environment.
            info (dict): A dictionary with information about the dynamics and constraints symbolic models.
        '''

        super().before_reset(seed=seed)
        # PyBullet simulation reset.
        p.resetSimulation(physicsClientId=self.PYB_CLIENT)
        p.setGravity(0, 0, -self.GRAVITY_ACC, physicsClientId=self.PYB_CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.PYB_CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.PYB_CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.PYB_CLIENT)
        # Choose randomized or deterministic inertial properties.
        prop_values = {'pole_length': self.EFFECTIVE_POLE_LENGTH, 'cart_mass': self.CART_MASS, 'pole_mass': self.POLE_MASS}
        if self.RANDOMIZED_INERTIAL_PROP:
            prop_values = self._randomize_values_by_info(prop_values, self.INERTIAL_PROP_RAND_INFO)
            if any(phy_quantity < 0 for phy_quantity in prop_values.values()):
                raise ValueError('[ERROR] in CartPole.reset(), negative randomized inertial properties.')
        self.OVERRIDDEN_EFFECTIVE_POLE_LENGTH = prop_values['pole_length']
        self.OVERRIDDEN_CART_MASS = prop_values['cart_mass']
        self.OVERRIDDEN_POLE_MASS = prop_values['pole_mass']
        # See `slender rod`, https://en.wikipedia.org/wiki/List_of_moments_of_inertia.
        OVERRIDDEN_POLE_INERTIA = (1 / 12) * self.OVERRIDDEN_POLE_MASS * (2 * self.OVERRIDDEN_EFFECTIVE_POLE_LENGTH)**2
        # Load the cartpole with new urdf.
        override_urdf_tree = self._create_urdf(self.URDF_PATH, length=self.OVERRIDDEN_EFFECTIVE_POLE_LENGTH, inertia=OVERRIDDEN_POLE_INERTIA)
        self.override_path = os.path.join(self.output_dir, f'pid-{os.getpid()}_id-{self.idx}_cartpole.urdf')
        override_urdf_tree.write(self.override_path)
        self.CARTPOLE_ID = p.loadURDF(
            self.override_path,
            basePosition=[0, 0, 0],
            physicsClientId=self.PYB_CLIENT)
        # Remove cache file after loading it into PyBullet.
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
        init_values = {'init_x': self.INIT_X, 'init_x_dot': self.INIT_X_DOT, 'init_theta': self.INIT_THETA, 'init_theta_dot': self.INIT_THETA_DOT}
        if self.RANDOMIZED_INIT:
            init_values = self._randomize_values_by_info(init_values, self.INIT_STATE_RAND_INFO)
        OVERRIDDEN_INIT_X = init_values['init_x']
        OVERRIDDEN_INIT_X_DOT = init_values['init_x_dot']
        OVERRIDDEN_INIT_THETA = init_values['init_theta']
        OVERRIDDEN_INIT_THETA_DOT = init_values['init_theta_dot']
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
        obs, info = self._get_observation(), self._get_reset_info()
        obs, info = super().after_reset(obs, info)
        # Return either an observation and dictionary or just the observation.
        if self.INFO_IN_RESET:
            return obs, info
        else:
            return obs

    def render(self, mode='human'):
        '''Retrieves a frame from PyBullet rendering.

        Args:
            mode (str): Unused.

        Returns:
            frame (ndarray): A multidimensional array with the RGB frame captured by PyBullet's camera.
        '''
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
        return np.reshape(rgb, (h, w, 4))

    def close(self):
        '''Clean up the environment and PyBullet connection.'''
        if self.PYB_CLIENT >= 0:
            p.disconnect(physicsClientId=self.PYB_CLIENT)
        self.PYB_CLIENT = -1

    def _setup_symbolic(self, prior_prop={}, **kwargs):
        '''Creates symbolic (CasADi) models for dynamics, observation, and cost.

        Args:
            prior_prop (dict): specify the prior inertial prop to use in the symbolic model.
        '''
        l = prior_prop.get('pole_length', self.EFFECTIVE_POLE_LENGTH)
        m = prior_prop.get('pole_mass', self.POLE_MASS)
        M = prior_prop.get('cart_mass', self.CART_MASS)
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
        dynamics = {'dyn_eqn': X_dot, 'obs_eqn': Y, 'vars': {'X': X, 'U': U}}
        cost = {'cost_func': cost_func, 'vars': {'X': X, 'U': U, 'Xr': Xr, 'Ur': Ur, 'Q': Q, 'R': R}}
        # Additional params to cache
        params = {
            # prior inertial properties
            'pole_length': l,
            'pole_mass': m,
            'cart_mass': M,
            # equilibrium point for linearization
            'X_EQ': np.zeros(self.state_dim),
            'U_EQ': np.atleast_2d(self.U_GOAL)[0, :],
        }
        # Setup symbolic model.
        self.symbolic = SymbolicModel(dynamics=dynamics, cost=cost, dt=dt, params=params)

    def _set_action_space(self):
        '''Sets the action space of the environment.'''
        self.action_scale = 10
        self.physical_action_bounds = (-1 * np.atleast_1d(self.action_scale), np.atleast_1d(self.action_scale))
        self.action_threshold = 1 if self.NORMALIZED_RL_ACTION_SPACE else self.action_scale
        self.action_space = spaces.Box(low=-self.action_threshold, high=self.action_threshold, shape=(1,))
        # Define action/input labels and units.
        self.ACTION_LABELS = ['U']
        self.ACTION_UNITS = ['N'] if not self.NORMALIZED_RL_ACTION_SPACE else ['-']

    def _set_observation_space(self):
        '''Sets the observation space of the environment.'''
        # Angle at which to fail the episode.
        self.theta_threshold_radians = 90 * math.pi / 180
        # NOTE: different value in PyBullet gym (0.4) and OpenAI gym (2.4).
        self.x_threshold = 2.4
        # Limit set to 2x: i.e. a failing observation is still within bounds.
        obs_bound = np.array([self.x_threshold * 2, np.finfo(np.float32).max, self.theta_threshold_radians * 2, np.finfo(np.float32).max])
        self.state_space = spaces.Box(low=-obs_bound, high=obs_bound, dtype=np.float32)

        # Concatenate goal info for RL
        if self.COST == Cost.RL_REWARD and self.TASK == Task.TRAJ_TRACKING and self.obs_goal_horizon > 0:
            # include future goal state(s)
            # e.g. horizon=1, obs = {state, state_target}
            mul = 1 + self.obs_goal_horizon
            obs_bound = np.concatenate([obs_bound] * mul)
        elif self.COST == Cost.RL_REWARD and self.TASK == Task.STABILIZATION and self.obs_goal_horizon > 0:
            obs_bound = np.concatenate([obs_bound] * 2)
        # Define obs space exposed to the controller
        # Note obs space is often different to state space for RL (with additional task info)
        self.observation_space = spaces.Box(low=-obs_bound, high=obs_bound, dtype=np.float32)

        # Define obs/state labels and units.
        self.STATE_LABELS = ['x', 'x_dot', 'theta', 'theta_dot']
        self.STATE_UNITS = ['m', 'm/s', 'rad', 'rad/s']

    def _preprocess_control(self, action):
        '''Converts the raw action input into the one used by .step().

        Args:
            action (ndarray): The raw action input.

        Returns:
            force (float): The scalar, clipped force to apply to the cart.
        '''
        action = self.denormalize_action(action)
        self.current_physical_action = action

        # Apply disturbances.
        if 'action' in self.disturbances:
            action = self.disturbances['action'].apply(action, self)
        if self.adversary_disturbance == 'action' and self.adv_action is not None:
            action = action + self.adv_action
        self.current_noisy_physical_action = action

        # Save the actual input.
        force = np.clip(action, self.physical_action_bounds[0], self.physical_action_bounds[1])
        self.current_clipped_action = force

        return force[0]  # Only use the scalar value.

    def normalize_action(self, action):
        '''Converts a physical action into an normalized action if necessary.

        Args:
            action (ndarray): The action to be converted.

        Returns:
            normalized_action (ndarray): The action in the correct action space.
        '''
        if self.NORMALIZED_RL_ACTION_SPACE:
            action = action / self.action_scale

        return action

    def denormalize_action(self, action):
        '''Converts a normalized action into a physical action if necessary.

        Args:
            action (ndarray): The action to be converted.

        Returns:
            physical_action (ndarray): The physical action.
        '''
        if self.NORMALIZED_RL_ACTION_SPACE:
            action = self.action_scale * action

        return action

    def _advance_simulation(self, force):
        '''Apply the commanded forces and adversarial actions to the cartpole.

        The PyBullet simulation is stepped PYB_FREQ/CTRL_FREQ times.

        Args:
            force (float): The force to apply to the slider-to-cart joint.
        '''
        tab_force = None
        # Determine the disturbance force.
        passive_disturb = 'dynamics' in self.disturbances
        adv_disturb = self.adversary_disturbance == 'dynamics'
        if passive_disturb or adv_disturb:
            tab_force = np.zeros(2)
        if passive_disturb:
            tab_force = self.disturbances['dynamics'].apply(tab_force, self)
        if adv_disturb and self.adv_action is not None:
            tab_force = tab_force + self.adv_action
            # Clear adversary's action, wait for the next one.
            self.adv_action = None
        for _ in range(self.PYB_STEPS_PER_CTRL):
            # apply disturbance (by tabbing pole on x-z plane).
            if tab_force is not None:
                # Convert 2D force to 3D on for PyBullet.
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
            # Apply control.
            p.setJointMotorControl2(
                self.CARTPOLE_ID,
                jointIndex=0,  # Slider-to-cart joint.
                controlMode=p.TORQUE_CONTROL,
                force=force,
                physicsClientId=self.PYB_CLIENT)
            # Step simulation and counter.
            p.stepSimulation(physicsClientId=self.PYB_CLIENT)

    def _get_observation(self):
        '''Returns the current observation (state) of the environment.

        Returns:
            obs (ndarray): The state (x, x_dot, theta, theta_dot) of the cartpole.
        '''
        if not np.array_equal(self.state, np.clip(self.state, self.state_space.low, self.state_space.high)) and self.VERBOSE:
            print('[WARNING]: observation was clipped in CartPole._get_observation().')
        # Apply observation disturbance.
        obs = deepcopy(self.state)
        if 'observation' in self.disturbances:
            obs = self.disturbances['observation'].apply(obs, self)
        # Wrap angle to constrain state space, useful in swing-up task.
        if self.obs_wrap_angle:
            obs[2] = normalize_angle(obs[2])

        # Concatenate goal info (goal state(s)) for RL
        # Plus two because ctrl_step_counter has not incremented yet, and we want to return the obs (which would be
        # ctrl_step_counter + 1 as the action has already been applied), and the next state (+ 2) for the RL to see
        # the next state.
        if self.at_reset:
            obs = self.extend_obs(obs, 1)
        else:
            obs = self.extend_obs(obs, self.ctrl_step_counter + 2)
        return obs

    def _get_reward(self):
        '''Computes the current step's reward value.

        Returns:
            reward (float): The evaluated reward/cost.
        '''
        if self.COST == Cost.RL_REWARD:
            # negative quadratic reward with angle wrapped around
            state = deepcopy(self.state)
            state[2] = normalize_angle(state[2])
            act = np.asarray(self.current_noisy_physical_action)
            if self.TASK == Task.STABILIZATION:
                state_error = state - self.X_GOAL
                dist = np.sum(self.rew_state_weight * state_error * state_error)
                dist += np.sum(self.rew_act_weight * act * act)
            if self.TASK == Task.TRAJ_TRACKING:
                wp_idx = min(self.ctrl_step_counter + 1, self.X_GOAL.shape[0] - 1)  # +1 because state has already advanced but counter not incremented.
                state_error = state - self.X_GOAL[wp_idx]
                dist = np.sum(self.rew_state_weight * state_error * state_error)
                dist += np.sum(self.rew_act_weight * act * act)
            rew = -dist
            # convert rew to be positive and bounded [0,1]
            if self.rew_exponential:
                rew = np.exp(rew)
            return rew
        if self.COST == Cost.QUADRATIC:
            if self.TASK == Task.STABILIZATION:
                return float(
                    -1 * self.symbolic.loss(x=self.state,
                                            Xr=self.X_GOAL,
                                            u=self.current_clipped_action,
                                            Ur=self.U_GOAL,
                                            Q=self.Q,
                                            R=self.R)['l'])
            if self.TASK == Task.TRAJ_TRACKING:
                return float(
                    -1 * self.symbolic.loss(x=self.state,
                                            Xr=self.X_GOAL[self.ctrl_step_counter, :],
                                            u=self.current_clipped_action,
                                            Ur=self.U_GOAL,
                                            Q=self.Q,
                                            R=self.R)['l'])

    def _get_done(self):
        '''Computes the conditions for termination of an episode.

        Returns:
            done (bool): Whether an episode is over.
        '''
        # Done if goal reached for stabilization task with quadratic cost.
        if self.TASK == Task.STABILIZATION and self.COST == Cost.QUADRATIC:
            self.goal_reached = bool(np.linalg.norm(self.state - self.X_GOAL) < self.TASK_INFO['stabilization_goal_tolerance'])
            if self.goal_reached:
                return True
        # Done if state is out-of-bounds.
        if self.done_on_out_of_bound:
            x, _, theta, _ = self.state
            if x < -self.x_threshold or x > self.x_threshold or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians:
                self.out_of_bounds = True
                return True
        self.out_of_bounds = False
        return False

    def _get_info(self):
        '''Generates the info dictionary returned by every call to .step().

        Returns:
            info (dict): A dictionary with information about the constraints evaluations and violations.
        '''
        info = {}
        if self.TASK == Task.STABILIZATION and self.COST == Cost.QUADRATIC:
            info['goal_reached'] = self.goal_reached  # Add boolean flag for the goal being reached.
        if self.done_on_out_of_bound:
            info['out_of_bounds'] = self.out_of_bounds
        # Add MSE.
        state = deepcopy(self.state)
        info['mse'] = np.sum(state ** 2)
        return info

    def _get_reset_info(self):
        '''Generates the info dictionary returned by every call to .reset().

        Returns:
            info (dict): A dictionary with information about the dynamics and constraints symbolic models.
        '''
        info = {}
        info['symbolic_model'] = self.symbolic
        info['physical_parameters'] = {
            'pole_effective_length': self.OVERRIDDEN_EFFECTIVE_POLE_LENGTH,
            'pole_mass': self.OVERRIDDEN_POLE_MASS,
            'cart_mass': self.OVERRIDDEN_CART_MASS
        }
        info['x_reference'] = self.X_GOAL
        info['u_reference'] = self.U_GOAL
        if self.constraints is not None:
            info['symbolic_constraints'] = self.constraints.get_all_symbolic_models()
            # NOTE: Cannot evaluate constraints on reset/without inputs.
            info['constraint_values'] = self.constraints.get_values(self, only_state=True)  # Fix for input constraints only
        return info

    def _parse_urdf_parameters(self, file_name):
        '''Parses an URDF file for the robot's properties.

        Args:
            file_name (str, optional): The .urdf file from which the properties should be pased.

        Returns:
            EFFECTIVE_POLE_LENGTH (float): The effective pole length.
            POLE_MASS (float): The pole mass.
            CART_MASS (float): The cart mass.
        '''
        URDF_TREE = (etxml.parse(file_name)).getroot()
        EFFECTIVE_POLE_LENGTH = 0.5 * float(URDF_TREE[3][0][0][0].attrib['size'].split(' ')[-1])  # Note: HALF length of pole.
        POLE_MASS = float(URDF_TREE[3][1][1].attrib['value'])
        CART_MASS = float(URDF_TREE[1][2][0].attrib['value'])
        return EFFECTIVE_POLE_LENGTH, POLE_MASS, CART_MASS

    def _create_urdf(self, file_name, length=None, inertia=None):
        '''For domain randomization.

        Args:
            file_name (str): Path to the base URDF with attributes to modify.
            length (float): Overriden effective pole length.
            inertia (float): Pole inertia (symmetric, Ixx & Iyy).

        Returns:
            tree (obj): xml tree object.
        '''
        tree = etxml.parse(file_name)
        root = tree.getroot()
        # Overwrite pod length.
        if length is not None:
            # Pole visual geometry box.
            out = root[3][0][0][0].attrib['size']
            out = ' '.join(out.split(' ')[:-1] + [str(2 * length)])
            root[3][0][0][0].attrib['size'] = out
            # Pole visual origin.
            out = root[3][0][1].attrib['xyz']
            out = ' '.join(out.split(' ')[:-1] + [str(length)])
            root[3][0][1].attrib['xyz'] = out
            # Pole inertial origin.
            out = root[3][1][0].attrib['xyz']
            out = ' '.join(out.split(' ')[:-1] + [str(length)])
            root[3][1][0].attrib['xyz'] = out
            # Pole inertia.
            root[3][1][2].attrib['ixx'] = str(inertia)
            root[3][1][2].attrib['iyy'] = str(inertia)
            root[3][1][2].attrib['izz'] = str(0.0)
            # Pole collision geometry box.
            out = root[3][2][0][0].attrib['size']
            out = ' '.join(out.split(' ')[:-1] + [str(2 * length)])
            root[3][2][0][0].attrib['size'] = out
            # Pole collision origin.
            out = root[3][2][1].attrib['xyz']
            out = ' '.join(out.split(' ')[:-1] + [str(length)])
            root[3][2][1].attrib['xyz'] = out
        return tree
