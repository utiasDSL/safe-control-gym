"""1D and 2D quadrotor environment using PyBullet physics.

Based on UTIAS Dynamic Systems Lab's gym-pybullet-drones:
    * https://github.com/utiasDSL/gym-pybullet-drones

"""
import math
from copy import deepcopy
import casadi as cs
from gym import spaces
import numpy as np
import pybullet as p

from safe_control_gym.envs.benchmark_env import Cost, Task
from safe_control_gym.envs.constraints import GENERAL_CONSTRAINTS
from safe_control_gym.math_and_models.symbolic_systems import SymbolicModel
from safe_control_gym.envs.gym_pybullet_drones.base_aviary import BaseAviary
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType, cmd2pwm, pwm2rpm
from safe_control_gym.math_and_models.normalization import normalize_angle


class Quadrotor(BaseAviary):
    """1D and 2D quadrotor environment task.

    Including symbolic model, constraints, randomization, adversarial disturbances,
    multiple cost functions, stabilization and trajectory tracking references.

    """
    NAME = "quadrotor"
    AVAILABLE_CONSTRAINTS = deepcopy(GENERAL_CONSTRAINTS)

    DISTURBANCE_MODES = {
        "observation": {
            "dim": 6
        },
        "action": {
            "dim": 2
        },
        "dynamics": {
            "dim": 2
        }
    }

    INERTIAL_PROP_RAND_INFO = {
        "M": { # Nominal: 0.027
            'distrib': "uniform",
            'low': 0.022,
            'high': 0.032
        },
        "Iyy": { # Nominal: 1.4e-5
            'distrib': "uniform",
            'low': 1.3e-5,
            'high': 1.5e-5
        }
    }

    INIT_STATE_RAND_INFO = {
        "init_x": {
            'distrib': "uniform",
            'low': -0.5,
            'high': 0.5
        },
        "init_x_dot": {
            'distrib': "uniform",
            'low': -0.01,
            'high': 0.01
        },
        "init_z": {
            'distrib': "uniform",
            'low': 0.1,
            'high': 1.5
        },
        "init_z_dot": {
            'distrib': "uniform",
            'low': -0.01,
            'high': 0.01
        },
        "init_theta": {
            'distrib': "uniform",
            'low': -0.3,
            'high': 0.3
        },
        "init_theta_dot": {
            'distrib': "uniform",
            'low': -0.01,
            'high': 0.01
        }
    }

    TASK_INFO = {
        "stabilization_goal": [0, 1],
        "stabilization_goal_tolerance": 0.05,
        "trajectory_type": "circle",
        "num_cycles": 1,
        "trajectory_plane": "zx",
        "trajectory_position_offset": [0.5, 0],
        "trajectory_scale": -0.5
    }

    def __init__(self,
                 init_state=None,
                 inertial_prop=None,
                 # custom args
                 quad_type: QuadType = QuadType.TWO_D,
                 norm_act_scale=0.1,
                 obs_goal_horizon=1,
                 rew_state_weight=1.0,
                 rew_act_weight=0.0001,
                 rew_exponential=True,
                 done_on_out_of_bound=True,
                 info_mse_metric_state_weight=None,
                 **kwargs
                 ):
        """Initialize a quadrotor environment.

        Args:
            init_state  (ndarray, optional): The initial state of the environment, (z, z_dot) or (x, x_dot, z, z_dot theta, theta_dot).
            inertial_prop (ndarray, optional): The inertial properties of the environment (mass, Iyy).
            quad_type (QuadType, optional): The choice of motion type (1D along z or 2D in the x-z plane).
            norm_act_scale (float): scaling the [-1,1] action space around hover thrust when `normalized_action_space` is True.
            obs_goal_horizon (int): how many future goal states to append to obervation.
            rew_state_weight (list/ndarray): quadratic weights for state in rl reward.
            rew_act_weight (list/ndarray): quadratic weights for action in rl reward.
            rew_exponential (bool): if to exponentiate negative quadratic cost to positive, bounded [0,1] reward.
            done_on_out_of_bound (bool): if to termiante when state is out of bound.
            info_mse_metric_state_weight (list/ndarray): quadratic weights for state in mse calculation for info dict.

        """
        # Select the 1D (moving along z) or 2D (moving in the xz plane) quadrotor.
        self.QUAD_TYPE = QuadType(quad_type)
        self.norm_act_scale = norm_act_scale
        self.obs_goal_horizon = obs_goal_horizon
        self.rew_state_weight = np.array(rew_state_weight, ndmin=1, dtype=float)
        self.rew_act_weight = np.array(rew_act_weight, ndmin=1, dtype=float)
        self.rew_exponential = rew_exponential
        self.done_on_out_of_bound = done_on_out_of_bound
        if info_mse_metric_state_weight is None:
            if self.QUAD_TYPE == QuadType.ONE_D:
                self.info_mse_metric_state_weight = np.array([1,0], ndmin=1, dtype=float)
            elif self.QUAD_TYPE == QuadType.TWO_D:
                self.info_mse_metric_state_weight = np.array([1,0,1,0,0,0], ndmin=1, dtype=float)
            else:
                raise ValueError("[ERROR] in Quadrotor.__init__(), not implemented quad type.")
        else:
            if (self.QUAD_TYPE == QuadType.ONE_D and len(info_mse_metric_state_weight)==2) or \
                    (self.QUAD_TYPE == QuadType.TWO_D and len(info_mse_metric_state_weight)==6):
                self.info_mse_metric_state_weight = np.array(info_mse_metric_state_weight, ndmin=1, dtype=float)
            else:
                raise ValueError("[ERROR] in Quadrotor.__init__(), wrong info_mse_metric_state_weight argument size.")
        # BaseAviary constructor, called after defining the custom args, 
        # since some BenchmarkEnv init setup can be task(custom args)-dependent. 
        super().__init__(init_state=init_state, inertial_prop=inertial_prop, **kwargs)
        
        # Custom disturbance info.
        # 1D quad disturbances have lower dimensions
        if self.QUAD_TYPE == QuadType.ONE_D:
            self.DISTURBANCE_MODES["observation"]["dim"] = 2
            self.DISTURBANCE_MODES["action"]["dim"] = 1
            self.DISTURBANCE_MODES["dynamics"]["dim"] = 1

        # Store initial state info.
        if init_state is None:
            self.INIT_X, self.INIT_X_DOT, self.INIT_Z, self.INIT_Z_DOT, self.INIT_THETA, self.INIT_THETA_DOT = np.zeros(6)
        elif self.QUAD_TYPE == QuadType.ONE_D:
            self.INIT_X, self.INIT_X_DOT, self.INIT_THETA, self.INIT_THETA_DOT = np.zeros(4)
            if isinstance(init_state, np.ndarray):
                self.INIT_Z, self.INIT_Z_DOT = init_state
            elif isinstance(init_state, dict):
                self.INIT_Z = init_state.get("init_z", 0)
                self.INIT_Z_DOT = init_state.get("init_z_dot", 0)
            else:
                raise ValueError("[ERROR] in Quadrotor.__init__(), init_state incorrect format.")
        elif self.QUAD_TYPE == QuadType.TWO_D:
            if isinstance(init_state, np.ndarray):
                self.INIT_X, self.INIT_X_DOT, self.INIT_Z, self.INIT_Z_DOT, self.INIT_THETA, self.INIT_THETA_DOT = init_state
            elif isinstance(init_state, dict):
                self.INIT_X = init_state.get("init_x", 0)
                self.INIT_X_DOT = init_state.get("init_x_dot", 0)
                self.INIT_Z = init_state.get("init_z", 0)
                self.INIT_Z_DOT = init_state.get("init_z_dot", 0)
                self.INIT_THETA = init_state.get("init_theta", 0)
                self.INIT_THETA_DOT = init_state.get("init_theta_dot", 0)
            else:
                raise ValueError("[ERROR] in Quadrotor.__init__(), init_state incorrect format.")
        # Decide whether to randomize the initial state and how (see info dictionary).
        #self.RANDOMIZED_INIT = randomized_init
        #if init_state_randomization_info is not None:
        #    self.INIT_STATE_RAND_INFO = init_state_randomization_info
        # Do NOT randomize x, x_dot, theta, theta_dot for the 1D quadrotor.
        if self.QUAD_TYPE == QuadType.ONE_D:
            for init_name in ["init_x", "init_x_dot", "init_theta", "init_theta_dot"]:
                self.INIT_STATE_RAND_INFO.pop(init_name, None)
                
        # Decide whether to randomize the inertial properties and how (see info dictionary).
        # self.RANDOMIZED_INERTIAL_PROP = randomized_inertial_prop
        # if inertial_prop_randomization_info is not None:
        #     self.INERTIAL_PROP_RAND_INFO = inertial_prop_randomization_info
        # Do NOT randomize J for the 1D quadrotor.
        if self.QUAD_TYPE == QuadType.ONE_D:
            self.INERTIAL_PROP_RAND_INFO.pop("Iyy", None)
                
        # Override inertial properties of passed as arguments.
        if inertial_prop is None:
            pass
        elif np.array(inertial_prop).shape == (2,):
            self.MASS, self.J[1, 1] = inertial_prop
        elif  isinstance(inertial_prop, dict):
            self.MASS = inertial_prop.get("mass", 0)
            self.J[1, 1] = inertial_prop.get("iyy", 0)
        else:
            raise ValueError("[ERROR] in Quadrotor.__init__(), inertial_prop is not of shape (2,).")
        # Set prior/symbolic info.
        self._setup_symbolic()
            
        # Create X_GOAL and U_GOAL references for the assigned task.
        self.U_GOAL = np.ones(self.action_dim) * self.MASS * self.GRAVITY_ACC / self.action_dim
        if self.TASK == Task.STABILIZATION:
            if self.QUAD_TYPE == QuadType.ONE_D:
                self.X_GOAL = np.hstack(
                    [self.TASK_INFO["stabilization_goal"][1],
                     0.0])  # x = {z, z_dot}.
            elif self.QUAD_TYPE == QuadType.TWO_D:
                self.X_GOAL = np.hstack([
                    self.TASK_INFO["stabilization_goal"][0], 0.0,
                    self.TASK_INFO["stabilization_goal"][1], 0.0, 0.0, 0.0
                ])  # x = {x, x_dot, z, z_dot, theta, theta_dot}.
        elif self.TASK == Task.TRAJ_TRACKING:
            POS_REF, \
            VEL_REF, \
            SPEED = self._generate_trajectory(traj_type=self.TASK_INFO["trajectory_type"],
                                              traj_length=self.EPISODE_LEN_SEC,
                                              num_cycles=self.TASK_INFO["num_cycles"],
                                              traj_plane=self.TASK_INFO["trajectory_plane"],
                                              position_offset=self.TASK_INFO["trajectory_position_offset"],
                                              scaling=self.TASK_INFO["trajectory_scale"],
                                              sample_time=self.CTRL_TIMESTEP
                                              )
            if self.QUAD_TYPE == QuadType.ONE_D:
                self.X_GOAL = np.vstack([
                    POS_REF[:, 2],
                    VEL_REF[:, 2]
                ]).transpose()
            elif self.QUAD_TYPE == QuadType.TWO_D:
                self.X_GOAL = np.vstack([
                    POS_REF[:, 0],
                    VEL_REF[:, 0],
                    POS_REF[:, 2],
                    VEL_REF[:, 2],
                    np.zeros(POS_REF.shape[0]),
                    np.zeros(VEL_REF.shape[0])
                ]).transpose()
        # Equilibrium point at hover for linearization.
        self.X_EQ = np.zeros(self.state_dim)
        self.U_EQ = self.U_GOAL


    def reset(self):
        """(Re-)initializes the environment to start an episode.

        Mandatory to call at least once after __init__().

        Returns:
            ndarray: The initial state of the environment.
            dict: A dictionary with information about the dynamics and constraints symbolic models.

        """
        super().before_reset()
        # PyBullet simulation reset.  
        super()._reset_simulation()
        # Choose randomized or deterministic inertial properties.
        prop_values = {
            "M": self.MASS,
            "Iyy": self.J[1, 1],
        }
        if self.RANDOMIZED_INERTIAL_PROP:
            prop_values = self._randomize_values_by_info(
                prop_values, self.INERTIAL_PROP_RAND_INFO)
            if any(phy_quantity < 0 for phy_quantity in prop_values.values()):
                raise ValueError("[ERROR] in CartPole.reset(), negative randomized inertial properties.")
        self.OVERRIDDEN_QUAD_MASS = prop_values["M"]
        self.OVERRIDDEN_QUAD_INERTIA = [self.J[0, 0], prop_values["Iyy"], self.J[2, 2]]
        # Override inertial properties.
        p.changeDynamics(
            self.DRONE_IDS[0],
            linkIndex=-1,  # Base link.
            mass=self.OVERRIDDEN_QUAD_MASS,
            localInertiaDiagonal=self.OVERRIDDEN_QUAD_INERTIA,
            physicsClientId=self.PYB_CLIENT)
        # Randomize initial state.
        init_values = {
            "init_x": self.INIT_X,
            "init_x_dot": self.INIT_X_DOT,
            "init_z": self.INIT_Z,
            "init_z_dot": self.INIT_Z_DOT,
            "init_theta": self.INIT_THETA,
            "init_theta_dot": self.INIT_THETA_DOT,
        }
        if self.RANDOMIZED_INIT:
            init_values = self._randomize_values_by_info(init_values, self.INIT_STATE_RAND_INFO)
        OVERRIDDEN_INIT_X = init_values["init_x"]
        OVERRIDDEN_INIT_X_DOT = init_values["init_x_dot"]
        OVERRIDDEN_INIT_Z = init_values["init_z"]
        OVERRIDDEN_INIT_Z_DOT = init_values["init_z_dot"]
        OVERRIDDEN_INIT_THETA = init_values["init_theta"]
        OVERRIDDEN_INIT_THETA_DOT = init_values["init_theta_dot"]
        p.resetBasePositionAndOrientation(self.DRONE_IDS[0], [OVERRIDDEN_INIT_X, 0, OVERRIDDEN_INIT_Z],
                                          p.getQuaternionFromEuler([0, OVERRIDDEN_INIT_THETA, 0]),
                                          physicsClientId=self.PYB_CLIENT)
        p.resetBaseVelocity(self.DRONE_IDS[0],
                            [OVERRIDDEN_INIT_X_DOT, 0, OVERRIDDEN_INIT_Z_DOT],
                            [0, OVERRIDDEN_INIT_THETA_DOT, 0],
                            physicsClientId=self.PYB_CLIENT)
        # Update BaseAviary internal variables before calling self._get_observation().
        self._update_and_store_kinematic_information()
        obs, info = self._get_observation(), self._get_reset_info()
        obs, info = super().after_reset(obs, info)
        # Return either an observation and dictionary or just the observation.
        if self.INFO_IN_RESET:
            return obs, info
        else:
            return obs
        

    def step(self, action):
        """Advances the environment by one control step.
        
        Pass the commanded RPMs and the adversarial force to the superclass .step().
        The PyBullet simulation is stepped PYB_FREQ/CTRL_FREQ times in BaseAviary.

        Args:
            action (ndarray): the action applied to the environment for the step.

        Returns:
            ndarray: The state of the environment after the step.
            float: The scalar reward/cost of the step.
            bool: Whether the conditions for the end of an episode are met in the step.
            dict: A dictionary with information about the constraints evaluations and violations.

        """
        # Get the preprocessed rpm for each motor
        rpm = super().before_step(action)
        # Determine disturbance force.
        disturb_force = None
        passive_disturb = "dynamics" in self.disturbances
        adv_disturb = self.adversary_disturbance == "dynamics"
        if passive_disturb or adv_disturb:
            disturb_force = np.zeros(2)
        if passive_disturb:
            disturb_force = self.disturbances["dynamics"].apply(
                disturb_force, self)
        if adv_disturb and self.adv_action is not None:
            disturb_force = disturb_force + self.adv_action
            # Clear the adversary action, wait for the next one.
            self.adv_action = None
        # Construct full (3D) disturbance force.
        if disturb_force is not None:
            if self.QUAD_TYPE == QuadType.ONE_D:
                # Only disturb on z direction.
                disturb_force = [0, 0, float(disturb_force)]
            elif self.QUAD_TYPE == QuadType.TWO_D:
                # Only disturb on x-z plane.
                disturb_force = [
                    float(disturb_force[0]), 0,
                    float(disturb_force[1])
                ]
            else:
                raise NotImplementedError(
                    "[ERROR] in Quadrotor._advance_simulation(), disturb force for quad 3D is not available."
                )
        # Advance the simulation.        
        super()._advance_simulation(rpm, disturb_force)
        # Standard Gym return.
        obs = self._get_observation()
        rew = self._get_reward()
        done = self._get_done()
        info = self._get_info()
        obs, rew, done, info = super().after_step(obs, rew, done, info)
        return obs, rew, done, info
    
    def render(self, mode='human'):
        """Retrieves a frame from PyBullet rendering.

        Args:
            mode (str): Unused.

        Returns:
            ndarray: A multidimensional array with the RGB frame captured by PyBullet's camera.

        """
        [w, h, rgb, dep, seg] = p.getCameraImage(width=self.RENDER_WIDTH,
                                                 height=self.RENDER_HEIGHT,
                                                 shadow=1,
                                                 viewMatrix=self.CAM_VIEW,
                                                 projectionMatrix=self.CAM_PRO,
                                                 renderer=p.ER_TINY_RENDERER,
                                                 flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                 physicsClientId=self.PYB_CLIENT)
        # Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA').show()
        return np.reshape(rgb, (h, w, 4))

    def _setup_symbolic(self):
        """Creates symbolic (CasADi) models for dynamics, observation, and cost.

        Returns:
            SymbolicModel: CasADi symbolic model of the environment.

        """
        m, g, l = self.MASS, self.GRAVITY_ACC, self.L
        Iyy = self.J[1, 1]
        dt = self.CTRL_TIMESTEP
        # Define states.
        z = cs.MX.sym('z')
        z_dot = cs.MX.sym('z_dot')
        if self.QUAD_TYPE == QuadType.ONE_D:
            nx, nu = 2, 1
            # Define states.
            X = cs.vertcat(z, z_dot)
            # Define input thrust.
            T = cs.MX.sym('T')
            U = cs.vertcat(T)
            # Define dynamics equations.
            X_dot = cs.vertcat(z_dot, T / m - g)
            # Define observation equation.
            Y = cs.vertcat(z, z_dot)
        elif self.QUAD_TYPE == QuadType.TWO_D:
            nx, nu = 6, 2
            # Define states.
            x = cs.MX.sym('x')
            x_dot = cs.MX.sym('x_dot')
            theta = cs.MX.sym('theta')
            theta_dot = cs.MX.sym('theta_dot')
            X = cs.vertcat(x, x_dot, z, z_dot, theta, theta_dot)
            # Define input thrusts.
            T1 = cs.MX.sym('T1')
            T2 = cs.MX.sym('T2')
            U = cs.vertcat(T1, T2)
            # Define dynamics equations.
            X_dot = cs.vertcat(x_dot,
                               cs.sin(theta) * (T1 + T2) / m, z_dot,
                               cs.cos(theta) * (T1 + T2) / m - g, theta_dot,
                               l * (T2 - T1) / Iyy / np.sqrt(2))
            # Define observation.
            Y = cs.vertcat(x, x_dot, z, z_dot, theta, theta_dot)
        # Define cost (quadratic form).
        Q = cs.MX.sym('Q', nx, nx)
        R = cs.MX.sym('R', nu, nu)
        Xr = cs.MX.sym('Xr', nx, 1)
        Ur = cs.MX.sym('Ur', nu, 1)
        cost_func = 0.5 * (X - Xr).T @ Q @ (X - Xr) + 0.5 * (U - Ur).T @ R @ (U - Ur)
        # Define dynamics and cost dictionaries.
        dynamics = {"dyn_eqn": X_dot, "obs_eqn": Y, "vars": {"X": X, "U": U}}
        cost = {
            "cost_func": cost_func,
            "vars": {
                "X": X,
                "U": U,
                "Xr": Xr,
                "Ur": Ur,
                "Q": Q,
                "R": R
            }
        }
        # Setup symbolic model.
        self.symbolic = SymbolicModel(dynamics=dynamics, cost=cost, dt=dt)

    def _set_action_space(self):
        """Returns the action space of the environment.

        Returns:
            gym.spaces: The quadrotor environment's action space, of size 1 or 2 depending on QUAD_TYPE.

        """
        # Define action/input dimension, labels, and units.
        if self.QUAD_TYPE == QuadType.ONE_D:
            action_dim = 1 
            self.ACTION_LABELS = ['T']
            self.ACTION_UNITS = ['N'] if not self.NORMALIZED_RL_ACTION_SPACE else ['-']
        elif self.QUAD_TYPE == QuadType.TWO_D:
            action_dim = 2 
            self.ACTION_LABELS = ['T1', 'T2']
            self.ACTION_UNITS = ['N', 'N'] if not self.NORMALIZED_RL_ACTION_SPACE else ['-', '-']
        else:
            raise NotImplementedError(
                "[ERROR] in Quadrotor._set_action_space(), quad_type not supported."
            )
        if self.NORMALIZED_RL_ACTION_SPACE:
            # normalized thrust (around hover thrust)
            self.hover_thrust = self.GRAVITY_ACC * self.MASS / action_dim
            self.action_space = spaces.Box(low=-np.ones(action_dim), 
                                           high=np.ones(action_dim), 
                                           dtype=np.float32)
        else:
            # direct thrust control 
            n_mot = 4 / action_dim
            a_low = self.KF * n_mot * (self.PWM2RPM_SCALE * self.MIN_PWM + self.PWM2RPM_CONST)**2
            a_high = self.KF * n_mot * (self.PWM2RPM_SCALE * self.MAX_PWM + self.PWM2RPM_CONST)**2
            self.action_space = spaces.Box(low=np.full(action_dim, a_low, np.float32), 
                                           high=np.full(action_dim, a_high, np.float32), 
                                           dtype=np.float32)

    def _set_observation_space(self):
        """Returns the observation space of the environment.

        Returns:
            gym.spaces: The bounded observation (state) space, of size 2 or 6 depending on QUAD_TYPE.

        """
        self.x_threshold = 2
        self.z_threshold = 2
        self.theta_threshold_radians = 85 * math.pi / 180
        # Define obs/state bounds, labels and units.
        if self.QUAD_TYPE == QuadType.ONE_D:
            # obs/state = {z, z_dot}.
            low = np.array([self.GROUND_PLANE_Z * 2, -np.finfo(np.float32).max])
            high = np.array([self.z_threshold * 2, np.finfo(np.float32).max])
            self.STATE_LABELS = ['z', 'z_dot']
            self.STATE_UNITS = ['m', 'm/s']
        elif self.QUAD_TYPE == QuadType.TWO_D:
            # obs/state = {x, x_dot, z, z_dot, theta, theta_dot}.
            low = np.array([
                -self.x_threshold * 2, -np.finfo(np.float32).max,
                self.GROUND_PLANE_Z * 2, -np.finfo(np.float32).max,
                -self.theta_threshold_radians * 2, -np.finfo(np.float32).max
            ])
            high = np.array([
                self.x_threshold * 2, np.finfo(np.float32).max, 
                self.z_threshold * 2, np.finfo(np.float32).max, 
                self.theta_threshold_radians * 2, np.finfo(np.float32).max
            ])
            self.STATE_LABELS = ['x', 'x_dot', 'z', 'z_dot', 'theta', 'theta_dot']
            self.STATE_UNITS = ['m', 'm/s', 'm', 'm/s', 'rad', 'rad/s']
        # Define underlying state space in dynamics transition
        self.state_space = spaces.Box(low=low, high=high, dtype=np.float32)
            
        # Concatenate goal info for RL 
        if self.COST == Cost.RL_REWARD and self.TASK == Task.TRAJ_TRACKING:
            # include future goal state(s) 
            # e.g. horizon=1, obs = {state, state_target}
            mul = 1 + self.obs_goal_horizon
            low = np.concatenate([low] * mul)
            high = np.concatenate([high] * mul)
        elif self.COST == Cost.RL_REWARD and self.TASK == Task.STABILIZATION:
            low = np.concatenate([low] * 2)
            high = np.concatenate([high] * 2)
        # Define obs space exposed to the controller 
        # Note obs space is often different to state space for RL (with additional task info)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
    
    def _preprocess_control(self, action):
        """Converts the action passed to .step() into motors' RPMs (ndarray of shape (4,)).

        Args:
            action (ndarray): The raw action input, of size 1 or 2 depending on QUAD_TYPE.

        Returns:
            ndarray: The motors RPMs to apply to the quadrotor.

        """
        if self.NORMALIZED_RL_ACTION_SPACE:
            # rescale action to around hover thrust
            action = np.clip(action, self.action_space.low, self.action_space.high)
            thrust = (1 + self.norm_act_scale * action) * self.hover_thrust
        else:
            thrust = np.clip(action, self.action_space.low, self.action_space.high)
        if not np.array_equal(thrust, np.array(action)) and self.VERBOSE:
            print("[WARNING]: action was clipped in Quadrotor._preprocess_control().")
        self.current_preprocessed_action = thrust
        # Apply disturbances.
        if "action" in self.disturbances:
            thrust = self.disturbances["action"].apply(thrust, self)
        if self.adversary_disturbance == "action":
            thrust = thrust + self.adv_action
        # convert to quad motor rpm commands
        pwm = cmd2pwm(thrust, self.PWM2RPM_SCALE, self.PWM2RPM_CONST, self.KF, self.MIN_PWM, self.MAX_PWM)
        rpm = pwm2rpm(pwm, self.PWM2RPM_SCALE, self.PWM2RPM_CONST)
        return rpm

    def _get_observation(self):
        """Returns the current observation (state) of the environment.

        Returns:
            ndarray: The state of the quadrotor, of size 2 or 6 depending on QUAD_TYPE.

        """
        full_state = self._get_drone_state_vector(0)
        pos, _, rpy, vel, ang_v, _ = np.split(full_state, [3, 7, 10, 13, 16])
        if self.QUAD_TYPE == QuadType.ONE_D:
            # {z, z_dot}.
            self.state = np.hstack([pos[2], vel[2]]).reshape((2,))
        elif self.QUAD_TYPE == QuadType.TWO_D:
            # {x, x_dot, z, z_dot, theta, theta_dot}.
            self.state = np.hstack(
                [pos[0], vel[0], pos[2], vel[2], rpy[1], ang_v[1]]
            ).reshape((6,))
        # if not np.array_equal(self.state,
        #                       np.clip(self.state, self.observation_space.low, self.observation_space.high)):
        #     if self.GUI and self.VERBOSE:
        #         print(
        #             "[WARNING]: observation was clipped in Quadrotor._get_observation()."
        #         )
        # Apply observation disturbance.
        obs = deepcopy(self.state)
        if "observation" in self.disturbances:
            obs = self.disturbances["observation"].apply(obs, self) 
        # Concatenate goal info (goal state(s)) for RL 
        if self.COST == Cost.RL_REWARD and self.TASK == Task.TRAJ_TRACKING:            
            # increment by 1 since counter is post-updated after _get_observation(),
            # obs should contain goal state desired for the next state
            next_step = self.ctrl_step_counter + 1 
            wp_idx = [
                min(next_step + i, self.X_GOAL.shape[0]-1) 
                for i in range(self.obs_goal_horizon)
            ]
            goal_state = self.X_GOAL[wp_idx].flatten()
            obs = np.concatenate([obs, goal_state])
        elif self.COST == Cost.RL_REWARD and self.TASK == Task.STABILIZATION:
            goal_state = self.X_GOAL.flatten()
            obs = np.concatenate([obs, goal_state])
        return obs

    def _get_reward(self):
        """Computes the current step's reward value.

        Returns:
            float: The evaluated reward/cost.

        """
        # RL cost.
        if self.COST == Cost.RL_REWARD:
            state = self.state
            act = np.asarray(self.current_preprocessed_action)
            # quadratic costs w.r.t state and action
            # TODO: consider using action error with goal action the hover thrust
            # TODO: consider using multiple future goal states for cost in tracking
            if self.TASK == Task.STABILIZATION:
                state_error = state - self.X_GOAL
                dist = np.sum(self.rew_state_weight * state_error * state_error)
                dist += np.sum(self.rew_act_weight * act * act)
            if self.TASK == Task.TRAJ_TRACKING:
                wp_idx = min(self.ctrl_step_counter, self.X_GOAL.shape[0]-1)
                state_error = state - self.X_GOAL[wp_idx]
                dist = np.sum(self.rew_state_weight * state_error * state_error)
                dist += np.sum(self.rew_act_weight * act * act)
            rew = -dist
            # convert rew to be positive and bounded [0,1]
            if self.rew_exponential:
                rew = np.exp(rew)
            return rew 
        # Control cost.
        if self.COST == Cost.QUADRATIC:
            if self.TASK == Task.STABILIZATION:
                return float(-1 * self.symbolic.loss(x=self.state,
                                                     Xr=self.X_GOAL,
                                                     u=self.current_preprocessed_action,
                                                     Ur=self.U_GOAL,
                                                     Q=self.Q,
                                                     R=self.R)["l"])
            if self.TASK == Task.TRAJ_TRACKING:
                return float(-1 * self.symbolic.loss(x=self.state,
                                                     Xr=self.X_GOAL[self.ctrl_step_counter,:],
                                                     u=self.current_preprocessed_action,
                                                     Ur=self.U_GOAL,
                                                     Q=self.Q,
                                                     R=self.R)["l"])

    def _get_done(self):
        """Computes the conditions for termination of an episode.

        Returns:
            bool: Whether an episode is over.

        """
        # Done if goal reached for stabilization task with quadratic cost.
        if self.TASK == Task.STABILIZATION and self.COST == Cost.QUADRATIC:
            self.goal_reached = bool(np.linalg.norm(self.state - self.X_GOAL) < self.TASK_INFO["stabilization_goal_tolerance"])
            if self.goal_reached:
                return True
        # # Done if the episode length is exceeded.
        # if (self.ctrl_step_counter + 1) / self.CTRL_FREQ >= self.EPISODE_LEN_SEC:
        #     return True
        # # Done if a constraint is violated.
        # if self.constraints is not None:
        #     if self.DONE_ON_VIOLATION and self.constraints.is_violated(self):
        #         return True
        # Done if state is out-of-bounds.
        if self.done_on_out_of_bound:
            if self.QUAD_TYPE == QuadType.ONE_D:
                z, _ = self.state
                out_of_bound = bool(z < -self.z_threshold 
                                    or z > self.z_threshold)
            if self.QUAD_TYPE == QuadType.TWO_D:
                x, _, z, _, theta, _ = self.state
                out_of_bound = bool(x < -self.x_threshold 
                                    or x > self.x_threshold 
                                    or z < -self.z_threshold 
                                    or z > self.z_threshold 
                                    or theta < -self.theta_threshold_radians 
                                    or theta > self.theta_threshold_radians)
            # early terminate if needed, but does not return False right away, 
            # allowing other done conditions to be chained afterwards
            if out_of_bound:
                return True 
        return False 

    def _get_info(self):
        """Generates the info dictionary returned by every call to .step().

        Returns:
            dict: A dictionary with information about the constraints evaluations and violations.

        """
        info = {}
        if self.TASK == Task.STABILIZATION and self.COST == Cost.QUADRATIC:
            info["goal_reached"] = self.goal_reached  # Add boolean flag for the goal being reached.
        # Add MSE.
        state = deepcopy(self.state) 
        if self.TASK == Task.STABILIZATION:
            state_error = state - self.X_GOAL
        elif self.TASK == Task.TRAJ_TRACKING:
            # TODO: should use angle wrapping  
            # state[4] = normalize_angle(state[4])
            wp_idx = min(self.ctrl_step_counter, self.X_GOAL.shape[0]-1)
            state_error = state - self.X_GOAL[wp_idx]
        # filter only relevant dimensions 
        state_error = state_error * self.info_mse_metric_state_weight
        info["mse"] = np.sum(state_error ** 2)
        # if self.constraints is not None:
        #     info["constraint_values"] = self.constraints.get_values(self)
        #     info["constraint_violations"] = self.constraints.get_violations(self)
        return info

    def _get_reset_info(self):
        """Generates the info dictionary returned by every call to .reset().

        Returns:
            dict: A dictionary with information about the dynamics and constraints symbolic models.

        """
        info = {}
        info["symbolic_model"] = self.symbolic
        info["physical_parameters"] = {
            "quadrotor_mass": self.MASS,
            "quadrotor_iyy_inertia": self.J[1, 1]
        }
        info["x_reference"] = self.X_GOAL
        info["u_reference"] = self.U_GOAL
        if self.constraints is not None:
            info["symbolic_constraints"] = self.constraints.get_all_symbolic_models()
        return info
