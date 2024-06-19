"""Quadrotor environment using PyBullet physics.

Based on UTIAS Dynamic Systems Lab's gym-pybullet-drones:
    * https://github.com/utiasDSL/gym-pybullet-drones

"""

import os
import logging
from copy import deepcopy
import casadi as cs
import numpy as np
import pybullet as p

from safe_control_gym.math_and_models.symbolic_systems import SymbolicModel
from safe_control_gym.envs.base_aviary import BaseAviary
from safe_control_gym.envs.quadrotor_utils import QuadType, cmd2pwm, pwm2rpm
from safe_control_gym.math_and_models.transformations import csRotXYZ
from gymnasium import spaces

logger = logging.getLogger(__name__)


class Quadrotor(BaseAviary):
    """1D and 2D quadrotor environment task.

    Including symbolic model, constraints, randomization, adversarial disturbances,
    multiple cost functions, stabilization and trajectory tracking references.

    """

    GATE_Z_LOW = 0.525
    GATE_Z_HIGH = 1.0
    OBSTACLE_Z = 1.05

    def __init__(
        self,
        init_state=None,
        # custom args
        quad_type: QuadType = QuadType.TWO_D,
        norm_act_scale=0.1,
        obs_goal_horizon=0,
        rew_state_weight=1.0,
        rew_act_weight=0.0001,
        rew_exponential=True,
        done_on_out_of_bound=True,
        **kwargs,
    ):
        """Initialize a quadrotor environment.

        Args:
            init_state (ndarray, optional): The initial state of the environment, (z, z_dot) or (x, x_dot, z, z_dot theta, theta_dot).
            quad_type (QuadType, optional): The choice of motion type (1D along z or 2D in the x-z plane).
            norm_act_scale (float): scaling the [-1,1] action space around hover thrust when `normalized_action_space` is True.
            obs_goal_horizon (int): how many future goal states to append to obervation.
            rew_state_weight (list/ndarray): quadratic weights for state in rl reward.
            rew_act_weight (list/ndarray): quadratic weights for action in rl reward.
            rew_exponential (bool): if to exponentiate negative quadratic cost to positive, bounded [0,1] reward.
            done_on_out_of_bound (bool): if to termiante when state is out of bound.

        """
        self._init_state = init_state
        self._init_state_randomization = kwargs["init_state_randomization_info"]
        self._enable_init_state_randomization = kwargs["randomized_init"]

        self._inertial = None
        self._inertial_randomization = kwargs["inertial_prop_randomization_info"]
        self._enable_inertial_randomization = kwargs["randomized_inertial_prop"]

        # Select the 1D (moving along z) or 2D (moving in the xz plane) quadrotor.
        self.QUAD_TYPE = QuadType(quad_type)
        self.norm_act_scale = norm_act_scale
        self.obs_goal_horizon = obs_goal_horizon
        self.rew_state_weight = np.array(rew_state_weight, ndmin=1, dtype=float)
        self.rew_act_weight = np.array(rew_act_weight, ndmin=1, dtype=float)
        self.rew_exponential = rew_exponential
        self.done_on_out_of_bound = done_on_out_of_bound

        # BaseAviary constructor, called after defining the custom args,
        # since some BenchmarkEnv init setup can be task(custom args)-dependent.
        super().__init__(init_state=init_state, **kwargs)

        # Store initial state info.
        self.INIT_STATE_LABELS = [
            "init_x",
            "init_x_dot",
            "init_y",
            "init_y_dot",
            "init_z",
            "init_z_dot",
            "init_phi",
            "init_theta",
            "init_psi",
            "init_p",
            "init_q",
            "init_r",
        ]

        assert isinstance(init_state, dict), "Expected init_state as dictionary."
        for init_name in self.INIT_STATE_LABELS:
            self.__dict__[init_name.upper()] = init_state.get(init_name, 0.0)

        # Set prior/symbolic info.
        self._setup_symbolic()

        # Equilibrium point at hover for linearization.
        self.X_EQ = np.zeros(spaces.flatdim(self.state_space))

        # IROS 2022 - Load maze.
        self.OBSTACLES = []
        self.GATES = []
        if "obstacles" in kwargs:
            self.OBSTACLES = kwargs["obstacles"]
        self.OBSTACLES = np.array(self.OBSTACLES)
        if "gates" in kwargs:
            self.GATES = kwargs["gates"]
        self.GATES = np.array(self.GATES)
        self.NUM_GATES = len(self.GATES)
        self.n_obstacles = len(self.OBSTACLES)
        if kwargs.get("randomized_gates_and_obstacles", False):
            self.RANDOMIZED_GATES_AND_OBS = True
            if "gates_and_obstacles_randomization_info" not in kwargs:
                raise ValueError(
                    "Missing 'gates_and_obstacles_randomization_info' in YAML configuration."
                )
            self.GATES_AND_OBS_RAND_INFO = kwargs["gates_and_obstacles_randomization_info"]
        else:
            self.RANDOMIZED_GATES_AND_OBS = False

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

        # IROS 2022 - Create maze.
        self.OBSTACLES_IDS = []
        self.obstacle_poses = []
        if self.RANDOMIZED_GATES_AND_OBS:
            rand_info_copy = deepcopy(self.GATES_AND_OBS_RAND_INFO)
            distrib = getattr(self.np_random, rand_info_copy["obstacles"].pop("distrib"))
            d_args = rand_info_copy["obstacles"].pop("args", [])
            d_kwargs = rand_info_copy["obstacles"]
        for obstacle in self.OBSTACLES:
            obs_height = 0.525  # URDF dependent, places 'obstacle.urdf' at z == 0.
            if self.RANDOMIZED_GATES_AND_OBS:
                offset = np.array(
                    [distrib(*d_args, **d_kwargs), distrib(*d_args, **d_kwargs), obs_height]
                )
                pose_disturbance = np.array([0, 0, distrib(*d_args, **d_kwargs)])
            else:
                offset = np.array([0, 0, obs_height])
                pose_disturbance = np.array([0, 0, 0])
            self.obstacle_poses.append(
                list(np.array(obstacle[0:3]) + offset)
                + list(np.array(obstacle[3:6]) + pose_disturbance)
            )
            TMP_ID = p.loadURDF(
                os.path.join(self.URDF_DIR, "obstacle.urdf"),
                np.array(obstacle[0:3]) + offset,
                p.getQuaternionFromEuler(np.array(obstacle[3:6]) + pose_disturbance),
                physicsClientId=self.PYB_CLIENT,
            )
            p.addUserDebugText(
                str(TMP_ID),
                textPosition=[0, 0, 0.5],
                textColorRGB=[1, 0, 0],
                textSize=1.5,
                parentObjectUniqueId=TMP_ID,
                parentLinkIndex=-1,
                physicsClientId=self.PYB_CLIENT,
            )
            self.OBSTACLES_IDS.append(TMP_ID)
        self.obstacle_poses = np.array(self.obstacle_poses)
        self.obstacle_poses[:, 2] = self.OBSTACLE_Z
        self.GATES_IDS = []
        self._gates_pose = []
        if self.RANDOMIZED_GATES_AND_OBS:
            rand_info_copy = deepcopy(self.GATES_AND_OBS_RAND_INFO)
            distrib = getattr(self.np_random, rand_info_copy["gates"].pop("distrib"))
            d_args = rand_info_copy["gates"].pop("args", [])
            d_kwargs = rand_info_copy["gates"]
        for gate in self.GATES:
            if gate[6] == 0:
                urdf_file = "portal.urdf"
                gate_height = self.GATE_Z_HIGH  # URDF dependent, places 'portal.urdf' at z == 0.
            elif gate[6] == 1:
                urdf_file = "low_portal.urdf"
                gate_height = self.GATE_Z_LOW  # URDF dependent, places 'low_portal.urdf' at z == 0.
            else:
                raise ValueError("[ERROR] Unknown gate type.")
            if self.RANDOMIZED_GATES_AND_OBS:
                offset = np.array(
                    [distrib(*d_args, **d_kwargs), distrib(*d_args, **d_kwargs), gate_height]
                )
                pose_disturbance = np.array([0, 0, distrib(*d_args, **d_kwargs)])
            else:
                offset = np.array([0, 0, gate_height])
                pose_disturbance = np.array([0, 0, 0])
            self._gates_pose.append(
                list(np.array(gate[0:3]) + offset) + list(np.array(gate[3:6]) + pose_disturbance)
            )
            TMP_ID = p.loadURDF(
                os.path.join(self.URDF_DIR, urdf_file),
                np.array(gate[0:3]) + offset,
                p.getQuaternionFromEuler(np.array(gate[3:6]) + pose_disturbance),
                physicsClientId=self.PYB_CLIENT,
            )
            p.addUserDebugText(
                str(TMP_ID),
                textPosition=[0, 0, 0.5],
                textColorRGB=[1, 0, 0],
                textSize=1.5,
                parentObjectUniqueId=TMP_ID,
                parentLinkIndex=-1,
                physicsClientId=self.PYB_CLIENT,
            )
            self.GATES_IDS.append(TMP_ID)
        self._gates_pose = np.array(self._gates_pose)
        self.current_gate = 0

        # Initialize IROS-specific attributes.
        self.stepped_through_gate = False
        self.currently_collided = False
        self.at_goal_pos = False
        self.steps_at_goal_pos = 0
        self.task_completed = False

        # Choose randomized or deterministic inertial properties.
        prop_values = {
            "M": self.drone.params.mass,
            "Ixx": self.J[0, 0],
            "Iyy": self.J[1, 1],
            "Izz": self.J[2, 2],
        }
        if self._enable_inertial_randomization:
            prop_values = self._randomize_values_by_info(prop_values, self._inertial_randomization)
            if any(phy_quantity < 0 for phy_quantity in prop_values.values()):
                raise ValueError(
                    "[ERROR] in Quadrotor.reset(), negative randomized inertial properties."
                )
        self.OVERRIDDEN_QUAD_MASS = prop_values["M"]
        self.OVERRIDDEN_QUAD_INERTIA = [prop_values["Ixx"], prop_values["Iyy"], prop_values["Izz"]]

        # Override inertial properties.
        p.changeDynamics(
            self.DRONE_IDS[0],
            linkIndex=-1,  # Base link.
            mass=self.OVERRIDDEN_QUAD_MASS,
            localInertiaDiagonal=self.OVERRIDDEN_QUAD_INERTIA,
            physicsClientId=self.PYB_CLIENT,
        )

        # Randomize initial state.
        init_values = {
            init_name: self.__dict__[init_name.upper()] for init_name in self.INIT_STATE_LABELS
        }
        if self._enable_init_state_randomization:
            init_values = self._randomize_values_by_info(
                init_values, self._init_state_randomization
            )
        INIT_XYZ = [init_values.get("init_" + k, 0.0) for k in ["x", "y", "z"]]
        INIT_VEL = [init_values.get("init_" + k + "_dot", 0.0) for k in ["x", "y", "z"]]
        INIT_RPY = [init_values.get("init_" + k, 0.0) for k in ["phi", "theta", "psi"]]
        INIT_ANG_VEL = [init_values.get("init_" + k, 0.0) for k in ["p", "q", "r"]]
        p.resetBasePositionAndOrientation(
            self.DRONE_IDS[0],
            INIT_XYZ,
            p.getQuaternionFromEuler(INIT_RPY),
            physicsClientId=self.PYB_CLIENT,
        )
        p.resetBaseVelocity(
            self.DRONE_IDS[0], INIT_VEL, INIT_ANG_VEL, physicsClientId=self.PYB_CLIENT
        )

        # Update BaseAviary internal variables before calling self._get_observation().
        self._update_and_store_kinematic_information()
        obs, info = self._get_observation(), self._get_reset_info()

        # Return either an observation and dictionary or just the observation.
        return obs, info

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
        self.current_raw_input_action = action
        rpm = self._preprocess_control(action)  # Pre-process/clip the action

        # Determine disturbance force.
        disturb_force = None
        passive_disturb = "dynamics" in self.disturbances
        if passive_disturb:
            disturb_force = np.zeros(3)
            disturb_force = self.disturbances["dynamics"].apply(disturb_force, self)
        # Construct full (3D) disturbance force.
        if disturb_force is not None:
            disturb_force = np.asarray(disturb_force).flatten()

        # Advance the simulation.
        super()._advance_simulation(rpm, disturb_force)
        # Standard Gym return.
        obs = self._get_observation()
        info = self._get_info()
        # IROS 2022 - After _get_info() to use this step's 'self' attributes.
        done = self._get_done()
        rew = self._get_reward()
        obs, rew, done, info = super().after_step(obs, rew, done, info)
        return obs, rew, done, info

    def render(self, mode="human"):
        """Retrieves a frame from PyBullet rendering.

        Args:
            mode (str): Unused.

        Returns:
            ndarray: A multidimensional array with the RGB frame captured by PyBullet's camera.

        """
        [w, h, rgb, dep, seg] = p.getCameraImage(
            width=self.RENDER_WIDTH,
            height=self.RENDER_HEIGHT,
            shadow=1,
            viewMatrix=self.CAM_VIEW,
            projectionMatrix=self.CAM_PRO,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            physicsClientId=self.PYB_CLIENT,
        )
        return np.reshape(rgb, (h, w, 4))

    def _setup_symbolic(self):
        """Creates symbolic (CasADi) models for dynamics, observation, and cost.

        Returns:
            SymbolicModel: CasADi symbolic model of the environment.

        """
        m, g, l = self.drone.params.mass, self.GRAVITY_ACC, self.L
        Iyy = self.J[1, 1]
        dt = 1 / self.sim_settings.ctrl_freq
        # Define states.
        z = cs.MX.sym("z")
        z_dot = cs.MX.sym("z_dot")

        # Set up the dynamics model for a 3D quadrotor.
        nx, nu = 12, 4
        Ixx = self.J[0, 0]
        Izz = self.J[2, 2]
        J = cs.blockcat([[Ixx, 0.0, 0.0], [0.0, Iyy, 0.0], [0.0, 0.0, Izz]])
        Jinv = cs.blockcat([[1.0 / Ixx, 0.0, 0.0], [0.0, 1.0 / Iyy, 0.0], [0.0, 0.0, 1.0 / Izz]])
        gamma = self.KM / self.KF
        x = cs.MX.sym("x")
        y = cs.MX.sym("y")
        phi = cs.MX.sym("phi")  # Roll
        theta = cs.MX.sym("theta")  # Pitch
        psi = cs.MX.sym("psi")  # Yaw
        x_dot = cs.MX.sym("x_dot")
        y_dot = cs.MX.sym("y_dot")
        p = cs.MX.sym("p")  # Body frame roll rate
        q = cs.MX.sym("q")  # body frame pith rate
        r = cs.MX.sym("r")  # body frame yaw rate
        # PyBullet Euler angles use the SDFormat for rotation matrices.
        Rob = csRotXYZ(
            phi, theta, psi
        )  # rotation matrix transforming a vector in the body frame to the world frame.

        # Define state variables.
        X = cs.vertcat(x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r)

        # Define inputs.
        f1 = cs.MX.sym("f1")
        f2 = cs.MX.sym("f2")
        f3 = cs.MX.sym("f3")
        f4 = cs.MX.sym("f4")
        U = cs.vertcat(f1, f2, f3, f4)

        # From Ch. 2 of Luis, Carlos, and Jérôme Le Ny. "Design of a trajectory tracking
        # controller for a nanoquadcopter." arXiv preprint arXiv:1608.05786 (2016).

        # Defining the dynamics function.
        # We are using the velocity of the base wrt to the world frame expressed in the world
        # frame. Note that the reference expresses this in the body frame.
        oVdot_cg_o = Rob @ cs.vertcat(0, 0, f1 + f2 + f3 + f4) / m - cs.vertcat(0, 0, g)
        pos_ddot = oVdot_cg_o
        pos_dot = cs.vertcat(x_dot, y_dot, z_dot)
        Mb = cs.vertcat(
            l / cs.sqrt(2.0) * (f1 + f2 - f3 - f4),
            l / cs.sqrt(2.0) * (-f1 + f2 + f3 - f4),
            gamma * (f1 - f2 + f3 - f4),
        )
        rate_dot = Jinv @ (Mb - (cs.skew(cs.vertcat(p, q, r)) @ J @ cs.vertcat(p, q, r)))
        ang_dot = cs.blockcat(
            [
                [1, cs.sin(phi) * cs.tan(theta), cs.cos(phi) * cs.tan(theta)],
                [0, cs.cos(phi), -cs.sin(phi)],
                [0, cs.sin(phi) / cs.cos(theta), cs.cos(phi) / cs.cos(theta)],
            ]
        ) @ cs.vertcat(p, q, r)
        X_dot = cs.vertcat(
            pos_dot[0],
            pos_ddot[0],
            pos_dot[1],
            pos_ddot[1],
            pos_dot[2],
            pos_ddot[2],
            ang_dot,
            rate_dot,
        )

        Y = cs.vertcat(x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r)

        # Define cost (quadratic form).
        Q = cs.MX.sym("Q", nx, nx)
        R = cs.MX.sym("R", nu, nu)
        Xr = cs.MX.sym("Xr", nx, 1)
        Ur = cs.MX.sym("Ur", nu, 1)
        cost_func = 0.5 * (X - Xr).T @ Q @ (X - Xr) + 0.5 * (U - Ur).T @ R @ (U - Ur)
        # Define dynamics and cost dictionaries.
        dynamics = {"dyn_eqn": X_dot, "obs_eqn": Y, "vars": {"X": X, "U": U}}
        cost = {
            "cost_func": cost_func,
            "vars": {"X": X, "U": U, "Xr": Xr, "Ur": Ur, "Q": Q, "R": R},
        }
        # Setup symbolic model.
        self.symbolic = SymbolicModel(dynamics=dynamics, cost=cost, dt=dt)

    def _preprocess_control(self, action):
        """Converts the action passed to .step() into motors' RPMs (ndarray of shape (4,)).

        Args:
            action (ndarray): The raw action input, of size 1 or 2 depending on QUAD_TYPE.

        Returns:
            ndarray: The motors RPMs to apply to the quadrotor.

        """
        thrust = np.clip(action, self.action_space.low, self.action_space.high)
        self.current_preprocessed_action = thrust
        # Apply disturbances.
        if "action" in self.disturbances:
            thrust = self.disturbances["action"].apply(thrust, self)
        # convert to quad motor rpm commands
        pwm = cmd2pwm(
            thrust, self.PWM2RPM_SCALE, self.PWM2RPM_CONST, self.KF, self.MIN_PWM, self.MAX_PWM
        )
        rpm = pwm2rpm(pwm, self.PWM2RPM_SCALE, self.PWM2RPM_CONST)
        return rpm

    def _get_observation(self):
        """Returns the current observation (state) of the environment.

        Returns:
            ndarray: The state of the quadrotor, of size 2 or 6 depending on QUAD_TYPE.

        """
        full_state = self._get_drone_state_vector(0)
        pos, _, rpy, vel, ang_v, _ = np.split(full_state, [3, 7, 10, 13, 16])

        Rob = np.array(p.getMatrixFromQuaternion(self.quat[0])).reshape((3, 3))
        Rbo = Rob.T
        ang_v_body_frame = Rbo @ ang_v
        # {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r}.
        self.state = np.hstack(
            [pos[0], vel[0], pos[1], vel[1], pos[2], vel[2], rpy, ang_v_body_frame]
        ).reshape((12,))
        # Apply observation disturbance.
        obs = deepcopy(self.state)
        if "observation" in self.disturbances:
            obs = self.disturbances["observation"].apply(obs, self)
        return {"pos": pos, "rpy": rpy, "vel": vel, "ang_vel": ang_v_body_frame}
        return obs

    def _get_reward(self):
        """Computes the current step's reward value.

        Returns:
            float: The evaluated reward/cost.

        """
        # IROS 2022 - Competition sparse reward signal.
        reward = 0
        # Reward for stepping through the (correct) next gate.
        if self.stepped_through_gate:
            reward += 100
        # Reward for reaching goal position (after navigating the gates in the correct order).
        if self.at_goal_pos:
            reward += 100
        # Penalize by collision.
        if self.currently_collided:
            reward -= 1000
        return reward

    def _get_done(self):
        """Computes the conditions for termination of an episode.

        Returns:
            bool: Whether an episode is over.

        """
        # Done if state is out-of-bounds.
        if self.done_on_out_of_bound:
            assert False
            mask = np.array([1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0])
            # Element-wise or to check out-of-bound conditions.
            out_of_bound = np.logical_or(
                self.state < self.state_space.low, self.state > self.state_space.high
            )
            # Mask out un-included dimensions (i.e. velocities)
            out_of_bound = np.any(out_of_bound * mask)
            # Early terminate if needed.
            if out_of_bound:
                return True
        if self.currently_collided:
            return True

        return False

    def _get_info(self):
        """Generates the info dictionary returned by every call to .step().

        Returns:
            dict: A dictionary with information about the constraints evaluations and violations.

        """
        info = {}
        # IROS 2022 - Per-step info.
        # Collisions
        for GATE_OBS_ID in self.GATES_IDS + self.OBSTACLES_IDS + [self.PLANE_ID]:
            ret = p.getContactPoints(
                bodyA=GATE_OBS_ID,
                bodyB=self.DRONE_IDS[0],
                physicsClientId=self.PYB_CLIENT,
            )
            if ret:
                info["collision"] = (GATE_OBS_ID, True)
                self.currently_collided = True
                break  # Note: only returning the first collision per step.
        else:
            info["collision"] = (None, False)
            self.currently_collided = False
        #
        # Gates progress (note: allow 0.5 seconds for initial drop if objects are not on the gound).
        if (
            self.pyb_step_counter > 0.5 * self.sim_settings.sim_freq
            and self.NUM_GATES > 0
            and self.current_gate < self.NUM_GATES
        ):
            x, y, _, _, _, rot = self._gates_pose[self.current_gate]
            if self.GATES[self.current_gate][6] == 0:
                height = self.GATE_Z_HIGH  # URDF dependent.
            elif self.GATES[self.current_gate][6] == 1:
                height = self.GATE_Z_LOW  # URDF dependent.
            else:
                raise ValueError("Unknown gate type.")
            half_length = 0.1875  # Obstacle URDF dependent.
            delta_x = 0.05 * np.cos(rot)
            delta_y = 0.05 * np.sin(rot)
            fr = [[x, y, height - half_length]]
            to = [[x, y, height + half_length]]
            for i in [1, 2, 3]:
                fr.append([x + i * delta_x, y + i * delta_y, height - half_length])
                fr.append([x - i * delta_x, y - i * delta_y, height - half_length])
                to.append([x + i * delta_x, y + i * delta_y, height + half_length])
                to.append([x - i * delta_x, y - i * delta_y, height + half_length])
            rays = p.rayTestBatch(
                rayFromPositions=fr, rayToPositions=to, physicsClientId=self.PYB_CLIENT
            )
            self.stepped_through_gate = False
            for r in rays:
                if r[2] < 0.9999:
                    self.current_gate += 1
                    self.stepped_through_gate = True
                    break
        # LSY Drone Racing
        # Always add the nominal gate positions. If any gates are in range, update the position. If
        # any obstacles are in range, also update the obstacle positions.
        VISIBILITY_RANGE = 0.45
        info["gates_pose"] = self.GATES[:, :6].copy()
        info["gates_pose"][:, 2] = np.where(self.GATES[:, 6], self.GATE_Z_LOW, self.GATE_Z_HIGH)
        info["obstacles_pose"] = self.OBSTACLES.copy()
        info["obstacles_pose"][:, 2] = self.OBSTACLE_Z
        info["gates_in_range"] = np.zeros(self.NUM_GATES, dtype=bool)
        info["obstacles_in_range"] = np.zeros(self.n_obstacles, dtype=bool)
        for i in range(self.NUM_GATES):
            closest_points = p.getClosestPoints(
                bodyA=self.GATES_IDS[i],
                bodyB=self.DRONE_IDS[0],
                distance=VISIBILITY_RANGE,
                physicsClientId=self.PYB_CLIENT,
            )
            if len(closest_points) > 0:
                info["gates_pose"][i] = self._gates_pose[i]
                info["gates_in_range"][i] = True
        for i in range(self.n_obstacles):
            closest_points = p.getClosestPoints(
                bodyA=self.OBSTACLES_IDS[i],
                bodyB=self.DRONE_IDS[0],
                distance=VISIBILITY_RANGE,
                physicsClientId=self.PYB_CLIENT,
            )
            if len(closest_points) > 0:
                info["obstacles_pose"][i] = self.obstacle_poses[i]
                info["obstacles_in_range"][i] = True

        info["gates_type"] = self.GATES[:, 6]
        info["current_gate_id"] = self.current_gate if self.current_gate < self.NUM_GATES else -1

        # Final goal position reached
        info["at_goal_position"] = False
        info["task_completed"] = False
        return info

    def _get_reset_info(self):
        """Generates the info dictionary returned by every call to .reset().

        Returns:
            dict: A dictionary with information about the dynamics and constraints symbolic models.

        """
        info = {}
        info["symbolic_model"] = self.symbolic
        info["nominal_physical_parameters"] = {
            "quadrotor_mass": self.drone.params.mass,
            "quadrotor_ixx_inertia": self.J[0, 0],
            "quadrotor_iyy_inertia": self.J[1, 1],
            "quadrotor_izz_inertia": self.J[2, 2],
        }
        if self.constraints is not None:
            info["symbolic_constraints"] = self.constraints.symbolic_model()
        else:
            info["symbolic_constraints"] = {}

        # IROS 2022 - Reset info.
        info["ctrl_timestep"] = 1 / self.sim_settings.ctrl_freq
        info["ctrl_freq"] = self.sim_settings.ctrl_freq
        info["quadrotor_kf"] = self.KF
        info["quadrotor_km"] = self.KM
        info["gate_dimensions"] = {
            "tall": {"shape": "square", "height": self.GATE_Z_HIGH, "edge": 0.45},
            "low": {"shape": "square", "height": self.GATE_Z_LOW, "edge": 0.45},
        }
        info["obstacle_dimensions"] = {
            "shape": "cylinder",
            "height": self.OBSTACLE_Z,
            "radius": 0.05,
        }
        info["nominal_gates_pos_and_type"] = self.GATES
        info["nominal_obstacles_pos"] = self.OBSTACLES

        if self._enable_init_state_randomization:
            info["initial_state_randomization"] = self._init_state_randomization
        else:
            info["initial_state_randomization"] = {}
        if self._enable_inertial_randomization:
            info["inertial_prop_randomization"] = self._inertial_randomization
        else:
            info["inertial_prop_randomization"] = {}
        if self.RANDOMIZED_GATES_AND_OBS:
            info["gates_and_obs_randomization"] = self.GATES_AND_OBS_RAND_INFO
        else:
            info["gates_and_obs_randomization"] = {}
        info["disturbances"] = self._disturbance_config
        info["pyb_client"] = self.PYB_CLIENT
        info["urdf_dir"] = self.URDF_DIR
        info["constraint_values"] = self.constraints.value(self.state, only_state=True)

        info.update(self._get_info())
        return info
