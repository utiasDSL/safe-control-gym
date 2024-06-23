"""Quadrotor environment using PyBullet physics.

Based on UTIAS Dynamic Systems Lab's gym-pybullet-drones:
    * https://github.com/utiasDSL/gym-pybullet-drones
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
from safe_control_gym.envs.physics import PhysicsMode, Physics

import pybullet as p
import pybullet_data
from pathlib import Path
import logging
from safe_control_gym.envs.utils import map2pi
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from safe_control_gym.envs.constraints import ConstraintList
from safe_control_gym.envs.disturbances import DisturbanceList
from safe_control_gym.envs.drone import Drone
from safe_control_gym.envs.physics import Constants

from safe_control_gym.math_and_models.symbolic_systems import SymbolicModel
from safe_control_gym.envs.quadrotor_utils import thrust2pwm
from safe_control_gym.envs.symbolic import symbolic

logger = logging.getLogger(__name__)


@dataclass
class SimSettings:
    """Simulation settings dataclass."""

    sim_freq: int = 500
    ctrl_freq: int = 500
    gui: bool = False
    pybullet_id: int = 0
    # Camera settings
    render_resolution: tuple[int, int] = (640, 480)
    camera_view: tuple[float, ...] = (0,) * 16
    camera_projection: tuple[float, ...] = (0,) * 16

    def __post_init__(self):
        assert self.sim_freq % self.ctrl_freq == 0, "sim_freq must be divisible by ctrl_freq."
        self.camera_projection = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=self.render_resolution[0] / self.render_resolution[1],
            nearVal=0.1,
            farVal=1000.0,
        )
        self.camera_view = p.computeViewMatrixFromYawPitchRoll(
            distance=3,
            yaw=-30,
            pitch=-30,
            roll=0,
            cameraTargetPosition=[0, 0, 0],
            upAxisIndex=2,
            physicsClientId=0,
        )


class DroneSim(gymnasium.Env):
    """Drone simulation based on gym-pybullet-drones."""

    URDF_DIR = Path(__file__).resolve().parent / "assets"

    def __init__(
        self,
        track: dict,
        sim_freq: int = 500,
        ctrl_freq: int = 500,
        constraints: list[dict] = [],
        disturbances: dict = {},
        randomization: dict = {},
        gui: bool = False,
        camera_view: tuple[float, ...] = (5.0, -40.0, -40.0, 0.5, -1.0, 0.5),
        n_drones: int = 1,
        physics: PhysicsMode = PhysicsMode.PYB,
    ):
        """Initialization method for BenchmarkEnv.

        Args:
            gui: Option to show PyBullet's GUI.
            sim_freq: The frequency at which PyBullet steps (a multiple of ctrl_freq).
            ctrl_freq: The frequency at which the environment steps.
            constraints: Dictionary to specify the constraints being used.
            disturbances: Dictionary to specify disturbances being used.
        """
        assert n_drones == 1, "Only one drone is supported at the moment."
        self.drone = Drone(controller="mellinger")
        self.n_drones = n_drones
        self.pyb_client = p.connect(p.GUI if gui else p.DIRECT)
        self.settings = SimSettings(sim_freq, ctrl_freq, gui, pybullet_id=self.pyb_client)
        self.physics = Physics(self.pyb_client, 1 / sim_freq, PhysicsMode(physics))

        # Create action, observation and state spaces.
        min_thrust, max_thrust = self.drone.params.min_thrust, self.drone.params.max_thrust
        self.action_space = spaces.Box(low=min_thrust, high=max_thrust, shape=(4,))
        # pos in meters, rpy in radians, vel in m/s ang_vel in rad/s
        rpy_max = np.array([85 / 180 * np.pi, 85 / 180 * np.pi, np.pi], np.float32)  # Yaw unbounded
        max_flt = np.full(3, np.finfo(np.float32).max, np.float32)
        pos_low, pos_high = np.array([-5, -5, 0]), np.array([5, 5, 2.5])
        self.observation_space = spaces.Dict(
            {
                "pos": spaces.Box(low=pos_low, high=pos_high),
                "rpy": spaces.Box(low=-rpy_max, high=rpy_max),
                "vel": spaces.Box(low=-max_flt, high=max_flt),
                "ang_vel": spaces.Box(low=-max_flt, high=max_flt),
            }
        )
        # State space uses 64-bit floats for better compatibility with pycffirmware.
        self.state_space = spaces.Dict(
            {
                "pos": spaces.Box(low=pos_low, high=pos_high, dtype=np.float64),
                "rpy": spaces.Box(low=-rpy_max, high=rpy_max, dtype=np.float64),
                "vel": spaces.Box(low=-max_flt, high=max_flt, dtype=np.float64),
                "ang_vel": spaces.Box(low=-max_flt, high=max_flt, dtype=np.float64),
            }
        )
        self.constraints = ConstraintList.from_specs(
            self.state_space, self.action_space, constraints
        )
        self.disturbance_config = disturbances
        self.disturbances = self._setup_disturbances(disturbances)
        self.randomization = randomization
        if self.settings.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=camera_view[0],
                cameraYaw=camera_view[1],
                cameraPitch=camera_view[2],
                cameraTargetPosition=[camera_view[3], camera_view[4], camera_view[5]],
                physicsClientId=self.pyb_client,
            )

        assert isinstance(track.drone, dict), "Expected drone state as dictionary."
        for key, val in track.drone.items():
            assert key in self.drone.init_state, f"Unknown key '{key}' in drone state."
            self.drone.init_state[key][:] = val

        self.pyb_objects = {}  # Populated when objects are loaded into PyBullet.

        self.gates = {}
        if gates := track.get("gates"):
            self.gates = {i: g.toDict() for i, g in enumerate(gates)}
            # Add nominal values to not loose the default when randomizing.
            for i, gate in self.gates.items():
                self.gates[i].update({"nominal_" + k: v for k, v in gate.items()})
        self.n_gates = len(self.gates)

        self.obstacles = {}
        if obstacles := track.get("obstacles"):
            self.obstacles = {i: o.toDict() for i, o in enumerate(obstacles)}
            for i, obstacle in self.obstacles.items():
                self.obstacles[i].update({"nominal_" + k: v for k, v in obstacle.items()})
        self.n_obstacles = len(self.obstacles)

        # Helper variables
        self._recording = None  # PyBullet recording.

    def step(self, desired_thrust: npt.NDArray[np.float_]):
        """Advance the environment by one control step.

        Args:
            desired_thrust: The desired thrust for the drone.
        """
        self.drone.desired_thrust[:] = desired_thrust
        rpm = self._thrust_to_rpm(desired_thrust)  # Pre-process/clip the action
        disturb_force = np.zeros(3)
        if "dynamics" in self.disturbances:  # Add dynamics disturbance force.
            disturb_force = self.disturbances["dynamics"].apply(disturb_force)
        for _ in range(self.settings.sim_freq // self.settings.ctrl_freq):
            self.drone.rpm[:] = rpm  # Save the last applied action (e.g. to compute drag)
            force_torques = self.physics.calculate_force_torques(self.drone, rpm, [])
            self.physics.apply(self.drone, force_torques, disturb_force)
            self.physics.step(self.drone)
            self._sync_pyb_to_sim()

    def reset(self):
        """Reset the simulation to its original state."""
        for mode in self.disturbances.keys():
            self.disturbances[mode].reset()
        self.drone.reset()
        self._reset_pybullet()
        self._randomize_drone()
        self._sync_pyb_to_sim()

    @property
    def collisions(self) -> list[int]:
        """Return the pybullet object IDs of the objects currently in collision with the drone."""
        collisions = []
        for o_id in self.pyb_objects.values():
            if p.getContactPoints(bodyA=o_id, bodyB=self.drone.id, physicsClientId=self.pyb_client):
                collisions.append(o_id)
        return collisions

    def in_range(self, bodies: dict, target_body, distance: float) -> npt.NDArray[np.bool_]:
        """Return a mask array of objects within a certain distance of the drone."""
        in_range = np.zeros(len(bodies), dtype=bool)
        for i, body in enumerate(bodies.values()):
            assert "id" in body, "Body must have a PyBullet ID."
            closest_points = p.getClosestPoints(
                bodyA=body["id"],
                bodyB=target_body.id,
                distance=distance,
                physicsClientId=self.pyb_client,
            )
            in_range[i] = len(closest_points) > 0
        return in_range

    @property
    def state(self):
        # TODO: Remove in favor of the state property in the drone class.
        pos = self.drone.state["pos"]
        vel = self.drone.state["vel"]
        rpy = self.drone.state["rpy"]
        ang_vel = self.drone.state["ang_vel"]
        ang_v_body_frame = R.from_euler("XYZ", rpy).as_matrix().T @ ang_vel
        state = np.hstack(
            [pos[0], vel[0], pos[1], vel[1], pos[2], vel[2], rpy, ang_v_body_frame]
        ).reshape((12,))
        return state

    @property
    def n_constraints(self):
        return self.constraints.n_constraints

    def seed(self, seed: int | None = None):
        """Set up a random number generator for a given seed."""
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        for d in self.disturbances.values():
            d.seed(seed)
        return seed

    def render(self) -> npt.NDArray[np.uint8]:
        """Retrieve a frame from PyBullet rendering.

        Returns:
            The RGB frame captured by PyBullet's camera as [h, w, 4] tensor.
        """
        [w, h, rgb, _, _] = p.getCameraImage(
            width=self.settings.render_resolution[0],
            height=self.settings.render_resolution[1],
            shadow=1,
            viewMatrix=self.settings.camera_view,
            projectionMatrix=self.settings.camera_projection,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            physicsClientId=self.pyb_client,
        )
        return np.reshape(rgb, (h, w, 4))

    def record(self, path: Path):
        """Start the recording of a video output.

        The format of the video output is .mp4, if GUI is True, or .png, otherwise.
        The video is saved under folder `files/videos`.

        """
        if not self.settings.gui:
            logger.warning("Cannot record video without GUI.")
            return
        self._recording = p.startStateLogging(
            loggingType=p.STATE_LOGGING_VIDEO_MP4,
            fileName=str(path.absolute()),
            physicsClientId=self.pyb_client,
        )

    def close(self):
        """Terminates the environment."""
        if self._recording is not None:
            p.stopStateLogging(self._recording, physicsClientId=self.pyb_client)
        if self.pyb_client >= 0:
            p.disconnect(physicsClientId=self.pyb_client)
        self.pyb_client = -1

    def _setup_disturbances(self, disturbances: dict | None = None) -> dict[str, DisturbanceList]:
        """Creates attributes and action spaces for the disturbances."""
        dist = {}
        if disturbances is None:  # Default: no passive disturbances.
            return dist
        modes = {
            "observation": {"dim": spaces.flatdim(self.observation_space)},
            "action": {"dim": spaces.flatdim(self.action_space)},
            "dynamics": {"dim": 3},
        }
        for mode, spec in disturbances.items():
            assert mode in modes, "Disturbance mode not available."
            spec["dim"] = modes[mode]["dim"]
            dist[mode] = DisturbanceList.from_specs([spec])
        return dist

    def _reset_pybullet(self):
        """Reset PyBullet's simulation environment."""
        p.resetSimulation(physicsClientId=self.pyb_client)
        p.setGravity(0, 0, -Constants.GRAVITY, physicsClientId=self.pyb_client)
        p.setRealTimeSimulation(0, physicsClientId=self.pyb_client)
        p.setTimeStep(1 / self.settings.sim_freq, physicsClientId=self.pyb_client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.pyb_client)
        # Load ground plane, drone and obstacles models.
        i = p.loadURDF("plane.urdf", [0, 0, 0], physicsClientId=self.pyb_client)
        self.pyb_objects["plane"] = i

        self.drone.id = p.loadURDF(
            str(self.URDF_DIR / "cf2x.urdf"),
            self.drone.init_state["pos"],
            p.getQuaternionFromEuler(self.drone.init_state["rpy"]),
            flags=p.URDF_USE_INERTIA_FROM_FILE,  # Use URDF inertia tensor.
            physicsClientId=self.pyb_client,
        )
        # Remove default damping.
        p.changeDynamics(self.drone.id, -1, linearDamping=0, angularDamping=0)
        # Load obstacles into the simulation and perturb their poses if configured.
        self._reset_obstacles()
        self._reset_gates()
        self._sync_pyb_to_sim()

    def _reset_obstacles(self):
        """Reset the obstacles in the simulation."""
        for i, obstacle in self.obstacles.items():
            pos_offset = np.zeros(3)
            if obstacle_pos := self.randomization.get("obstacle_pos"):
                distrib = getattr(self.np_random, obstacle_pos.get("type"))
                kwargs = {k: v for k, v in obstacle_pos.items() if k != "type"}
                pos_offset = distrib(**kwargs)
            self.obstacles[i]["pos"] = np.array(obstacle["nominal_pos"]) + pos_offset
            self.obstacles[i]["id"] = self._load_urdf_into_sim(
                self.URDF_DIR / "obstacle.urdf",
                self.obstacles[i]["pos"] + pos_offset,
                marker=str(i),
            )
            self.pyb_objects[f"obstacle_{i}"] = self.obstacles[i]["id"]

    def _reset_gates(self):
        for i, gate in self.gates.items():
            pos_offset = np.zeros_like(gate["nominal_pos"])
            if gate_pos := self.randomization.get("gate_pos"):
                distrib = getattr(self.np_random, gate_pos.get("type"))
                pos_offset = distrib(**{k: v for k, v in gate_pos.items() if k != "type"})
            rpy_offset = np.zeros(3)
            if gate_rpy := self.randomization.get("gate_rpy"):
                distrib = getattr(self.np_random, gate_rpy.get("type"))
                rpy_offset = distrib(**{k: v for k, v in gate_rpy.items() if k != "type"})
            gate["pos"] = np.array(gate["nominal_pos"]) + pos_offset
            gate["rpy"] = map2pi(np.array(gate["nominal_rpy"]) + rpy_offset)  # Ensure [-pi, pi]
            gate["id"] = self._load_urdf_into_sim(
                self.URDF_DIR / "gate.urdf", gate["pos"], gate["rpy"], marker=str(i)
            )
            self.pyb_objects[f"gate_{i}"] = gate["id"]

    def _load_urdf_into_sim(
        self,
        urdf_path: Path,
        pos: npt.NDArray[np.float_],
        rpy: npt.NDArray[np.float_] | None = None,
        marker: str | None = None,
    ) -> int:
        """Load a URDF file into the simulation.

        Args:
            urdf_path: Path to the URDF file.
            pos: Position of the object in the simulation.
            rpy: Roll, pitch, yaw orientation of the object in the simulation.
            marker: Optional text marker to display above the object

        Returns:
            int: The ID of the object in the simulation.
        """
        quat = p.getQuaternionFromEuler(rpy if rpy is not None else np.zeros(3))
        pyb_id = p.loadURDF(str(urdf_path), pos, quat, physicsClientId=self.pyb_client)
        if marker is not None:
            p.addUserDebugText(
                str(marker),
                textPosition=[0, 0, 0.5],
                textColorRGB=[1, 0, 0],
                textSize=1.5,
                parentObjectUniqueId=pyb_id,
                parentLinkIndex=-1,
                physicsClientId=self.pyb_client,
            )
        return pyb_id

    def _randomize_drone(self):
        """Randomize the drone's position, orientation and physical properties."""
        inertia_diag = self.drone.nominal_params.J.diagonal()
        if drone_inertia := self.randomization.get("drone_inertia"):
            distrib = getattr(self.np_random, drone_inertia.type)
            kwargs = {k: v for k, v in drone_inertia.items() if k != "type"}
            inertia_diag = inertia_diag + distrib(**kwargs)
            assert all(inertia_diag > 0), "Negative randomized inertial properties."
        self.drone.params.J = np.diag(inertia_diag)

        mass = self.drone.nominal_params.mass
        if drone_mass := self.randomization.get("drone_mass"):
            distrib = getattr(self.np_random, drone_mass.type)
            mass += distrib(**{k: v for k, v in drone_mass.items() if k != "type"})
            assert mass > 0, "Negative randomized mass."
        self.drone.params.mass = mass

        p.changeDynamics(
            self.drone.id,
            linkIndex=-1,  # Base link.
            mass=mass,
            localInertiaDiagonal=inertia_diag,
            physicsClientId=self.pyb_client,
        )

        pos = self.drone.init_state["pos"]
        if drone_pos := self.randomization.get("drone_pos"):
            distrib = getattr(self.np_random, drone_pos.type)
            pos += distrib(**{k: v for k, v in drone_pos.items() if k != "type"})

        rpy = self.drone.init_state["rpy"]
        if drone_rpy := self.randomization.get("drone_rpy"):
            distrib = getattr(self.np_random, drone_rpy.type)
            kwargs = {k: v for k, v in drone_rpy.items() if k != "type"}
            rpy = np.clip(rpy + distrib(**kwargs), -np.pi, np.pi)

        p.resetBasePositionAndOrientation(
            self.drone.id,
            pos,
            p.getQuaternionFromEuler(rpy),
            physicsClientId=self.pyb_client,
        )
        p.resetBaseVelocity(self.drone.id, [0, 0, 0], [0, 0, 0], physicsClientId=self.pyb_client)

    def _sync_pyb_to_sim(self):
        """Read state values from PyBullet and write them to the simulator class.

        We cache the state values in the simulator class to avoid calling PyBullet too frequently.
        """
        pos, quat = p.getBasePositionAndOrientation(self.drone.id, physicsClientId=self.pyb_client)
        self.drone.state["pos"] = np.array(pos, float)
        self.drone.state["rpy"] = np.array(p.getEulerFromQuaternion(quat), float)
        vel, ang_vel = p.getBaseVelocity(self.drone.id, physicsClientId=self.pyb_client)
        self.drone.state["vel"] = np.array(vel, float)
        self.drone.state["ang_vel"] = np.array(ang_vel)

    def symbolic(self) -> SymbolicModel:
        """Create a symbolic (CasADi) model for dynamics, observation, and cost.

        Returns:
            CasADi symbolic model of the environment.
        """
        return symbolic(self.drone, 1 / self.settings.sim_freq)

    def _thrust_to_rpm(self, thrust: npt.NDArray[np.float_]):
        """Convert the desired_thrust into motors' RPMs.

        Args:
            thrust: The desired thrust per motor.

        Returns:
            The motors' RPMs to apply to the quadrotor.
        """
        thrust = np.clip(thrust, self.drone.params.min_thrust, self.drone.params.max_thrust)
        if "action" in self.disturbances:
            thrust = self.disturbances["action"].apply(thrust)
        pwm = thrust2pwm(
            thrust,
            self.drone.params.pwm2rpm_scale,
            self.drone.params.pwm2rpm_const,
            self.drone.params.kf,
            self.drone.params.min_pwm,
            self.drone.params.max_pwm,
        )
        return self.drone.params.pwm2rpm_const + self.drone.params.pwm2rpm_scale * pwm
