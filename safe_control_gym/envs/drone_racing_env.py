from __future__ import annotations

import numpy as np
import logging

import gymnasium
from safe_control_gym.envs.drone_sim import DroneSim
from safe_control_gym.envs.drone import Drone
from scipy.spatial.transform import Rotation as R
import importlib.util
import pybullet as p
import time

spec = importlib.util.find_spec("pycffirmware")
firm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(firm)
# import pycffirmware as firm

logger = logging.getLogger(__name__)


class DroneRacingEnv(gymnasium.Env):
    CONTROLLER = "mellinger"  # specifies controller type

    def __init__(self, config: dict):
        """Initialize the DroneRacingEnv.

        Args:
            config: Configuration dictionary for the environment.

        Attributes:
            env: safe-control environment
        """
        super().__init__()
        self.config = config
        self.drone = Drone(self.CONTROLLER)
        self.step_freq = config.env.freq
        self.sim = DroneSim(
            track=config.env.track,
            sim_freq=config.sim.sim_freq,
            ctrl_freq=config.sim.ctrl_freq,
            constraints=config.env.constraints,
            disturbances=config.env.disturbances,
            gui=config.sim.gui,
            n_drones=1,
            physics=config.sim.physics,
        )
        self.sim.seed(config.env.seed)
        self.action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(4,))
        fm = np.finfo(np.float32).max
        low = np.array([-5, -fm, -5, -fm, -0.25, -fm, -np.pi, -np.pi, -np.pi, -fm, -fm, -fm])
        high = np.array([5, fm, 5, fm, 2.5, fm, np.pi, np.pi, np.pi, fm, fm, fm])
        self.observation_space = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)
        self.target_gate = 0
        self.symbolic = config.env.symbolic
        self._steps = 0
        self._debug_time = 0  # TODO: Remove this

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self.config.env.reseed:
            self.sim.seed(self.config.env.seed)
        if seed is not None:
            self.sim.seed(seed)
        self.sim.reset()
        self.target_gate = 0
        self._debug_time = 0  # TODO: Remove this
        self._steps = 0
        obs = self.obs
        pos = obs["pos"]
        rpy = obs["rpy"]
        vel = obs["vel"]
        ang_vel = obs["ang_vel"]
        self.drone.reset(pos, rpy, vel)
        obs = np.concatenate(
            [np.array([pos[0], vel[0], pos[1], vel[1], pos[2], vel[2]]), rpy, ang_vel]
        )
        if self.sim.n_drones > 1:
            raise NotImplementedError("Firmware wrapper does not support multiple drones.")
        return obs.astype(np.float32), self.info

    def step(self, action: np.ndarray):
        """Step the firmware_wrapper class and its environment.

        This function should be called once at the rate of ctrl_freq. Step processes and high level
        commands, and runs the firmware loop and simulator according to the frequencies set.

        Args:
            action: Full-state command [x_des, y_des, z_des, yaw_des] to follow.
        """
        zeros = np.zeros(3, dtype=np.float64)
        action = action.astype(np.float64)  # Drone firmware expects float64
        pos, yaw = action[:3], action[3]
        self.drone.full_state_cmd(pos, zeros, zeros, yaw, zeros)

        thrust = self.drone.desired_thrust
        collision = False
        while self.drone.tick / self.drone.firmware_freq < (self._steps + 1) / self.step_freq:
            self.sim.step(thrust)
            t1 = time.perf_counter()
            self.check_gate_progress()
            t2 = time.perf_counter()
            self._debug_time += t2 - t1
            pos = self.sim.drone.state["pos"]
            rpy = self.sim.drone.state["rpy"]
            vel = self.sim.drone.state["vel"]
            ang_vel = self.sim.drone.state["ang_vel"]

            obs = self.obs
            pos, vel, rpy, ang_vel = obs["pos"], obs["vel"], obs["rpy"], obs["ang_vel"]
            obs = np.concatenate(
                [np.array([pos[0], vel[0], pos[1], vel[1], pos[2], vel[2]]), rpy, ang_vel]
            )
            if self.sim.collisions:
                collision = True
            thrust = self.drone.step_controller(pos, rpy, vel)[::-1]
        self.sim.drone.desired_thrust[:] = thrust
        self._steps += 1
        terminated = self.terminated or collision
        if terminated:
            print(f"Time spent in check_gate_progress: {self._debug_time:.5f} s")
        return obs.astype(np.float32), self.reward, terminated, False, self.info

    @property
    def obs(self):
        obs = self.sim.drone.state.copy()
        # Convert angular velocity to body frame
        obs["ang_vel"] = R.from_euler("XYZ", obs["rpy"]).as_matrix().T @ obs["ang_vel"]
        if "observation" in self.sim.disturbances:
            obs = self.sim.disturbances["observation"].apply(obs)
        return obs

    @property
    def reward(self) -> float:
        return -1

    @property
    def terminated(self) -> bool:
        if self.sim.drone.state not in self.sim.state_space:
            return True  # Drone is out of bounds
        if self.sim.collisions:
            return True
        if self.target_gate == -1:  # Drone has passed all gates
            return True
        return False

    @property
    def info(self):
        info = {}
        VISIBILITY_RANGE = 0.45  # TODO: Make this a parameter
        info["collisions"] = self.sim.collisions
        gates = self.sim.gates
        info["target_gate"] = self.target_gate if self.target_gate < len(gates) else -1
        # Add the gate and obstacle poses to the info. If gates or obstacles are in sensor range,
        # use the actual pose, otherwise use the nominal pose.
        in_range = self.sim.in_range(gates, self.sim.drone, VISIBILITY_RANGE)
        gates_pos = np.stack([g["nominal_pos"] for g in gates.values()])
        gates_pos[in_range] = np.stack([g["pos"] for g in gates.values()])[in_range]
        gates_rpy = np.stack([g["nominal_rpy"] for g in gates.values()])
        gates_rpy[in_range] = np.stack([g["rpy"] for g in gates.values()])[in_range]
        info["gates.pos"] = gates_pos
        info["gates.rpy"] = gates_rpy
        info["gates.in_range"] = in_range

        obstacles = self.sim.obstacles
        in_range = self.sim.in_range(obstacles, self.sim.drone, VISIBILITY_RANGE)
        obstacles_pos = np.stack([o["nominal_pos"] for o in obstacles.values()])
        obstacles_pos[in_range] = np.stack([o["pos"] for o in obstacles.values()])[in_range]
        info["obstacles.pos"] = obstacles_pos
        info["obstacles.in_range"] = in_range

        c_value = self.sim.constraints.value(self.sim.state, self.sim.drone.desired_thrust)
        info["constraint.values"] = c_value
        info["constraint.violation"] = self.sim.constraints.is_violated(
            self.sim.state, self.sim.drone.desired_thrust, c_value=c_value
        )
        if self.symbolic:
            info["symbolic.model"] = self.sim.symbolic()
            info["symbolic.constraints"] = self.sim.constraints.symbolic()
        return info

    def check_gate_progress(self):
        # TODO: Check with an analytical solution instead of ray
        if self.sim.n_gates > 0 and self.target_gate < self.sim.n_gates and self.target_gate != -1:
            x, y, z = self.sim.gates[self.target_gate]["pos"]
            _, _, ry = self.sim.gates[self.target_gate]["rpy"]
            half_length = 0.1875  # Obstacle URDF dependent. TODO: Make this a parameter
            delta_x = 0.05 * np.cos(ry)
            delta_y = 0.05 * np.sin(ry)
            fr = [[x, y, z - half_length]]
            to = [[x, y, z + half_length]]
            for i in [1, 2, 3]:
                fr.append([x + i * delta_x, y + i * delta_y, z - half_length])
                fr.append([x - i * delta_x, y - i * delta_y, z - half_length])
                to.append([x + i * delta_x, y + i * delta_y, z + half_length])
                to.append([x - i * delta_x, y - i * delta_y, z + half_length])
            rays = p.rayTestBatch(
                rayFromPositions=fr, rayToPositions=to, physicsClientId=self.sim.pyb_client
            )
            if any(r[2] < 0.9999 and r[0] == self.sim.drone.id for r in rays):
                self.target_gate += 1
            if self.target_gate == self.sim.n_gates:
                self.target_gate = -1

    def render(self):
        self.sim.render()

    def close(self):
        self.sim.close()
