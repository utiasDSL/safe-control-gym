from __future__ import annotations

import numpy as np
import logging
import math
from typing import Callable

import gymnasium
from safe_control_gym.envs.quadrotor import Quadrotor
from safe_control_gym.envs.drone import Drone

import importlib.util

spec = importlib.util.find_spec("pycffirmware")
firm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(firm)
# import pycffirmware as firm

logger = logging.getLogger(__name__)


class FirmwareWrapper(gymnasium.Env):
    CONTROLLER = "mellinger"  # specifies controller type

    def __init__(self, env_func: Callable[[], Quadrotor], firmware_freq: int, ctrl_freq: int):
        """Initializes a FirmwareWrapper object.

        This allows users to simulate the on board controllers of CrazyFlie (CF), including access
        to a portion of the CF command API. FirmwareWrapper.reset() must be called at the beginning
        of every episode.

        Args:
            env_func (function): initilizer for safe-control-gym environment
            firmware_freq (int): frequency to run the firmware loop at (typically 500)
            ctrl_freq (int): frequency that .step will be called (typically < 60)

        Attributes:
            env: safe-control environment

        Todo:
            * Add support for state estimation
        """
        super().__init__()
        self.action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(4,))
        self.observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(10,))
        self.drone = Drone(self.CONTROLLER)
        self.firmware_freq = firmware_freq
        self.ctrl_freq = ctrl_freq
        self.step_freq = ctrl_freq
        self.env = env_func()

    def reset(self):
        obs, info = self.env.reset()
        pos, vel, rpy = obs["pos"], obs["vel"], obs["rpy"]
        self.drone.reset(pos, rpy, vel)
        obs = np.concatenate(
            [np.array([pos[0], vel[0], pos[1], vel[1], pos[2], vel[2]]), rpy, obs["ang_vel"]]
        )
        if self.env.n_drones > 1:
            raise NotImplementedError("Firmware wrapper does not support multiple drones.")
        return obs, info

    def close(self):
        self.env.close()

    def step(self, sim_time: float, action: np.ndarray):
        """Step the firmware_wrapper class and its environment.

        This function should be called once at the rate of ctrl_freq. Step processes and high level
        commands, and runs the firmware loop and simulator according to the frequencies set.

        Args:
            sim_time (float): Time in seconds since beginning the episode
            action (np.ndarray(4)): Motor control forces to be applied. Order is: front left, back
                left, back right, front right.

        Todo:
            * Add support for state estimation
        """
        while self.drone.tick / self.drone.firmware_freq < sim_time + 1 / self.step_freq:
            obs, reward, done, info = self.env.step(action)
            pos, vel, rpy = obs["pos"], obs["vel"], obs["rpy"]
            obs = np.concatenate(
                [np.array([pos[0], vel[0], pos[1], vel[1], pos[2], vel[2]]), rpy, obs["ang_vel"]]
            )
            action = self.drone.step_controller(pos, rpy, vel)[::-1]
        return obs, reward, done, info, action

    # region Controller functions

    def sendFullStateCmd(self, pos, vel, acc, yaw, rpy_rate, timestep):
        self.drone.full_state_cmd(pos, vel, acc, yaw, rpy_rate, timestep)

    def sendTakeoffCmd(self, height, duration):
        self.drone.takeoff_cmd(height, duration)

    def sendTakeoffYawCmd(self, height, duration, yaw):
        self.drone.takeoff_cmd(height, duration, yaw)

    def sendTakeoffVelCmd(self, height, vel, relative):
        self.drone.takeoff_vel_cmd(height, vel, relative)

    def sendLandCmd(self, height, duration):
        self.drone.land_cmd(height, duration)

    def sendLandYawCmd(self, height, duration, yaw):
        self.drone.land_cmd(height, duration, yaw)

    def sendLandVelCmd(self, height, vel, relative):
        self.drone.land_vel_cmd(height, vel, relative)

    def sendStopCmd(self):
        self.drone.stop_cmd()

    def sendGotoCmd(self, pos, yaw, duration_s, relative):
        self.drone.go_to_cmd(pos, yaw, duration_s, relative)

    def notifySetpointStop(self):
        self.drone.notify_setpoint_stop()

    # endregion
