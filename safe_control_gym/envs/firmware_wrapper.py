from __future__ import annotations

import numpy as np
import logging
import math
from typing import Callable

from scipy.spatial.transform import Rotation as R

from safe_control_gym.envs.quadrotor import Quadrotor
from safe_control_gym.envs.drone import Drone
from safe_control_gym.envs.constants import DroneConstants

import importlib.util

spec = importlib.util.find_spec("pycffirmware")
firm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(firm)
# import pycffirmware as firm

logger = logging.getLogger(__name__)


class FirmwareWrapper:
    CONTROLLER = "mellinger"  # specifies controller type

    # Configurations to match firmware. Not recommended to change
    GYRO_LPF_CUTOFF_FREQ = 80
    ACCEL_LPF_CUTOFF_FREQ = 30

    # Motor and power supply settings
    BRUSHED = True

    RAD_TO_DEG = 180 / math.pi

    def __init__(
        self,
        env_func: Callable[[], Quadrotor],
        firmware_freq: int,
        ctrl_freq: int,
        PWM2RPM_SCALE: float = 0.2685,
        PWM2RPM_CONST: float = 4070.3,
        KF: float = 3.16e-10,
        MIN_PWM: int = 20000,
        MAX_PWM: int = 65535,
        verbose: bool = False,
    ):
        """Initializes a FirmwareWrapper object.

        This allows users to simulate the on board controllers of CrazyFlie (CF), including access
        to a portion of the CF command API. FirmwareWrapper.reset() must be called at the beginning
        of every episode.

        Args:
            env_func (function): initilizer for safe-control-gym environment
            firmware_freq (int): frequency to run the firmware loop at (typically 500)
            ctrl_freq (int): frequency that .step will be called (typically < 60)
            PWM2RPM_SCALE (float): mapping factor from PWM to RPM
            PWM2RPM_CONST (float): mapping constant from PWM to RPM
            KF = (float): motor force factor
            MIN_PWM (int): minimum PWM command
            MAX_PWM (int): maximum pwm command
            verbose (bool): displays additional information
            **kwargs: to be passed to BaseController

        Attributes:
            env: safe-control environment

        Todo:
            * Add support for state estimation
        """
        self.drone = Drone(self.CONTROLLER)
        self.firmware_freq = firmware_freq
        self.ctrl_freq = ctrl_freq

        self.PWM2RPM_SCALE = float(PWM2RPM_SCALE)
        self.PWM2RPM_CONST = float(PWM2RPM_CONST)
        self.KF = float(KF)
        self.MIN_PWM = float(MIN_PWM)
        self.MAX_PWM = float(MAX_PWM)
        self.verbose = verbose

        self.env = env_func()

    # region Controller functions
    def reset(self):
        """Resets the firmware_wrapper object.

        Todo:
            * Add support for state estimation
        """
        self.takeoff_sent = False

        # Initialize gyro lpf
        self.acclpf = [firm.lpf2pData() for _ in range(3)]
        self.gyrolpf = [firm.lpf2pData() for _ in range(3)]
        for i in range(3):
            firm.lpf2pInit(self.acclpf[i], self.firmware_freq, self.GYRO_LPF_CUTOFF_FREQ)
            firm.lpf2pInit(self.gyrolpf[i], self.firmware_freq, self.ACCEL_LPF_CUTOFF_FREQ)

        # Initialize state objects
        self.control = firm.control_t()
        self.setpoint = firm.setpoint_t()
        self.sensorData = firm.sensorData_t()
        self.state = firm.state_t()
        self.tick = 0
        self.pwms = [0, 0, 0, 0]
        self.action = [0, 0, 0, 0]
        self.command_queue = []

        self.tumble_counter = 0
        self.prev_vel = np.array([0, 0, 0])
        self.prev_rpy = np.array([0, 0, 0])
        self.prev_time_s = None
        self.last_pos_pid_call = 0
        self.last_att_pid_call = 0

        # Initialize state flags
        self._error = False
        self.sensorData_set = False
        self.state_set = False
        self.full_state_cmd_override = True  # When true, high level commander is not called

        # Initialize controller
        if self.CONTROLLER == "pid":
            firm.controllerPidInit()
            logger.debug("PID controller init test:", firm.controllerPidTest())
        elif self.CONTROLLER == "mellinger":
            self.drone.firmware.controllerMellingerInit()
            firm.controllerMellingerInit()
            assert (
                self.firmware_freq == 500
            ), "Mellinger controller requires a firmware frequency of 500Hz."
            logger.debug("Mellinger controller init test:", firm.controllerMellingerTest())

        # Reset environment
        init_obs, init_info = self.env.reset()
        init_pos = np.array([init_obs[0], init_obs[2], init_obs[4]])  # global coord, m
        init_vel = np.array([init_obs[1], init_obs[3], init_obs[5]])  # global coord, m/s
        init_rpy = np.array([init_obs[6], init_obs[7], init_obs[8]])  # body coord, rad
        if self.env.NUM_DRONES > 1:
            raise NotImplementedError(
                "Firmware controller wrapper does not support multiple drones."
            )

        # Initilaize high level commander
        firm.crtpCommanderHighLevelInit()
        self.drone.firmware.crtpCommanderHighLevelInit()
        self._update_state(
            0, init_pos, init_vel, np.array([0.0, 0.0, 1.0]), init_rpy * self.RAD_TO_DEG
        )
        self._update_initial_state(init_obs)
        firm.crtpCommanderHighLevelTellState(self.state)
        self.drone.firmware.crtpCommanderHighLevelTellState(self.state)

        self.ctrl_dt = 1 / self.ctrl_freq
        self.firmware_dt = 1 / self.firmware_freq

        # Initialize visualization tools
        self.first_motor_killed_print = True
        self.pyb_client = init_info["pyb_client"]
        self.last_visualized_setpoint = None

        self.results_dict = {
            "obs": [],
            "reward": [],
            "done": [],
            "info": [],
            "action": [],
        }

        return init_obs, init_info

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
        self._process_command_queue(sim_time)

        while self.tick / self.firmware_freq < sim_time + self.ctrl_dt:
            # Step the environment and print all returned information.
            obs, reward, done, info = self.env.step(action)

            # Get state values from pybullet
            cur_pos = np.array([obs[0], obs[2], obs[4]])  # global coord, m
            cur_vel = np.array([obs[1], obs[3], obs[5]])  # global coord, m/s
            cur_rpy = np.array([obs[6], obs[7], obs[8]])  # body coord, rad
            body_rot = R.from_euler("XYZ", cur_rpy).inv()

            # Estimate rates
            cur_rotation_rates = (cur_rpy - self.prev_rpy) / self.firmware_dt  # body coord, rad/s
            self.prev_rpy = cur_rpy
            cur_acc = (cur_vel - self.prev_vel) / self.firmware_dt / 9.8 + np.array(
                [0, 0, 1]
            )  # global coord
            self.prev_vel = cur_vel

            # Update state
            state_timestamp = int(self.tick / self.firmware_freq * 1e3)
            self._update_state(
                state_timestamp,
                cur_pos,
                cur_vel,
                cur_acc,
                cur_rpy * self.RAD_TO_DEG,
            )  # , quat=cur_quat)

            # Update sensor data
            sensor_timestamp = int(self.tick / self.firmware_freq * 1e6)
            self._update_sensorData(
                sensor_timestamp,
                body_rot.apply(cur_acc),
                cur_rotation_rates * self.RAD_TO_DEG,
            )

            # Update setpoint
            self._updateSetpoint(self.tick / self.firmware_freq)  # setpoint looks right

            # Step controller
            self._step_controller()

            # Get action
            new_action = (
                self.KF
                * (
                    self.PWM2RPM_SCALE * np.clip(np.array(self.pwms), self.MIN_PWM, self.MAX_PWM)
                    + self.PWM2RPM_CONST
                )
                ** 2
            )
            action = new_action[[3, 2, 1, 0]]

            if self._error:
                action = np.zeros(4)
                if self.first_motor_killed_print:
                    logger.warning("Drone firmware error. Motors are killed.")
                    self.first_motor_killed_print = False
                done = True

            self.action = action
        return obs, reward, done, info, action

    def _update_initial_state(self, obs):
        self.prev_vel = np.array([obs[1], obs[3], obs[5]])
        self.prev_rpy = np.array([obs[6], obs[7], obs[8]])

    # endregion

    # region Sensor update
    def _update_sensorData(self, timestamp, acc_vals, gyro_vals, baro_vals=[1013.25, 25]):
        """
        Axis3f acc;               // Gs
        Axis3f gyro;              // deg/s
        Axis3f mag;               // gauss
        baro_t baro;              // C, Pa
        #ifdef LOG_SEC_IMU
            Axis3f accSec;            // Gs
            Axis3f gyroSec;           // deg/s
        #endif
        uint64_t interruptTimestamp;   // microseconds
        """
        # Only gyro and acc are used in controller. Mag and baro used in state etimation, but are
        # not implemented in this simulation
        self._update_acc(*acc_vals)
        self._update_gyro(*gyro_vals)

        self.sensorData.interruptTimestamp = timestamp
        self.sensorData_set = True

    def _update_gyro(self, x, y, z):
        self.sensorData.gyro.x = firm.lpf2pApply(self.gyrolpf[0], x)
        self.sensorData.gyro.y = firm.lpf2pApply(self.gyrolpf[1], y)
        self.sensorData.gyro.z = firm.lpf2pApply(self.gyrolpf[2], z)

    def _update_acc(self, x, y, z):
        self.sensorData.acc.x = firm.lpf2pApply(self.acclpf[0], x)
        self.sensorData.acc.y = firm.lpf2pApply(self.acclpf[1], y)
        self.sensorData.acc.z = firm.lpf2pApply(self.acclpf[2], z)

    # endregion

    # region State update
    def _update_state(self, timestamp, pos, vel, acc, rpy, quat=None):
        """
        attitude_t attitude;      // deg (legacy CF2 body coordinate system, where pitch is inverted)
        quaternion_t attitudeQuaternion;
        point_t position;         // m
        velocity_t velocity;      // m/s
        acc_t acc;                // Gs (but acc.z without considering gravity)
        """
        self._update_attitude_t(
            self.state.attitude, timestamp, *rpy
        )  # RPY required for PID and high level commander
        if self.CONTROLLER == "mellinger":
            self._update_attitudeQuaternion(
                self.state.attitudeQuaternion, timestamp, *rpy
            )  # Quat required for Mellinger

        self._update_3D_vec(self.state.position, timestamp, *pos)
        self._update_3D_vec(self.state.velocity, timestamp, *vel)
        self._update_3D_vec(self.state.acc, timestamp, *acc)
        self.state_set = True

    def _update_3D_vec(self, point, timestamp, x, y, z):
        point.x = x
        point.y = y
        point.z = z
        point.timestamp = timestamp

    def _update_attitudeQuaternion(self, quaternion_t, timestamp, qx, qy, qz, qw=None):
        """Updates attitude quaternion.

        Note:
            if qw is present, input is taken as a quat. Else, as roll, pitch, and yaw in deg
        """
        quaternion_t.timestamp = timestamp

        if qw is None:  # passed roll, pitch, yaw
            qx, qy, qz, qw = _get_quaternion_from_euler(
                qx / self.RAD_TO_DEG, qy / self.RAD_TO_DEG, qz / self.RAD_TO_DEG
            )

        quaternion_t.x = qx
        quaternion_t.y = qy
        quaternion_t.z = qz
        quaternion_t.w = qw

    def _update_attitude_t(self, attitude_t, timestamp, roll, pitch, yaw):
        attitude_t.timestamp = timestamp
        attitude_t.roll = roll
        attitude_t.pitch = -pitch  # Legacy representation in CF firmware
        attitude_t.yaw = yaw

    # endregion

    # region Controller
    def _step_controller(self):
        self.sensorData_set = False
        self.state_set = False
        self.drone._tick = self.tick
        self.drone._pwms[:] = self.pwms
        self.drone._control = self.control
        self.drone._setpoint = self.setpoint
        self.drone._sensor_data = self.sensorData
        self.drone._state = self.state
        self.drone._step_controller()
        self.control = self.drone._control
        self.state = self.drone._state
        self.setpoint = self.drone._setpoint
        self.sensorData = self.drone._sensor_data
        self.pwms = self.drone._pwms
        self.tick = self.drone._tick
        return
        if not (self.sensorData_set):
            logger.warning("sensorData has not been updated since last controller call.")
        if not (self.state_set):
            logger.warning("state has not been updated since last controller call.")
        self.sensorData_set = False
        self.state_set = False

        # Check for tumbling crazyflie
        if self.state.acc.z < -0.5:
            self.tumble_counter += 1
        else:
            self.tumble_counter = 0
        if self.tumble_counter >= 30:
            logger.warning("CrazyFlie is Tumbling. Killing motors to save propellers.")
            self.pwms = [0, 0, 0, 0]
            self.tick += 1
            self._error = True
            return

        # Determine tick based on time passed, allowing us to run pid slower than the 1000Hz it was
        # designed for
        cur_time = self.tick / self.firmware_freq
        if (cur_time - self.last_att_pid_call > 0.002) and (
            cur_time - self.last_pos_pid_call > 0.01
        ):
            _tick = 0  # Runs position and attitude controller
            self.last_pos_pid_call = cur_time
            self.last_att_pid_call = cur_time
        elif cur_time - self.last_att_pid_call > 0.002:
            self.last_att_pid_call = cur_time
            _tick = 2  # Runs attitude controller
        else:
            _tick = 1  # Runs neither controller

        # Step the chosen controller
        if self.CONTROLLER == "pid":
            firm.controllerPid(self.control, self.setpoint, self.sensorData, self.state, _tick)
        elif self.CONTROLLER == "mellinger":
            firm.controllerMellinger(
                self.control, self.setpoint, self.sensorData, self.state, _tick
            )

        # Get pwm values from control object
        self._powerDistribution(self.control)
        self.tick += 1

    def copy_control(self):
        control = firm.control_t()
        control.thrust = self.control.thrust
        control.roll = self.control.roll
        control.pitch = self.control.pitch
        control.yaw = self.control.yaw
        return control

    def _updateSetpoint(self, timestep):
        if not self.full_state_cmd_override:
            firm.crtpCommanderHighLevelTellState(self.state)
            firm.crtpCommanderHighLevelUpdateTime(
                timestep
            )  # Sets commander time variable --- this is time in s from start of flight
            firm.crtpCommanderHighLevelGetSetpoint(self.setpoint, self.state)

    def _process_command_queue(self, sim_time):
        if len(self.command_queue) > 0:
            firm.crtpCommanderHighLevelStop()  # Resets planner object
            firm.crtpCommanderHighLevelUpdateTime(
                sim_time
            )  # Sets commander time variable --- this is time in s from start of flight
            command, args = self.command_queue.pop(0)
            getattr(self, command)(*args)

    def sendFullStateCmd(self, pos, vel, acc, yaw, rpy_rate, timestep):
        """Adds a sendfullstate command to command processing queue.

        Notes:
            Overrides any high level commands being processed.

        Args:
            pos (list): [x, y, z] position of the CF (m)
            vel (list): [x, y, z] velocity of the CF (m/s)
            acc (list): [x, y, z] acceleration of the CF (m/s^2)
            yaw (float): yaw of the CF (rad)
            rpy_rate (list): roll, pitch, yaw rates (rad/s)
            timestep (float): simulation time when command is sent (s)
        """
        self.command_queue += [["_sendFullStateCmd", [pos, vel, acc, yaw, rpy_rate, timestep]]]

    def _sendFullStateCmd(self, pos, vel, acc, yaw, rpy_rate, timestep):
        self.setpoint.position.x = pos[0]
        self.setpoint.position.y = pos[1]
        self.setpoint.position.z = pos[2]
        self.setpoint.velocity.x = vel[0]
        self.setpoint.velocity.y = vel[1]
        self.setpoint.velocity.z = vel[2]
        self.setpoint.acceleration.x = acc[0]
        self.setpoint.acceleration.y = acc[1]
        self.setpoint.acceleration.z = acc[2]

        self.setpoint.attitudeRate.roll = rpy_rate[0] * self.RAD_TO_DEG
        self.setpoint.attitudeRate.pitch = rpy_rate[1] * self.RAD_TO_DEG
        self.setpoint.attitudeRate.yaw = rpy_rate[2] * self.RAD_TO_DEG

        quat = _get_quaternion_from_euler(0, 0, yaw)
        self.setpoint.attitudeQuaternion.x = quat[0]
        self.setpoint.attitudeQuaternion.y = quat[1]
        self.setpoint.attitudeQuaternion.z = quat[2]
        self.setpoint.attitudeQuaternion.w = quat[3]

        # self.setpoint.attitude.yaw = yaw * 180 / math.pi
        # self.setpoint.attitude.pitch = 0
        # self.setpoint.attitude.roll = 0

        # initilize setpoint modes to match cmdFullState
        self.setpoint.mode.x = firm.modeAbs
        self.setpoint.mode.y = firm.modeAbs
        self.setpoint.mode.z = firm.modeAbs

        self.setpoint.mode.quat = firm.modeAbs
        self.setpoint.mode.roll = firm.modeDisable
        self.setpoint.mode.pitch = firm.modeDisable
        self.setpoint.mode.yaw = firm.modeDisable

        # TODO: This may end up skipping control loops
        self.setpoint.timestamp = int(timestep * 1000)
        self.full_state_cmd_override = True

    def sendTakeoffCmd(self, height, duration):
        """Adds a takeoff command to command processing queue.

        Args:
            height (float): target takeoff height (m)
            duration: (float): length of manuever
        """
        self.command_queue += [["_sendTakeoffCmd", [height, duration]]]

    def _sendTakeoffCmd(self, height, duration):
        logger.info(f"{self.tick}: Takeoff command sent.")
        self.takeoff_sent = True
        firm.crtpCommanderHighLevelTakeoff(height, duration)
        self.full_state_cmd_override = False

    def sendTakeoffYawCmd(self, height, duration, yaw):
        """Adds a takeoffyaw command to command processing queue.

        Args:
            height (float): target takeoff height (m)
            duration: (float): length of manuever
            yaw (float): target yaw (rad)
        """
        self.command_queue += [["_sendTakeoffYawCmd", [height, duration, yaw]]]

    def _sendTakeoffYawCmd(self, height, duration, yaw):
        logger.info(f"{self.tick}: Takeoff command sent.")
        firm.crtpCommanderHighLevelTakeoffYaw(height, duration, yaw)
        self.full_state_cmd_override = False

    def sendTakeoffVelCmd(self, height, vel, relative):
        """Adds a takeoffvel command to command processing queue.

        Args:
            height (float): target takeoff height (m)
            vel (float): target takeoff velocity (m/s)
            relative: (bool): whether takeoff height is relative to CF's current position
        """
        self.command_queue += [["_sendTakeoffVelCmd", [height, vel, relative]]]

    def _sendTakeoffVelCmd(self, height, vel, relative):
        logger.info(f"{self.tick}: Takeoff command sent.")
        firm.crtpCommanderHighLevelTakeoffWithVelocity(height, vel, relative)
        self.full_state_cmd_override = False

    def sendLandCmd(self, height, duration):
        """Adds a land command to command processing queue.

        Args:
            height (float): target landing height (m)
            duration: (float): length of manuever
        """
        self.command_queue += [["_sendLandCmd", [height, duration]]]

    def _sendLandCmd(self, height, duration):
        logger.info(f"{self.tick}: Land command sent.")
        firm.crtpCommanderHighLevelLand(height, duration)
        self.full_state_cmd_override = False

    def sendLandYawCmd(self, height, duration, yaw):
        """Adds a landyaw command to command processing queue.

        Args:
            height (float): target landing height (m)
            duration: (float): length of manuever
            yaw (float): target yaw (rad)
        """
        self.command_queue += [["_sendLandYawCmd", [height, duration, yaw]]]

    def _sendLandYawCmd(self, height, duration, yaw):
        logger.info(f"{self.tick}: Land command sent.")
        firm.crtpCommanderHighLevelLandYaw(height, duration, yaw)
        self.full_state_cmd_override = False

    def sendLandVelCmd(self, height, vel, relative):
        """Adds a landvel command to command processing queue.

        Args:
            height (float): target landing height (m)
            vel (float): target landing velocity (m/s)
            relative: (bool): whether landing height is relative to CF's current position
        """
        self.command_queue += [["_sendLandVelCmd", [height, vel, relative]]]

    def _sendLandVelCmd(self, height, vel, relative):
        logger.info(f"{self.tick}: Land command sent.")
        firm.crtpCommanderHighLevelLandWithVelocity(height, vel, relative)
        self.full_state_cmd_override = False

    def sendStopCmd(self):
        """Adds a stop command to command processing queue."""
        self.command_queue += [["_sendStopCmd", []]]

    def _sendStopCmd(self):
        logger.info(f"{self.tick}: Stop command sent.")
        firm.crtpCommanderHighLevelStop()
        self.full_state_cmd_override = False

    def sendGotoCmd(self, pos, yaw, duration_s, relative):
        """Adds a goto command to command processing queue.

        Args:
            pos (list): [x, y, z] target position (m)
            yaw (float): target yaw (rad)
            duration_s (float): length of manuever
            relative (bool): whether setpoint is relative to CF's current position
        """
        self.command_queue += [["_sendGotoCmd", [pos, yaw, duration_s, relative]]]

    def _sendGotoCmd(self, pos, yaw, duration_s, relative):
        logger.info(f"{self.tick}: Go to command sent.")
        firm.crtpCommanderHighLevelGoTo(*pos, yaw, duration_s, relative)
        self.full_state_cmd_override = False

    def notifySetpointStop(self):
        """Adds a notifySetpointStop command to command processing queue."""
        self.command_queue += [["_notifySetpointStop", []]]

    def _notifySetpointStop(self):
        """Adds a notifySetpointStop command to command processing queue."""
        logger.info(f"{self.tick}: Notify setpoint stop command sent.")
        firm.crtpCommanderHighLevelTellState(self.state)
        self.full_state_cmd_override = False

    def _powerDistribution(self, control_t):
        # Quad formation is X
        r = control_t.roll / 2
        p = control_t.pitch / 2
        thrust = [
            control_t.thrust - r + p + control_t.yaw,
            control_t.thrust - r - p - control_t.yaw,
            control_t.thrust + r - p + control_t.yaw,
            control_t.thrust + r + p - control_t.yaw,
        ]
        thrust = np.clip(thrust, 0, DroneConstants.max_pwm)  # Limit thrust to motor range
        self.pwms = self.drone._thrust_to_pwm(thrust)

    # endregion


# region Utils
def _get_quaternion_from_euler(roll, pitch, yaw):
    """Convert an Euler angle to a quaternion.

    Args:
        roll (float): The roll (rotation around x-axis) angle in radians.
        pitch (float): The pitch (rotation around y-axis) angle in radians.
        yaw (float): The yaw (rotation around z-axis) angle in radians.

    Returns:
        list: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(
        pitch / 2
    ) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(
        pitch / 2
    ) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(
        pitch / 2
    ) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(
        pitch / 2
    ) * np.sin(yaw / 2)

    return [qx, qy, qz, qw]


# endregion
