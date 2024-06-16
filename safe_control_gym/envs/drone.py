import importlib.util
import logging
from typing import Literal
from types import ModuleType

import numpy as np
from scipy.spatial.transform import Rotation as R
import numpy.typing as npt
from safe_control_gym.envs.constants import DroneConstants as Constants
from safe_control_gym.envs.constants import RAD_TO_DEG

logger = logging.getLogger(__name__)


class Drone:
    def __init__(self, controller: Literal["pid", "mellinger"], ctrl_freq: int = 30):
        self.firmware = self._load_firmware()
        # Initialize firmware states
        self._state = self.firmware.state_t()
        self._control = self.firmware.control_t()
        self._setpoint = self.firmware.setpoint_t()
        self._sensor_data = self.firmware.sensorData_t()
        self._acc_lpf = [self.firmware.lpf2pData() for _ in range(3)]
        self._gyro_lpf = [self.firmware.lpf2pData() for _ in range(3)]

        self.ctrl_freq = ctrl_freq
        assert controller in ["pid", "mellinger"], f"Invalid controller {controller}."
        self._controller = controller
        if controller == "pid":
            self._controller = self.firmware.controllerPid
        else:
            self._controller = self.firmware.controllerMellinger
        # Helper variables for the controller
        self._pwms = np.zeros(4)  # PWM signals for each motor
        self._tick = 0  # Current controller tick
        self._n_tumble = 0  # Number of consecutive steps the drone is tumbling
        self._last_att_ctrl_call = 0  # Last time attitude controller was called
        self._last_pos_ctrl_call = 0  # Last time position controller was called
        self._last_vel = np.zeros(3)
        self._last_rpy = np.zeros(3)
        self._fullstate_cmd = True  # Disables high level commander if True

    def reset(
        self,
        pos: npt.NDArray[np.float64] = np.array([0.0, 0.0, 0.0]),
        rpy: npt.NDArray[np.float64] = np.array([0.0, 0.0, 0.0]),
        vel: npt.NDArray[np.float64] = np.array([0.0, 0.0, 0.0]),
    ):
        """Reset the drone state and controllers.

        Args:
            pos: Initial position.
            rpy: Initial roll, pitch, yaw.
            vel: Initial velocity.
        """
        self._reset_firmware_states()
        self._reset_low_pass_filters()
        self._reset_helper_variables()
        self._reset_controller()
        # Initilaize high level commander
        self.firmware.crtpCommanderHighLevelInit()
        self._update_state(pos, rpy, vel)
        self.firmware.crtpCommanderHighLevelTellState(self._state)

    def step(
        self,
        pos: npt.NDArray[np.float64],
        rpy: npt.NDArray[np.float64],
        vel: npt.NDArray[np.float64],
        sim_time: float,
    ):
        """Take a drone controller step.

        Args:
            sim_time: Time in s from start of flight.
        """
        self.firmware.crtpCommanderHighLevelStop()  # Resets planner object
        self.firmware.crtpCommanderHighLevelUpdateTime(sim_time)
        command, args = self.command_queue.pop(0)
        getattr(self, command)(*args)

        body_rot = R.from_euler("XYZ", rpy).inv()
        # Estimate rates
        rotation_rates = (rpy - self.prev_rpy) * Constants.firmware_freq  # body coord, rad/s
        self.prev_rpy = rpy
        # TODO: Convert to real acc, not multiple of g
        acc = (vel - self.prev_vel) * Constants.firmware_freq / 9.8 + np.array([0, 0, 1])
        self.prev_vel = vel
        # Update state
        timestamp = int(self._tick / Constants.firmware_freq * 1e3)
        self._update_state(timestamp, pos, vel, acc, rpy * RAD_TO_DEG)
        # Update sensor data
        sensor_timestamp = int(self._tick / Constants.firmware_freq * 1e6)
        self._update_sensorData(sensor_timestamp, body_rot.apply(acc), rotation_rates * RAD_TO_DEG)
        # Update setpoint
        self._updateSetpoint(self._tick / Constants.firmware_freq)
        # Step controller
        self._step_controller()
        # Get action. TODO: Is this really needed?
        # new_action = (
        #     self.KF
        #     * (
        #         self.PWM2RPM_SCALE * np.clip(np.array(self.pwms), self.MIN_PWM, self.MAX_PWM)
        #         + self.PWM2RPM_CONST
        #     )
        #     ** 2
        # )
        # action = new_action[[3, 2, 1, 0]]

    # region Commands

    def full_state_cmd(
        self,
        pos: npt.NDArray[np.float64],
        vel: npt.NDArray[np.float64],
        acc: npt.NDArray[np.float64],
        yaw: float,
        rpy_rate: npt.NDArray[np.float64],
        timestep: float,
    ):
        """Send a full state command to the controller.

        Notes:
            Overrides any high level commands being processed.

        Args:
            pos: [x, y, z] position of the CF (m)
            vel: [x, y, z] velocity of the CF (m/s)
            acc: [x, y, z] acceleration of the CF (m/s^2)
            yaw: yaw of the CF (rad)
            rpy_rate: roll, pitch, yaw rates (rad/s)
            timestep: simulation time when command is sent (s)
        """
        for name, x in zip(("pos", "vel", "acc", "rpy_rate"), (pos, vel, acc, rpy_rate)):
            assert isinstance(x, np.ndarray), f"{name} must be a numpy array."
            assert len(x) == 3, f"{name} must have length 3."
        self.setpoint.position.x, self.setpoint.position.y, self.setpoint.position.z = pos
        self.setpoint.velocity.x, self.setpoint.velocity.y, self.setpoint.velocity.z = vel
        s_acc = self.setpoint.acceleration
        s_acc.x, s_acc.y, s_acc.z = acc
        s_a_rate = self.setpoint.attitudeRate
        s_a_rate.roll, s_a_rate.pitch, s_a_rate.yaw = rpy_rate * RAD_TO_DEG
        s_quat = self.setpoint.attitudeQuaternion
        s_quat.x, s_quat.y, s_quat.z, s_quat.w = R.from_euler("XYZ", [0, 0, yaw]).as_quat()
        # initilize setpoint modes to match cmdFullState
        mode = self.setpoint.mode
        mode_abs, mode_disable = self.firmware.modeAbs, self.firmware.modeDisable
        mode.x, mode.y, mode.z = mode_abs, mode_abs, mode_abs
        mode.quat = mode_abs
        mode.roll, mode.pitch, mode.yaw = mode_disable, mode_disable, mode_disable
        # This may end up skipping control loops
        self.setpoint.timestamp = int(timestep * 1000)
        self._fullstate_cmd = True

    def takeoff_cmd(self, height: float, duration: float, yaw: float | None = None):
        """Send a takeoff command to the controller.

        Args:
            height: Target takeoff height (m)
            duration: Length of manuever (s)
            yaw: Target yaw (rad). If None, yaw is not set.
        """
        self._fullstate_cmd = False
        if yaw is None:
            return self.firmware.crtpCommanderHighLevelTakeoff(height, duration)
        self.firmware.crtpCommanderHighLevelTakeoffYaw(height, duration, yaw)

    def takeoff_vel_cmd(self, height: float, vel: float, relative: bool):
        """Send a takeoff vel command to the controller.

        Args:
            height: Target takeoff height (m)
            vel: Target takeoff velocity (m/s)
            relative: Flag if takeoff height is relative to CF's current position
        """
        self.firmware.crtpCommanderHighLevelTakeoffWithVelocity(height, vel, relative)
        self._fullstate_cmd = False

    def land_cmd(self, height: float, duration: float, yaw: float | None = None):
        """Send a land command to the controller.

        Args:
            height: Target landing height (m)
            duration:: Length of manuever (s)
            yaw: Target yaw (rad). If None, yaw is not set.
        """
        self._fullstate_cmd = False
        if yaw is None:
            return self.firmware.crtpCommanderHighLevelLand(height, duration)
        self.firmware.crtpCommanderHighLevelLandYaw(height, duration, yaw)

    def land_vel_cmd(self, height: float, vel: float, relative: bool):
        """Send a land vel command to the controller.

        Args:
            height: Target landing height (m)
            vel: Target landing velocity (m/s)
            relative: Flag if landing height is relative to CF's current position
        """
        self.firmware.crtpCommanderHighLevelLandWithVelocity(height, vel, relative)
        self._fullstate_cmd = False

    def stop_cmd(self):
        """Send a stop command to the controller."""
        self.firmware.crtpCommanderHighLevelStop()
        self._fullstate_cmd = False

    def go_to_cmd(self, pos: npt.NDArray[np.float64], yaw: float, duration: float, relative: bool):
        """Send a go to command to the controller.

        Args:
            pos: [x, y, z] target position (m)
            yaw: Target yaw (rad)
            duration: Length of manuever
            relative: Flag if setpoint is relative to CF's current position
        """
        self.firmware.crtpCommanderHighLevelGoTo(*pos, yaw, duration, relative)
        self._fullstate_cmd = False

    def notify_setpoint_stop(self):
        """Send a notify setpoint stop cmd to the controller."""
        self.firmware.crtpCommanderHighLevelTellState(self.state)
        self._fullstate_cmd = False

    # endregion
    # region reset

    def _reset_firmware_states(self):
        self._state = self.firmware.state_t()
        self._control = self.firmware.control_t()
        self._setpoint = self.firmware.setpoint_t()
        self._sensor_data = self.firmware.sensorData_t()
        self._tick = 0
        self._pwms[...] = 0

    def _reset_low_pass_filters(self):
        freq = Constants.firmware_freq
        self._acc_lpf = [self.firmware.lpf2pData() for _ in range(3)]
        self._gyro_lpf = [self.firmware.lpf2pData() for _ in range(3)]
        for i in range(3):
            self.firmware.lpf2pinit(self._acc_lpf[i], freq, Constants.acc_lpf_cutoff)
            self.firmware.lpf2pinit(self._gyro_lpf[i], freq, Constants.gyro_lpf_cutoff)

    def _reset_helper_variables(self):
        self._n_tumble = 0
        self._last_att_ctrl_call = 0
        self._last_pos_ctrl_call = 0
        self._last_vel = np.zeros(3)
        self._last_rpy = np.zeros(3)

    def _reset_controller(self):
        if self._controller == "pid":
            self.firmware.controllerPidInit()
        else:
            self.firmware.controllerMellingerInit()

    # endregion
    # region Drone step

    def _step_controller(self):
        """Step the controller."""
        # Check if the drone is tumblig. If yes, set the control signal to zero.
        self._n_tumble = 0 if self._state.acc.z > Constants.tumble_threshold else self._n_tumble + 1
        if self._n_tumble > Constants.tumble_duration:
            logger.debug("CrazyFlie is tumbling. Killing motors to simulate damage prevention.")
            self._pwms[...] = 0
            self._tick += 1
            return  # Skip controller step
        # Determine tick based on time passed, allowing us to run pid slower than the 1000Hz it was
        # designed for
        tick = self._determine_controller_tick()
        if self._controller == "pid":
            ctrl = self.firmware.controllerPid
        else:
            ctrl = self.firmware.controllerMellinger
        ctrl(self._control, self._setpoint, self._sensor_data, self._state, tick)
        self._update_pwms(self._control)
        self._tick += 1
        return

    def _determine_controller_tick(self) -> Literal[0, 1, 2]:
        """Determine which controller to run based on time passed.

        This allows us to run the PID controller slower than the 1000Hz it was designed for.

        Returns:
            0: Run position and attitude controller.
            1: Run neither controller.
            2: Run only attitude controller.
        """
        time = self._tick / Constants.firmware_freq
        if time - self._last_att_ctrl_call > 0.002 and time - self._last_pos_ctrl_call > 0.01:
            self._last_att_ctrl_call = time
            self._last_pos_ctrl_call = time
            return 0
        if time - self._last_att_ctrl_call > 0.002:
            self._last_att_ctrl_call = time
            return 2
        return 1

    def _update_pwms(self, control):
        """Update the motor PWMs based on the control input.

        Args:
            control: Control signal.
        """
        # Quad formation is X
        r = control.roll / 2
        p = control.pitch / 2
        thrust = [
            control.thrust - r + p + control.yaw,
            control.thrust - r - p - control.yaw,
            control.thrust + r - p + control.yaw,
            control.thrust + r + p - control.yaw,
        ]
        thrust = np.clip(thrust, 0, Constants.max_pwm)  # Limit thrust to motor range
        self._pwms = self._thrust_to_pwm(thrust)

    @staticmethod
    def _thrust_to_pwm(thrust: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convert thrust to PWM signal.

        Assumes brushed motors.

        Args:
            thrust: Thrust in Newtons.

        Returns:
            PWM signal.
        """
        thrust = thrust / Constants.max_pwm * 60
        volts = Constants.thrust_curve_a * thrust**2 + Constants.thrust_curve_b * thrust
        percentage = np.minimum(1, volts / Constants.supply_voltage)
        return percentage * Constants.max_pwm

    # endregion
    # region Utils

    def _load_firmware(self) -> ModuleType:
        """Load the firmware module.

        pycffirmware imitates the firmware of the Crazyflie 2.1. Since the module is stateful, we
        load a new instance of the module for each drone object. This allows us to simulate multiple
        drones with different states without interference.
        """
        spec = importlib.util.find_spec("pycffirmware")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # endregion
