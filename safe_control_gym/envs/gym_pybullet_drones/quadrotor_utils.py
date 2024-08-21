"""Helper functions for the quadrotor environment."""

from abc import ABC
from enum import IntEnum

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation


class QuadType(IntEnum):
    """Quadrotor types numeration class."""

    ONE_D = 1  # One-dimensional (along z) movement.
    TWO_D = 2  # Two-dimensional (in the x-z plane) movement.
    THREE_D = 3  # Three-dimensional movement.
    TWO_D_ATTITUDE = 4  # Two-dimensional (in the x-z plane) movement with attitude control.
    TWO_D_ATTITUDE_5S = 5  # Two-dimensional (in the x-z plane) movement with attitude control with 5 states.


def cmd2pwm(thrust, pwm2rpm_scale, pwm2rpm_const, ct, pwm_min, pwm_max):
    """Generic cmd to pwm function.

    For 1D, thrust is the total of all 4 motors; for 2D, 1st thrust is total of motor
    1 & 4, 2nd thrust is total of motor 2 & 3; for 4D, thrust is thrust of each motor.

    Args:
        thrust (ndarray): array of length 1, 2 containing target thrusts.
        pwm2rpm_scale (float): scaling factor between PWM and RPMs.
        pwm2rpm_const (float): constant factor between PWM and RPMs.
        ct (float): torque coefficient.
        pwm_min (float): pwm lower bound.
        pwm_max (float): pwm upper bound.

    Returns:
        ndarray: array of length 4 containing PWM.
    """
    n_motor = 4 // int(thrust.size)
    thrust = np.clip(thrust, np.zeros_like(thrust), None)  # Make sure thrust is not negative.
    motor_pwm = (np.sqrt(thrust / n_motor / ct) - pwm2rpm_const) / pwm2rpm_scale
    if thrust.size == 1:  # 1D case.
        motor_pwm = np.repeat(motor_pwm, 4)
    elif thrust.size == 2:  # 2D case.
        motor_pwm = np.concatenate([motor_pwm, motor_pwm[::-1]], 0)
    elif thrust.size == 4:  # 3D case.
        motor_pwm = np.array(motor_pwm)
    else:
        raise ValueError('Input action shape not supported.')
    motor_pwm = np.clip(motor_pwm, pwm_min, pwm_max)
    return motor_pwm


def pwm2rpm(pwm, pwm2rpm_scale, pwm2rpm_const):
    """Computes motor squared rpm from pwm.

    Args:
        pwm (ndarray): Array of length 4 containing PWM.
        pwm2rpm_scale (float): Scaling factor between PWM and RPMs.
        pwm2rpm_const (float): Constant factor between PWM and RPMs.

    Returns:
        ndarray: Array of length 4 containing RPMs.
    """
    rpm = pwm2rpm_scale * pwm + pwm2rpm_const
    return rpm


class AttitudeControl(ABC):
    """AttitudeControl Class."""

    def __init__(self,
                 control_timestep,
                 sim_timestep,
                 g: float = 9.8,
                 kf: float = 3.16e-10,
                 km: float = 7.94e-12,
                 p_coeff_tor=np.array([70000., 70000., 60000.]),
                 i_coeff_tor=np.array([.0, .0, 500.]),
                 d_coeff_tor=np.array([20000., 20000., 12000.]),
                 pwm2rpm_scale: float = 0.2685,
                 pwm2rpm_const: float = 4070.3,
                 min_pwm: float = 20000,
                 max_pwm: float = 65535,
                 ):
        """AttitudeControl class __init__ method.

        Args:
            control_timestep (float): The time step at which control is computed.
            sim_timestep (float): The time step at which simulation is carried out.
            g (float, optional): The gravitational acceleration in m/s^2.
            kf (float, optional): Thrust coefficient.
            km (float, optional): Torque coefficient.
            p_coeff_tor (ndarray, optional): Attitude proportional coefficients.
            i_coeff_tor (ndarray, optional): Attitude integral coefficients.
            d_coeff_tor (ndarray, optional): Attitude derivative coefficients.
            pwm2rpm_scale (float, optional): PWM-to-RPM scale factor.
            pwm2rpm_const (float, optional): PWM-to-RPM constant factor.
            min_pwm (float, optional): Minimum PWM.
            max_pwm (float, optional): Maximum PWM.
        """

        self.g = g
        self.KF = kf
        self.KM = km
        self.P_COEFF_TOR = np.array(p_coeff_tor)
        self.I_COEFF_TOR = np.array(i_coeff_tor)
        self.D_COEFF_TOR = np.array(d_coeff_tor)
        self.PWM2RPM_SCALE = np.array(pwm2rpm_scale)
        self.PWM2RPM_CONST = np.array(pwm2rpm_const)
        self.MIN_PWM = np.array(min_pwm)
        self.MAX_PWM = np.array(max_pwm)
        self.MIXER_MATRIX = np.array([[.5, -.5, -1], [.5, .5, 1], [-.5, .5, -1], [-.5, -.5, 1]])

        self.a_coeff = -1.1264
        self.b_coeff = 2.2541
        self.c_coeff = 0.0209

        self.last_rpy = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

        self.control_timestep = control_timestep
        self.sim_timestep = sim_timestep

    def reset(self):
        """Reinitialize just the controller before a new run."""

        # Clear PID control variables.
        self.last_rpy = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    def _dslPIDAttitudeControl(self,
                               thrust,
                               cur_quat,
                               target_euler,
                               target_rpy_rates=np.zeros(3)
                               ):
        """DSL's CF2.x PID attitude control.

        Parameters
        ----------
        thrust : ndarray
            (4,1)-shaped array of target thrust (Newton) along the drone z-axis.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        target_euler : ndarray
            (3,1)-shaped array of floats containing the computed target Euler angles.
        target_rpy_rates : ndarray
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """
        # control_timestep = self.control_timestep
        sim_timestep = self.sim_timestep
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w, x, y, z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()), cur_rotation) - np.dot(cur_rotation.transpose(), target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy) / sim_timestep
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e * sim_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)
        #### PID target torques ####################################
        target_torques = - np.multiply(self.P_COEFF_TOR, rot_e) \
            + np.multiply(self.D_COEFF_TOR, rpy_rates_e) \
            + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
        target_torques = np.clip(target_torques, -3200, 3200)
        # pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        # pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        # return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
        return thrust + self.pwm2thrust(np.dot(self.MIXER_MATRIX, target_torques))

    def pwm2thrust(self, pwm):
        """Convert pwm to thrust using a quadratic function."""

        pwm_scaled = pwm / self.MAX_PWM
        # pwm_scaled = pwm
        # solve quadratic equation using abc formula
        thrust = (-self.b_coeff + np.sqrt(self.b_coeff**2 - 4 * self.a_coeff * (self.c_coeff - pwm_scaled))) / (2 * self.a_coeff)
        return thrust

    def thrust2pwm(self, thrust):
        """Convert thrust to pwm using a quadratic function."""

        pwm = self.a_coeff * thrust * thrust + self.b_coeff * thrust + self.c_coeff
        pwm = np.maximum(pwm, 0.0)
        pwm = np.minimum(pwm, 1.0)
        thrust_pwm = pwm * self.MAX_PWM
        return thrust_pwm
