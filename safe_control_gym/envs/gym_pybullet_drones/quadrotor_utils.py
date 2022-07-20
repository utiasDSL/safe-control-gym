"""Helper functions for the quadrotor environment.

"""
import math
from enum import IntEnum
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation


class QuadType(IntEnum):
    """Quadrotor types numeration class.

    """

    ONE_D = 1  # One-dimensional (along z) movement.
    TWO_D = 2  # Two-dimensional (in the x-z plane) movement.
    THREE_D = 3  # Three-dimensional movement.


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
        raise ValueError("Input action shape not supported.")
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


class PIDController():
    """PID control class for Crazyflies.

    """

    def __init__(self,
                 g: float = 9.8,
                 m: float = 0.027,
                 kf: float = 3.16e-10,
                 km: float = 7.94e-12,
                 pwm2rpm_scale: float = 0.2685,
                 pwm2rpm_const: float = 4070.3,
                 min_pwm: float = 20000,
                 max_pwm: float = 65535,
                 p_coeff_for=np.array([.4, .4, 1.25]),
                 i_coeff_for=np.array([.05, .05, .05]),
                 d_coeff_for=np.array([.2, .2, .5]),
                 p_coeff_tor=np.array([70000., 70000., 60000.]),
                 i_coeff_tor=np.array([.0, .0, 500.]),
                 d_coeff_tor=np.array([20000., 20000., 12000.]),
                 ):
        """Common control classes __init__ method.

        Args:
            g (float, optional): The gravitational acceleration in m/s^2.
            m (float, optional): Mass of the quadrotor in kg.
            kf (float, optional): thrust coefficient.
            km (float, optional): torque coefficient.
            pwm2rpm_scale (float, optional): PWM-to-RPM scale factor.
            pwm2rpm_const (float, optional): PWM-to-RPM constant factor.
            min_pwm (float, optional): minimum PWM.
            max_pwm (float, optional): maximum PWM.
            p_coeff_for (ndarray, optional): position proportional coefficients.
            i_coeff_for (ndarray, optional): position integral coefficients.
            d_coeff_for (ndarray, optional): position derivative coefficients.
            p_coeff_tor (ndarray, optional): attitude proportional coefficients.
            i_coeff_tor (ndarray, optional): attitude integral coefficients.
            d_coeff_tor (ndarray, optional): attitude derivative coefficients.

        """
        self.GRAVITY = g * m # The gravitational force (M*g) acting on each drone.
        self.KF = kf
        self.KM = km
        self.P_COEFF_FOR = p_coeff_for
        self.I_COEFF_FOR = i_coeff_for
        self.D_COEFF_FOR = d_coeff_for
        self.P_COEFF_TOR = p_coeff_tor
        self.I_COEFF_TOR = i_coeff_tor
        self.D_COEFF_TOR = d_coeff_tor
        self.PWM2RPM_SCALE = pwm2rpm_scale
        self.PWM2RPM_CONST = pwm2rpm_const
        self.MIN_PWM = min_pwm
        self.MAX_PWM = max_pwm
        self.MIXER_MATRIX = np.array([[.5, -.5, -1], [.5, .5, 1], [-.5, .5, -1], [-.5, -.5, 1]])
        self.reset()

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        self.control_counter = 0  # Store the last roll, pitch, and yaw.
        self.last_rpy = np.zeros(3)  # Initialized PID control variables.
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    def compute_control(self,
                        control_timestep,
                        cur_pos,
                        cur_quat,
                        cur_vel,
                        cur_ang_vel,
                        target_pos,
                        target_rpy=np.zeros(3),
                        target_vel=np.zeros(3),
                        target_rpy_rates=np.zeros(3)
                        ):
        """Computes the PID control action (as RPMs) for a single drone.

        This methods sequentially calls `_compute_force_and_euler()` and `_compute_rpms()`.
        Parameter `cur_ang_vel` is unused.

        Args:
            control_timestep (float): The time step at which control is computed.
            cur_pos (ndarray): (3,1)-shaped array of floats containing the current position.
            cur_quat (ndarray): (4,1)-shaped array of floats containing the current orientation as a quaternion.
            cur_vel (ndarray): (3,1)-shaped array of floats containing the current velocity.
            cur_ang_vel (ndarray): (3,1)-shaped array of floats containing the current angular velocity.
            target_pos (ndarray): (3,1)-shaped array of floats containing the desired position.
            target_rpy (ndarray, optional): (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
            target_vel (ndarray, optional): (3,1)-shaped array of floats containing the desired velocity.
            target_rpy_rates (ndarray, optional): (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns:
            ndarray: (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
            ndarray: (3,1)-shaped array of floats containing the current XYZ position error.
            float: The current yaw error.

        """
        self.control_counter += 1
        thrust, computed_target_rpy, pos_e = self._compute_force_and_euler(control_timestep,
                                                                           cur_pos,
                                                                           cur_quat,
                                                                           cur_vel,
                                                                           target_pos,
                                                                           target_rpy,
                                                                           target_vel
                                                                           )
        rpm = self._compute_rpms(control_timestep,
                                 thrust,
                                 cur_quat,
                                 computed_target_rpy,
                                 target_rpy_rates
                                 )
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]

    def _compute_force_and_euler(self,
                                 control_timestep,
                                 cur_pos,
                                 cur_quat,
                                 cur_vel,
                                 target_pos,
                                 target_rpy,
                                 target_vel
                                 ):
        """DSL's CF2.x PID position control.

        Args:
            control_timestep (float): The time step at which control is computed.
            cur_pos (ndarray): (3,1)-shaped array of floats containing the current position.
            cur_quat (ndarray): (4,1)-shaped array of floats containing the current orientation as a quaternion.
            cur_vel (ndarray): (3,1)-shaped array of floats containing the current velocity.
            target_pos (ndarray): (3,1)-shaped array of floats containing the desired position.
            target_rpy (ndarray): (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
            target_vel (ndarray): (3,1)-shaped array of floats containing the desired velocity.

        Returns:
            float: The target thrust along the drone z-axis.
            ndarray: (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
            float: The current position error.

        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        self.integral_pos_e = self.integral_pos_e + pos_e*control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)
        # PID target thrust.
        target_thrust = np.multiply(self.P_COEFF_FOR, pos_e) \
                        + np.multiply(self.I_COEFF_FOR, self.integral_pos_e) \
                        + np.multiply(self.D_COEFF_FOR, vel_e) + np.array([0, 0, self.GRAVITY])
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:, 2]))
        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        # Target rotation.
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        if np.any(np.abs(target_euler) > math.pi):
            raise ValueError("\n[ERROR] ctrl it", self.control_counter, "in Control._compute_force_and_euler(), values outside range [-pi,pi]")
        return thrust, target_euler, pos_e

    def _compute_rpms(self,
                      control_timestep,
                      thrust,
                      cur_quat,
                      target_euler,
                      target_rpy_rates
                      ):
        """DSL's CF2.x PID attitude control.

        Args:
            control_timestep (float): The time step at which control is computed.
            thrust (float): The target thrust along the drone z-axis.
            cur_quat (ndarray): (4,1)-shaped array of floats containing the current orientation as a quaternion.
            target_euler (ndarray): (3,1)-shaped array of floats containing the computed target Euler angles.
            target_rpy_rates (ndarray): (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns:
            ndarray: (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w, x, y, z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()), cur_rotation) - np.dot(cur_rotation.transpose(), target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy)/control_timestep
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e*control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)
        # PID target torques.
        target_torques = - np.multiply(self.P_COEFF_TOR, rot_e) \
                         + np.multiply(self.D_COEFF_TOR, rpy_rates_e) \
                         + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
