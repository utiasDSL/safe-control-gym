'''Helper functions for the quadrotor environment.'''

from enum import IntEnum

import numpy as np


class QuadType(IntEnum):
    '''Quadrotor types numeration class.'''

    ONE_D = 1  # One-dimensional (along z) movement.
    TWO_D = 2  # Two-dimensional (in the x-z plane) movement.
    THREE_D = 3  # Three-dimensional movement.


def cmd2pwm(thrust, pwm2rpm_scale, pwm2rpm_const, ct, pwm_min, pwm_max):
    '''Generic cmd to pwm function.

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
    '''
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
    '''Computes motor squared rpm from pwm.

    Args:
        pwm (ndarray): Array of length 4 containing PWM.
        pwm2rpm_scale (float): Scaling factor between PWM and RPMs.
        pwm2rpm_const (float): Constant factor between PWM and RPMs.

    Returns:
        ndarray: Array of length 4 containing RPMs.
    '''
    rpm = pwm2rpm_scale * pwm + pwm2rpm_const
    return rpm
