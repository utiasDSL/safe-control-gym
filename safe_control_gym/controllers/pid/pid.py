"""PID control class for Crazyflies.

Based on work conducted at UTIAS' DSL by SiQi Zhou and James Xu.

"""


import numpy as np
import math
import pybullet as p
from scipy.spatial.transform import Rotation
from munch import munchify

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.envs.benchmark_env import Task


class PID(BaseController):
    """ PID Class.

    """

    def __init__(self,
                 env_func=None,
                 g = 9.8,
                 KF = 3.16e-10,
                 KM = 7.94e-12,
                 P_COEFF_FOR = np.array([.4, .4, 1.25]),
                 I_COEFF_FOR = np.array([.05, .05, .05]),
                 D_COEFF_FOR = np.array([.2, .2, .5]),
                 P_COEFF_TOR = np.array([70000., 70000., 60000.]),
                 I_COEFF_TOR = np.array([.0, .0, 500.]),
                 D_COEFF_TOR = np.array([20000., 20000., 12000.]),
                 PWM2RPM_SCALE = 0.2685,
                 PWM2RPM_CONST = 4070.3,
                 MIN_PWM = 20000,
                 MAX_PWM = 65535,
                 MIXER_MATRIX = np.array([ [.5, -.5,  -1], [.5, .5, 1], [-.5,  .5,  -1], [-.5, -.5, 1] ]),
                 **kwargs
                 ):
        """Common control classes __init__ method.

        Args
            g (float, optional): The gravitational acceleration in m/s^2.

        """

        super().__init__(env_func, **kwargs)

        self.GRAVITY = float(g) * 0.027
        self.KF = float(KF)
        self.KM = float(KM)
        self.P_COEFF_FOR = np.array(P_COEFF_FOR)
        self.I_COEFF_FOR = np.array(I_COEFF_FOR)
        self.D_COEFF_FOR = np.array(D_COEFF_FOR)
        self.P_COEFF_TOR = np.array(P_COEFF_TOR)
        self.I_COEFF_TOR = np.array(I_COEFF_TOR)
        self.D_COEFF_TOR = np.array(D_COEFF_TOR)
        self.PWM2RPM_SCALE = float(PWM2RPM_SCALE)
        self.PWM2RPM_CONST = float(PWM2RPM_CONST)
        self.MIN_PWM = float(MIN_PWM)
        self.MAX_PWM = float(MAX_PWM)
        self.MIXER_MATRIX = np.array(MIXER_MATRIX)

        self.reset()

    def run(self,
            iterations,
            **kwargs
            ):
        """Computes the PID control action (as RPMs) for a single drone.

        This methods sequentially calls `_dslPIDPositionControl()` and `_dslPIDAttitudeControl()`.
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
        action = np.zeros(2)

        for i in range(iterations):
            # Step the environment and print all returned information.
            obs, reward, done, info = self.env.step(action)

            cur_pos=np.array([obs[0], 0, obs[2]])
            cur_quat=np.array(p.getQuaternionFromEuler([0, obs[4], 0]))
            cur_vel=np.array([obs[1], 0, obs[3]])
            cur_ang_vel=np.array([0, obs[4], 0])

            if self.env.TASK == Task.TRAJ_TRACKING:
                target_pos=np.array([
                                        self.reference[i-1,0],
                                        0,
                                        self.reference[i-1,2]
                                    ])
                target_vel=np.array([
                                        self.reference[i-1,1],
                                        0,
                                        self.reference[i-1,3]
                                    ])
            elif self.env.TASK == Task.STABILIZATION:
                target_pos=np.array([self.reference[0], 0, self.reference[2] ])
                target_vel=np.array([0, 0, 0 ])
            else:
                raise NotImplementedError
            
            target_rpy = np.zeros(3)
            target_rpy_rates = np.zeros(3)

            # Compute the next action.
            self.control_counter += 1
            thrust, computed_target_rpy, pos_e = self._dslPIDPositionControl(self.control_timestep,
                                                                             cur_pos,
                                                                             cur_quat,
                                                                             cur_vel,
                                                                             target_pos,
                                                                             target_rpy,
                                                                             target_vel
                                                                             )
            rpm = self._dslPIDAttitudeControl(self.control_timestep,
                                             thrust,
                                             cur_quat,
                                             computed_target_rpy,
                                             target_rpy_rates
                                             )
            cur_rpy = p.getEulerFromQuaternion(cur_quat)
            
            action = rpm
            action = self.KF * action**2
            action = np.array([action[0]+action[3], action[1]+action[2]])
            
            self.results_dict['obs'].append(obs)
            self.results_dict['reward'].append(reward)
            self.results_dict['done'].append(done)
            self.results_dict['info'].append(info)
            self.results_dict['action'].append(action)

        self.close_results_dict()

        return self.results_dict
    
    def _dslPIDPositionControl(self,
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
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:,2]))
        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        
        # Target rotation.
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        
        if np.any(np.abs(target_euler) > math.pi):
            print("\n[ERROR] ctrl it", self.control_counter, "in Control._dslPIDPositionControl(), values outside range [-pi,pi]")
        
        return thrust, target_euler, pos_e
    
    def _dslPIDAttitudeControl(self,
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
        w,x,y,z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()),cur_rotation) - np.dot(cur_rotation.transpose(),target_rotation)
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

    def close(self):
        """Cleans up resources.

        """
        self.env.close()

    def close_results_dict(self):
        """Cleanup the rtesults dict and munchify it.

        """
        self.results_dict['obs'] = np.vstack(self.results_dict['obs'])
        self.results_dict['reward'] = np.vstack(self.results_dict['reward'])
        self.results_dict['done'] = np.vstack(self.results_dict['done'])
        self.results_dict['info'] = np.vstack(self.results_dict['info'])
        self.results_dict['action'] = np.vstack(self.results_dict['action'])

        self.results_dict = munchify(self.results_dict)

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        self.env = self.env_func()
        initial_obs, initial_info = self.env.reset()
        self.control_timestep = self.env.CTRL_TIMESTEP
        self.reference = initial_info['x_reference']

        self.control_counter = 0
        
        # Clear the last roll, pitch, and yaw.
        self.last_rpy = np.zeros(3)
        
        # Clear PID control variables.
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

        self.results_dict = { 'obs': [],
                        'reward': [],
                        'done': [],
                        'info': [],
                        'action': [],
                        }
