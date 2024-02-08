"""AER1217 course project utility classes and functions.

"""
import os
import math
import time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation
from enum import Enum
from functools import wraps


class Command(Enum):
    """Command types that can be used with pycffirmware.

    """
    FINISHED = -1 # Args: None (exits the control loop)
    NONE = 0 # Args: None (do nothing)

    FULLSTATE = 1 # Args: [pos, vel, acc, yaw, rpy_rate] 
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.cmdFullState
    TAKEOFF = 2 # Args: [height, duration]
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.takeoff
    LAND = 3 # Args: [height, duration]
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.land
    STOP = 4 # Args: None
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.stop
    GOTO = 5 # Args: [pos, yaw, duration, relative (bool)]
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.goTo

    NOTIFYSETPOINTSTOP = 6 # Args: None
        # Must be called to transfer drone state from low level control (cmdFullState) to high level control (takeoff, land, goto)
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.notifySetpointsStop


class PIDController():
    """PID control class for Crazyflies.

    """

    def __init__(self,
                 g: float = 9.8,
                 m: float = 0.036,
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
        self.grav = g
        self.mass = m
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
        self.MIXER_MATRIX = np.array([[.5, -.5, 1], [.5, .5, -1], [-.5, .5, 1], [-.5, -.5, -1]])
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
                        target_acc=np.zeros(3),
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
        print("----------------_compute_force_and_euler------------------")
        self.control_counter += 1
        # thrust, computed_target_rpy, pos_e = self._compute_force_and_euler(control_timestep,
        #                                                                    cur_pos,
        #                                                                    cur_quat,
        #                                                                    cur_vel,
        #                                                                    target_pos,
        #                                                                    target_rpy,
        #                                                                    target_vel,
        #                                                                    target_acc
        #                                                                    )
        thrust, computed_target_rpy, pos_e = self._compute_desired_force_and_euler(control_timestep,
                                                                           cur_pos,
                                                                           cur_quat,
                                                                           cur_vel,
                                                                           target_pos,
                                                                           target_rpy,
                                                                           target_vel,
                                                                           target_acc
                                                                           )



        rpm = self._compute_rpms(control_timestep,
                                 thrust,
                                 cur_quat,
                                 computed_target_rpy,
                                 target_rpy_rates
                                 )
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]

    #TODO: geometric control
    def _compute_desired_force_and_euler(self,
                                 control_timestep,
                                 cur_pos,
                                 cur_quat,
                                 cur_vel,
                                 target_pos,
                                 target_rpy,
                                 target_vel,
                                 target_acc
                                 ):
        desired_acc = target_acc
        desired_yaw = target_rpy[2]

        cur_rot = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        self.P_COEFF_GEO = np.array([10.0, 10.0, 5.0])
        self.D_COEFF_GEO = np.array([4.0, 4.0, 2.0])
        self.P_ERR_MAX= np.array([0.6, 0.6, 0.3])
        self.V_ERR_MAX = np.array([1.0, 1.0, 1.0])

        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel

        pos_e = np.clip(pos_e, -self.P_ERR_MAX, self.P_ERR_MAX)
        vel_e = np.clip(vel_e, -self.V_ERR_MAX, self.V_ERR_MAX)

        acc_cmd = np.multiply(self.P_COEFF_GEO, pos_e) \
                  + np.multiply(self.D_COEFF_GEO, vel_e) \
                  + desired_acc \
                  + np.array([0, 0, self.grav])
        acc_norm = np.linalg.norm(acc_cmd)      
        thrust_cmd = acc_norm * self.mass

        # scalar_thrust = max(0., np.dot(thrust_cmd, cur_rot[:, 2]))
        thrust = (math.sqrt(thrust_cmd / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE

        q_c = p.getQuaternionFromEuler([0,0,desired_yaw])
        q_c_rot = np.array(p.getMatrixFromQuaternion(q_c)).reshape(3, 3) 
        x_c = q_c_rot.dot(np.array([1, 0, 0]))
        y_c = q_c_rot.dot(np.array([0, 1, 0]))
 
        z_B = acc_cmd / acc_norm
        x_B = np.cross(y_c, z_B) / np.linalg.norm(np.cross(y_c, z_B))
        y_B = np.cross(z_B, x_B) / np.linalg.norm(np.cross(z_B, x_B))
        desired_rot = (np.vstack([x_B, y_B, z_B])).transpose()

        # # compute bodyrate command
        # self.P_COEFF_ATT_XY = 0.0
        # self.P_COEFF_ATT_Z = 0.0
        # rot_k_matrix = np.diag([self.P_COEFF_ATT_XY, self.P_COEFF_ATT_XY, self.P_COEFF_ATT_Z])
        # rot_e = np.dot((cur_rot.transpose()), desired_rot)
        # q_e = Rotation.from_matrix(rot_e).as_quat()
        # tmp = np.array([q_e[0] * q_e[1] - q_e[2] * q_e[3], q_e[0] * q_e[2] - q_e[1] * q_e[3], q_e[3]])
        # rate_cmd = rot_k_matrix.dot(tmp) * 2.0 / math.sqrt(q_e[0] * q_e[0] + q_e[3] * q_e[3])
        
        desired_euler = (Rotation.from_matrix(desired_rot)).as_euler('XYZ', degrees=False)
        if np.any(np.abs(desired_euler) > math.pi):
            raise ValueError("\n[ERROR] ctrl it", self.control_counter, "in Control._compute_desired_force_and_euler(), values outside range [-pi,pi]")

        return thrust, desired_euler, pos_e

    def _compute_force_and_euler(self,
                                 control_timestep,
                                 cur_pos,
                                 cur_quat,
                                 cur_vel,
                                 target_pos,
                                 target_rpy,
                                 target_vel,
                                 target_acc
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
        # self.I_COEFF_FOR = 0.0
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

def timing_step(function):
    """Interstep learning timing decorator.

    """
    @wraps(function)
    def wrap(*args, **keyword_args):
        start = time.time()
        result = function(*args, **keyword_args)
        elapsed = time.time()-start
        args[0].interstep_learning_time += elapsed
        args[0].interstep_learning_occurrences += 1
        if elapsed >= args[0].CTRL_TIMESTEP:
            print('\n[WARNING] Function "{}" took: {} sec (too slow)'.format(function.__name__, elapsed))
        if args[0].VERBOSE and args[0].interstep_counter%int(args[0].CTRL_FREQ/2) == 0:
            print('\n{}-th call to function "{}" took: {} sec'.format(args[0].interstep_counter, function.__name__, elapsed))
        return result
    return wrap

def timing_ep(function):
    """Interepisode learning timing decorator.

    """
    @wraps(function)
    def wrap(*args, **keyword_args):
        start = time.time()
        result = function(*args, **keyword_args)
        elapsed = time.time()-start
        args[0].interepisode_learning_time = elapsed
        if args[0].VERBOSE:
            print('\n{}-th call to function "{}" took: {} sec'.format(args[0].interepisode_counter, function.__name__, elapsed))
        return result
    return wrap

def plot_trajectory(t_scaled,
                    waypoints,
                    ref_x,
                    ref_y,
                    ref_z
                    ):
    """Plot the trajectory with matplotlib.

    """
    # Plot each dimension.
    _, axs = plt.subplots(3, 1)
    axs[0].plot(t_scaled, ref_x)
    axs[0].set_ylabel('x (m)')
    axs[1].plot(t_scaled, ref_y)
    axs[1].set_ylabel('y (m)')
    axs[2].plot(t_scaled, ref_z)
    axs[2].set_ylabel('z (m)')
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    # Plot in 3D.
    ax = plt.axes(projection='3d')
    ax.plot3D(ref_x, ref_y, ref_z)
    ax.scatter3D(waypoints[:,0], waypoints[:,1], waypoints[:,2])
    ax.set_xlim([-3.5, 3.5])
    ax.set_ylim([-3.5, 3.5])
    ax.set_zlim([0.0, 2.0])
    plt.show(block=False)
    plt.pause(2)
    plt.close()

def draw_trajectory(initial_info,
                    waypoints,
                    ref_x,
                    ref_y,
                    ref_z
                    ):
    """Draw a trajectory in PyBullet's GUI.

    """
    for point in waypoints:
        p.loadURDF(os.path.join(initial_info["urdf_dir"], "sphere.urdf"),
                   [point[0], point[1], point[2]],
                   p.getQuaternionFromEuler([0,0,0]),
                   physicsClientId=initial_info["pyb_client"])
    step = int(ref_x.shape[0]/50)
    for i in range(step, ref_x.shape[0], step):
        p.addUserDebugLine(lineFromXYZ=[ref_x[i-step], ref_y[i-step], ref_z[i-step]],
                           lineToXYZ=[ref_x[i], ref_y[i], ref_z[i]],
                           lineColorRGB=[1, 0, 0], lineWidth=3,
                           physicsClientId=initial_info["pyb_client"])
    p.addUserDebugLine(lineFromXYZ=[ref_x[i], ref_y[i], ref_z[i]],
                       lineToXYZ=[ref_x[-1], ref_y[-1], ref_z[-1]],
                       lineColorRGB=[1, 0, 0], lineWidth=3,
                       physicsClientId=initial_info["pyb_client"])

def thrusts(controller,
            ctrl_timestep,
            kf,
            obs,
            target_p,
            target_v,
            target_a
            ):
    """Compute thrusts from PID control.

    """
    rpms, _, _ = controller.compute_control(control_timestep=ctrl_timestep,
                                            cur_pos=np.array([obs[0],obs[2],obs[4]]),
                                            cur_quat=np.array(p.getQuaternionFromEuler([obs[6],obs[7],obs[8]])),
                                            cur_vel=np.array([obs[1],obs[3],obs[5]]),
                                            cur_ang_vel=np.array([obs[9],obs[10],obs[11]]),
                                            target_pos=target_p,
                                            target_vel=target_v,
                                            target_acc=target_a
                                            )
    return kf * rpms**2
