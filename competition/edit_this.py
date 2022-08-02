"""Create a controller

"""
import os
from enum import Enum
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import PIDController

class Command(Enum):
    NONE = 0 # Args: Empty
    FULLSTATE = 1 # Args: [pos, vel, acc, rpy, rpy_rate, iteration]
    TAKEOFF = 2 # Args: [height, duration]
    LAND = 3 # Args: [height, duration]
    STOP = 4 # Args: Empty
    GOTO = 5 # Args: [x, y, z, yaw, duration, relative (bool)]

class Controller():
    """Editable controller class.

    """

    def __init__(self,
                 env
                 ):
        """Initialization of the controller.

        """

        # PID Controller.
        self.ctrl = PIDController()

        # Save environment parameters.
        self.CTRL_TIMESTEP = env.CTRL_TIMESTEP
        self.KF = env.KF

        # Curve fitting with waypoints.
        waypoints = np.array([
                                (0, 0, 0),
                                (1, 0.5, 1.25),
                                (1.5, 0.5, 1.25),
                                (2, 1, 1.25), 
                                (2.5, 1, 1.25), 
                                (3, 2, 1.25),
                                (1.5, 3, 1.25),
                                (0.5, 3, 1.25),
                                (0.5, 5, 1.25),
                            ])
        deg = 12
        t = np.arange(waypoints.shape[0])
        fit_x = np.polyfit(t, waypoints[:,0], deg)
        fit_y = np.polyfit(t, waypoints[:,1], deg)
        fit_z = np.polyfit(t, waypoints[:,2], deg)
        fx = np.poly1d(fit_x)
        fy = np.poly1d(fit_y)
        fz = np.poly1d(fit_z)
        t_scaled = np.linspace(t[0], t[-1], env.EPISODE_LEN_SEC*env.CTRL_FREQ)
        self.CTRL_FREQ = env.CTRL_FREQ
        self.ref_x = fx(t_scaled)
        self.ref_y = fy(t_scaled)
        self.ref_z = fz(t_scaled)

        # Plot each dimension.
        # plt.plot(t_scaled, x_scaled)
        # plt.plot(t_scaled, x_scaled)
        # plt.plot(t_scaled, x_scaled)
        # plt.show()

        # Plot in 3D.
        ax = plt.axes(projection='3d')
        ax.plot3D(self.ref_x, self.ref_y, self.ref_z)
        ax.scatter3D(waypoints[:,0], waypoints[:,1], waypoints[:,2])
        plt.show()

        # Draw trajectory.
        for point in waypoints:
            p.loadURDF(os.path.join(env.URDF_DIR, "sphere.urdf"),
                       [point[0], point[1], point[2]],
                       p.getQuaternionFromEuler([0,0,0]),
                       physicsClientId=env.PYB_CLIENT)
        step = 10
        for i in range(step, self.ref_x.shape[0], step):
            p.addUserDebugLine(lineFromXYZ=[self.ref_x[i-step], self.ref_y[i-step], self.ref_z[i-step]],
                               lineToXYZ=[self.ref_x[i], self.ref_y[i], self.ref_z[i]],
                               lineColorRGB=[1, 0, 0],
                               physicsClientId=env.PYB_CLIENT)
        p.addUserDebugLine(lineFromXYZ=[self.ref_x[i], self.ref_y[i], self.ref_z[i]],
                           lineToXYZ=[self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]],
                           lineColorRGB=[1, 0, 0],
                           physicsClientId=env.PYB_CLIENT)

    def getCmd(self, time, vicon_pos=None, est_vel=None, est_acc=None, est_rpy=None, est_rpy_rates=None):
        '''
        This function should return the target position, velocity, acceleration, attitude, and attitude rates to be sent 
        from crazyswarm to the crazyflie using a cmdFullState call. 

        Arguments 
        * time (s) 
        * vicon_pos - contains feedback from the vicon tracking system about where your drone marker is (mm) 
        * est_vel - estimation of drone velocity from vicon system 
        * est_acc - estimation of drone acceleration from vicon system 
        * est_rpy - estimation of drone attitude from vicon system 
        * est_rpy_rates - estimation of drone body rates from vicon system 
        '''
        iteration = int(time*self.CTRL_FREQ)
        
        # Your code goes here 

        target_pos = np.array([self.ref_x[iteration], self.ref_y[iteration], self.ref_z[iteration]])
        target_vel = np.zeros(3)
        target_acc = np.zeros(3)
        target_rpy = np.zeros(3)
        target_rpy_rates = np.zeros(3)

        return Command(1), [target_pos, target_vel, target_acc, target_rpy, target_rpy_rates, iteration]

    def cmdFullState(self,
                     iteration,
                     obs
                     ):
        """Action selection.

        """
        rpms, _, _ = self.ctrl.compute_control(control_timestep=self.CTRL_TIMESTEP,
                                               cur_pos=np.array([obs[0],obs[2],obs[4]]),
                                               cur_quat=np.array(p.getQuaternionFromEuler([obs[6],obs[7],obs[8]])),
                                               cur_vel=np.array([obs[1],obs[3],obs[5]]),
                                               cur_ang_vel=np.array([obs[9],obs[10],obs[11]]),
                                               target_pos=np.array([self.ref_x[iteration], self.ref_y[iteration], self.ref_z[iteration]]),
                                               target_vel=np.zeros(3)
                                               )
        return self.KF * rpms**2
