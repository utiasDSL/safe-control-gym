"""Create a controller.

Then run:

    $ python3 getting_started.py --overrides ./getting_started.yaml

"""
import os
from enum import Enum
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import PIDController

try:
    import cffirmware
except ImportError:
    FIRMWARE_INSTALLED = False
else:
    FIRMWARE_INSTALLED = True
finally:
    print("Module 'cffirmware' available:", FIRMWARE_INSTALLED)


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
                 initial_obs,
                 initial_info,
                 use_firmware: bool = False
                 ):
        """Initialization of the controller.

        """

        if use_firmware:
            if not FIRMWARE_INSTALLED:
                raise RuntimeError("[ERROR] Module 'cffirmware' not installed.")
            self.ctrl = None
        else:
            # Simple PID Controller.
            self.ctrl = PIDController()

        # Save environment parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.KF = initial_info["quadrotor_kf"]
        self.NOMINAL_GATES = initial_info["nominal_gates_pos"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        #########################
        # REPLACE THIS (START) ##
        #########################

        # Curve fitting with waypoints.
        waypoints = [(0, 0, 0)]
        for idx, g in enumerate(self.NOMINAL_GATES):
            x = g[0]
            y = g[1]
            rot = g[5]
            if rot > 0.5*1.57:
                if idx == 2:
                    waypoints.append((x+0.3, y, initial_info["gate_dimensions"]["height"]))
                    waypoints.append((x-0.3, y, initial_info["gate_dimensions"]["height"]))
                else:
                    waypoints.append((x-0.3, y, initial_info["gate_dimensions"]["height"]))
                    waypoints.append((x+0.3, y, initial_info["gate_dimensions"]["height"]))
            else:
                waypoints.append((x, y-0.3, initial_info["gate_dimensions"]["height"]))
                waypoints.append((x, y+0.3, initial_info["gate_dimensions"]["height"]))
        waypoints.append((x+0.25, y+1.5, initial_info["gate_dimensions"]["height"]))
        waypoints = np.array(waypoints)
        deg = 6
        t = np.arange(waypoints.shape[0])
        fit_x = np.polyfit(t, waypoints[:,0], deg)
        fit_y = np.polyfit(t, waypoints[:,1], deg)
        fit_z = np.polyfit(t, waypoints[:,2], deg)
        fx = np.poly1d(fit_x)
        fy = np.poly1d(fit_y)
        fz = np.poly1d(fit_z)
        t_scaled = np.linspace(t[0], t[-1], int(0.75*initial_info["episode_len_sec"]*self.CTRL_FREQ))
        self.ref_x = fx(t_scaled)
        self.ref_y = fy(t_scaled)
        self.ref_z = fz(t_scaled)

        # Plot each dimension.
        _, axs = plt.subplots(3, 1)
        axs[0].plot(t_scaled, self.ref_x)
        axs[0].set_ylabel('x (m)')
        axs[1].plot(t_scaled, self.ref_y)
        axs[1].set_ylabel('y (m)')
        axs[2].plot(t_scaled, self.ref_z)
        axs[2].set_ylabel('z (m)')
        plt.show(block=False)
        plt.pause(2)
        plt.close()

        # Plot in 3D.
        ax = plt.axes(projection='3d')
        ax.plot3D(self.ref_x, self.ref_y, self.ref_z)
        ax.scatter3D(waypoints[:,0], waypoints[:,1], waypoints[:,2])
        plt.show(block=False)
        plt.pause(2)
        plt.close()

        # Draw trajectory.
        for point in waypoints:
            p.loadURDF(os.path.join(initial_info["urdf_dir"], "sphere.urdf"),
                       [point[0], point[1], point[2]],
                       p.getQuaternionFromEuler([0,0,0]),
                       physicsClientId=initial_info["pyb_client"])
        step = 10
        for i in range(step, self.ref_x.shape[0], step):
            p.addUserDebugLine(lineFromXYZ=[self.ref_x[i-step], self.ref_y[i-step], self.ref_z[i-step]],
                               lineToXYZ=[self.ref_x[i], self.ref_y[i], self.ref_z[i]],
                               lineColorRGB=[1, 0, 0],
                               physicsClientId=initial_info["pyb_client"])
        p.addUserDebugLine(lineFromXYZ=[self.ref_x[i], self.ref_y[i], self.ref_z[i]],
                           lineToXYZ=[self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]],
                           lineColorRGB=[1, 0, 0],
                           physicsClientId=initial_info["pyb_client"])

        #########################
        # REPLACE THIS (END) ####
        #########################

    def cmdFirmware(self,
                    time,
                    obs,
                    vicon_pos=None, est_vel=None, est_acc=None, est_rpy=None, est_rpy_rates=None):
        """
        This function should return the target position, velocity, acceleration, attitude, and attitude rates to be sent 
        from crazyswarm to the crazyflie using a cmdFullState call. 

        Arguments 
        * time (s) 
        * vicon_pos - contains feedback from the vicon tracking system about where your drone marker is (mm) 
        * est_vel - estimation of drone velocity from vicon system 
        * est_acc - estimation of drone acceleration from vicon system 
        * est_rpy - estimation of drone attitude from vicon system 
        * est_rpy_rates - estimation of drone body rates from vicon system 
        """
        if self.ctrl is not None:
            print("[WARNING] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False.")
        if not FIRMWARE_INSTALLED:
            raise RuntimeError("[ERROR] Module 'cffirmware' not installed.")

        iteration = int(time*self.CTRL_FREQ)

        #########################
        # REPLACE THIS (START) ##
        #########################

        if iteration < len(self.ref_x):
            target_pos = np.array([self.ref_x[iteration], self.ref_y[iteration], self.ref_z[iteration]])
        else:
            target_pos = np.array([self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]])
        target_vel = np.zeros(3)
        target_acc = np.zeros(3)
        target_rpy = np.zeros(3)
        target_rpy_rates = np.zeros(3)

        command_type = Command(1)
        args = [target_pos, target_vel, target_acc, target_rpy, target_rpy_rates, iteration]

        #########################
        # REPLACE THIS (END) ####
        #########################

        return command_type, args

    def cmdSimOnly(self,
                   time,
                   obs
                   ):
        """Action selection.

        """
        if FIRMWARE_INSTALLED:
            print("[WARNING] Using method 'cmdSimOnly' but module 'cffirmware' is available.")
        if self.ctrl is None:
            raise RuntimeError("[ERROR] Attempting to use method 'cmdSimOnly' but Controller was created with 'use_firmware' = True.")

        iteration = int(time*self.CTRL_FREQ)
        if iteration < len(self.ref_x):
            target = np.array([self.ref_x[iteration], self.ref_y[iteration], self.ref_z[iteration]])
        else:
            target = np.array([self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]])
        rpms, _, _ = self.ctrl.compute_control(control_timestep=self.CTRL_TIMESTEP,
                                               cur_pos=np.array([obs[0],obs[2],obs[4]]),
                                               cur_quat=np.array(p.getQuaternionFromEuler([obs[6],obs[7],obs[8]])),
                                               cur_vel=np.array([obs[1],obs[3],obs[5]]),
                                               cur_ang_vel=np.array([obs[9],obs[10],obs[11]]),
                                               target_pos=target,
                                               target_vel=np.zeros(3)
                                               )
        return self.KF * rpms**2
