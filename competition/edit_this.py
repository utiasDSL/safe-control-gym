"""Write your control strategy.

Then run:

    $ python3 getting_started.py --overrides ./getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS:` and `REPLACE THIS (START)` in this file.

    Change the code between the 4 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

"""
import os
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from enum import Enum
from collections import deque

from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import PIDController

try:
    import pycffirmware
except ImportError:
    FIRMWARE_INSTALLED = False
else:
    FIRMWARE_INSTALLED = True
finally:
    print("Module 'cffirmware' available:", FIRMWARE_INSTALLED)


class Command(Enum):
    NONE = 0 # Args: Empty
    FULLSTATE = 1 # Args: [pos, vel, acc, yaw, rpy_rate, iteration] 
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.cmdFullState
    TAKEOFF = 2 # Args: [height, duration]
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.takeoff
    LAND = 3 # Args: [height, duration]
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.land
    STOP = 4 # Args: Empty
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.stop
    GOTO = 5 # Args: [[x, y, z], yaw, duration, relative (bool)]
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.goTo


class Controller():
    """Template controller class.

    """

    def __init__(self,
                 initial_pos,
                 initial_info,
                 use_firmware: bool = False,
                 use_hardware: bool = False,
                 buffer_size: int = 100,
                 verbose: bool = False
                 ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori infromation
            contained in dictionary `initial_info`. Use this method to initialize constants, counters, pre-plan
            trajectories, etc.

        Args:
            initial_obs (ndarray): The initial observation of the quadrotor's state
                [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            initial_info (dict): The a priori information as a dictionary with keys
                'symbolic_model', 'nominal_physical_parameters', 'nominal_gates_pos', etc.
            use_firmware (bool, optional): Choice between the on-board controll in `pycffirmware`
                or simplified software-only alternative.
            use_hardware (bool, optional): Choice between simulation or hardware evaluation.
            buffer_size (int, optional): Size of the data buffers used in method `learn()`.
            verbose (bool, optional): Turn on and off additional printouts and plots.

        """

        self.use_hardware = use_hardware
        self.initial_pos = initial_pos

        # Save environment parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]

        # Store a priori scenario information.
        self.NOMINAL_GATES = initial_info["nominal_gates_pos"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        if use_firmware and not use_hardware:
            if not FIRMWARE_INSTALLED:
                raise RuntimeError("[ERROR] Module 'cffirmware' not installed, try set 'use_firmware' to False in 'getting_started.yaml'.")
            self.ctrl = None
        elif use_hardware:
            self.ctrl = None
        else:
            # Simple PID Controller.
            self.ctrl = PIDController()
            # Save additonal environment parameters.
            self.KF = initial_info["quadrotor_kf"]

        self.VERBOSE = verbose

        # Data buffers.
        self.action_buffer = deque([], maxlen=buffer_size)
        self.obs_buffer = deque([], maxlen=buffer_size)
        self.reward_buffer = deque([], maxlen=buffer_size)
        self.done_buffer = deque([], maxlen=buffer_size)
        self.info_buffer = deque([], maxlen=buffer_size)

        #########################
        # REPLACE THIS (START) ##
        #########################

        # Curve fitting with waypoints.
        if use_firmware:
            waypoints = [(0.1, 0.1, .75)]  # Hardcoded scenario knowledge.
        else:
            waypoints = [(0.1, 0.1, .0)]  # Hardcoded scenario knowledge.
        for idx, g in enumerate(self.NOMINAL_GATES):
            x = g[0]
            y = g[1]
            rot = g[5]
            if rot > 0.5*1.57 or rot < 0:  # Hardcoded scenario knowledge.
                if idx == 2:  # Hardcoded scenario knowledge.
                    waypoints.append((x+0.3, y-0.2, initial_info["gate_dimensions"]["height"]))
                    waypoints.append((x-0.3, y-0.2, initial_info["gate_dimensions"]["height"]))
                else:
                    waypoints.append((x-0.3, y, initial_info["gate_dimensions"]["height"]))
                    waypoints.append((x+0.3, y, initial_info["gate_dimensions"]["height"]))
            else:
                waypoints.append((x, y-0.3, initial_info["gate_dimensions"]["height"]))
                waypoints.append((x, y+0.3, initial_info["gate_dimensions"]["height"]))
        waypoints.append([initial_info["x_reference"][0], initial_info["x_reference"][2], initial_info["x_reference"][4]])
        self.waypoints = np.array(waypoints)
        deg = 6
        t = np.arange(self.waypoints.shape[0])
        fit_x = np.polyfit(t, self.waypoints[:,0], deg)
        fit_y = np.polyfit(t, self.waypoints[:,1], deg)
        fit_z = np.polyfit(t, self.waypoints[:,2], deg)
        fx = np.poly1d(fit_x)
        fy = np.poly1d(fit_y)
        fz = np.poly1d(fit_z)
        t_scaled = np.linspace(t[0], t[-1], int(0.5*initial_info["episode_len_sec"]*self.CTRL_FREQ))
        self.ref_x = fx(t_scaled)
        self.ref_y = fy(t_scaled)
        self.ref_z = fz(t_scaled)

        if self.VERBOSE:
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
            ax.scatter3D(self.waypoints[:,0], self.waypoints[:,1], self.waypoints[:,2])
            plt.show(block=False)
            plt.pause(2)
            plt.close()

        self._draw_trajectory(initial_info)

        #########################
        # REPLACE THIS (END) ####
        #########################

    def cmdFirmware(self,
                    time,
                    obs,
                    vicon_pos=None,
                    est_vel=None,
                    est_acc=None,
                    est_rpy=None,
                    est_rpy_rates=None
                    ):
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this function to return the target position, velocity, acceleration, attitude, and attitude rates to be sent
            from Crazyswarm to the Crazyflie using, e.g., a `cmdFullState` call. 

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].
            vicon_pos (ndarray, optional): Feedback from the vicon tracking system about where your drone marker is (mm).
            est_vel (ndarray, optional): Estimation of drone velocity from Vicon system.
            est_acc (ndarray, optional): Estimation of drone acceleration from Vicon system.
            est_rpy (ndarray, optional): Estimation of drone attitude from Vicon system
            est_rpy_rates (ndarray, optional): Estimation of drone body rates from vicon system.

        Returns:
            Command: selected type of command (takeOff, cmdFullState, etc., see Enum-like class `Command`).
            List: arguments for the type of command (see comments in class `Command`)

        """
        if not self.use_hardware:
            if self.ctrl is not None:
                raise RuntimeError("[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False.")
            if not FIRMWARE_INSTALLED:
                raise RuntimeError("[ERROR] Module 'cffirmware' not installed.")

        iteration = int(time*self.CTRL_FREQ)

        #########################
        # REPLACE THIS (START) ##
        #########################

        if iteration == 0:
            height = 1.0
            duration = 2

            command_type = Command(2)  # Take-off.
            args = [height, duration]

        elif iteration >= 3*self.CTRL_FREQ and iteration < 20*self.CTRL_FREQ:
            step = min(iteration-3*self.CTRL_FREQ, len(self.ref_x) -1)
            target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
            target_vel = np.zeros(3)
            target_acc = np.zeros(3)
            target_yaw = 0.
            target_rpy_rates = np.zeros(3)

            command_type = Command(1)  # cmdFullState.
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates, time]

        elif iteration == 20*self.CTRL_FREQ:
            x = self.ref_x[-1]
            y = self.ref_y[-1]
            z = self.ref_z[-1]
            yaw = 0.
            duration = 2.5

            command_type = Command(5)  # goTo.
            args = [x, y, z, yaw, duration, False]

        elif iteration == 20*self.CTRL_FREQ+1:
            x = self.ref_x[-1]
            y = self.ref_y[-1]
            z = 1.5
            yaw = 0.
            duration = 2.5

            command_type = Command(5)  # goTo.
            args = [x, y, z, yaw, duration, False]

        elif iteration == 23*self.CTRL_FREQ:
            x = 0.
            y = 0.
            z = 1.5
            yaw = 0.
            duration = 6

            command_type = Command(5)  # goTo.
            args = [x, y, z, yaw, duration, False]

        elif iteration == 30*self.CTRL_FREQ:
            height = 0.
            duration = 3

            command_type = Command(3)  # Land.
            args = [height, duration]


        else:
            command_type = Command(0)  # None.
            args = []

        #########################
        # REPLACE THIS (END) ####
        #########################

        return command_type, args

    def cmdSimOnly(self,
                   time,
                   obs
                   ):
        """PID per-propeller thrusts with a simplified, software-only PID quadrotor controller.

        INSTRUCTIONS:
            Re-implement this function to return the target position and velocity.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's state [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].

        Returns:
            List: thrusts (in m, len == 3) to be commanded to each motor.
            List: thrusts (in m/s, len == 3) to be commanded to each motor.

        """
        if self.use_hardware:
            raise NotImplementedError("Must control hardware through cmdFullState. Action space commands not supported.")
        if self.ctrl is None:
            raise RuntimeError("[ERROR] Attempting to use method 'cmdSimOnly' but Controller was created with 'use_firmware' = True.")
        # if FIRMWARE_INSTALLED:
        #     print("[WARNING] Using method 'cmdSimOnly' but module 'cffirmware' is available.")

        iteration = int(time*self.CTRL_FREQ)

        #########################
        # REPLACE THIS (START) ##
        #########################

        if iteration < len(self.ref_x):
            target_p = np.array([self.ref_x[iteration], self.ref_y[iteration], self.ref_z[iteration]])
        else:
            target_p = np.array([self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]])

        target_v = np.zeros(3)

        #########################
        # REPLACE THIS (END) ####
        #########################

        return target_p, target_v

    def interStepLearn(self,
                       action,
                       obs,
                       reward,
                       done,
                       info):
        """Learning and controller updates called between control steps.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions, observations,
            rewards, done flags, and information dictionaries to learn, adapt, and/or re-plan.

        Args:
            action (List): Most recent applied action.
            obs (List): Most recent observation of the quadrotor state.
            reward (float): Most recent reward.
            done (bool): Most recent done flag.
            info (dict): Most recent information dictionary.

        """

        self.action_buffer.append(action)
        self.obs_buffer.append(obs)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.info_buffer.append(info)

        #########################
        # REPLACE THIS (START) ##
        #########################

        pass

        #########################
        # REPLACE THIS (END) ####
        #########################

    def interEpisodeLearn(self):
        """Learning and controller updates called between episodes.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions, observations,
            rewards, done flags, and information dictionaries to learn, adapt, and/or re-plan.

        """

        #########################
        # REPLACE THIS (START) ##
        #########################

        _ = self.action_buffer
        _ = self.obs_buffer
        _ = self.reward_buffer
        _ = self.done_buffer
        _ = self.info_buffer

        #########################
        # REPLACE THIS (END) ####
        #########################

    def _thrusts(self,
                 obs,
                 target,
                 target_v
                 ):
        """Do not modify this.

        """
        rpms, _, _ = self.ctrl.compute_control(control_timestep=self.CTRL_TIMESTEP,
                                               cur_pos=np.array([obs[0],obs[2],obs[4]]),
                                               cur_quat=np.array(p.getQuaternionFromEuler([obs[6],obs[7],obs[8]])),
                                               cur_vel=np.array([obs[1],obs[3],obs[5]]),
                                               cur_ang_vel=np.array([obs[9],obs[10],obs[11]]),
                                               target_pos=target,
                                               target_vel=target_v
                                               )
        return self.KF * rpms**2

    def _draw_trajectory(self, initial_info):
        # Draw trajectory.
        for point in self.waypoints:
            p.loadURDF(os.path.join(initial_info["urdf_dir"], "sphere.urdf"),
                       [point[0], point[1], point[2]],
                       p.getQuaternionFromEuler([0,0,0]),
                       physicsClientId=initial_info["pyb_client"])
        step = int(self.ref_x.shape[0]/50)
        for i in range(step, self.ref_x.shape[0], step):
            p.addUserDebugLine(lineFromXYZ=[self.ref_x[i-step], self.ref_y[i-step], self.ref_z[i-step]],
                               lineToXYZ=[self.ref_x[i], self.ref_y[i], self.ref_z[i]],
                               lineColorRGB=[1, 0, 0],
                               physicsClientId=initial_info["pyb_client"])
        p.addUserDebugLine(lineFromXYZ=[self.ref_x[i], self.ref_y[i], self.ref_z[i]],
                           lineToXYZ=[self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]],
                           lineColorRGB=[1, 0, 0],
                           physicsClientId=initial_info["pyb_client"])
