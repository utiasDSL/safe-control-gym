"""Write your control strategy.

Then run:

    $ python3 getting_started.py --overrides ./getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS:` and `REPLACE THIS (START)` in this file.

    Change the code between the 5 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

    They are in methods:
        1) __init__
        2) cmdFirmware
        3) cmdSimOnly (optional)
        4) interStepLearn (optional)
        5) interEpisodeLearn (optional)

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
    """Command types that can be used with pycffirmware.

    """
    NONE = 0 # Args: Empty
    FULLSTATE = 1 # Args: [pos, vel, acc, yaw, rpy_rate, curr_time] 
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.cmdFullState
    TAKEOFF = 2 # Args: [height, duration]
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.takeoff
    LAND = 3 # Args: [height, duration]
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.land
    STOP = 4 # Args: Empty
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.stop
    GOTO = 5 # Args: [x, y, z, yaw, duration, relative (bool)]
        # https://crazyswarm.readthedocs.io/en/latest/api.html#pycrazyswarm.crazyflie.Crazyflie.goTo


class Controller():
    """Template controller class.

    """

    def __init__(self,
                 initial_obs,
                 initial_info,
                 use_firmware: bool = False,
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
            buffer_size (int, optional): Size of the data buffers used in method `learn()`.
            verbose (bool, optional): Turn on and off additional printouts and plots.

        """

        # Save environment parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose

        # Store a priori scenario information.
        self.NOMINAL_GATES = initial_info["nominal_gates_pos"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        # Check for pycffirmware.
        if use_firmware:
            if not FIRMWARE_INSTALLED:
                raise RuntimeError("[ERROR] Module 'cffirmware' not installed, try set 'use_firmware' to False in 'getting_started.yaml'.")
            else:
                self.ctrl = None
        else:
            # Initialized simple PID Controller.
            self.ctrl = PIDController()
            # Save additonal environment parameters.
            self.KF = initial_info["quadrotor_kf"]

        # Data buffers.
        self.action_buffer = deque([], maxlen=buffer_size)
        self.obs_buffer = deque([], maxlen=buffer_size)
        self.reward_buffer = deque([], maxlen=buffer_size)
        self.done_buffer = deque([], maxlen=buffer_size)
        self.info_buffer = deque([], maxlen=buffer_size)

        #########################
        # REPLACE THIS (START) ##
        #########################

        # Example: curve fitting with waypoints.
        if use_firmware:
            waypoints = [(self.initial_obs[0], self.initial_obs[2], .75)]  # Height is hardcoded scenario knowledge.
        else:
            waypoints = [(self.initial_obs[0], self.initial_obs[2], .0)]  # Height is hardcoded scenario knowledge
        for idx, g in enumerate(self.NOMINAL_GATES):
            x = g[0]
            y = g[1]
            rot = g[5]
            if rot > 0.5*1.57 or rot < 0:
                if idx == 2:  # Hardcoded scenario knowledge (direction in which to take gate 2.
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
        t_scaled = np.linspace(t[0], t[-1], int(15*self.CTRL_FREQ))
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

        # Draw the trajectory on PyBullet's GUI
        self._draw_trajectory(initial_info)

        #########################
        # REPLACE THIS (END) ####
        #########################

    def cmdFirmware(self,
                    time,
                    obs,
                    reward=None,
                    done=None,
                    info=None
                    ):
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this function to return the target position, velocity, acceleration, attitude, and attitude rates to be sent
            from Crazyswarm to the Crazyflie using, e.g., a `cmdFullState` call. 

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            Command: selected type of command (takeOff, cmdFullState, etc., see Enum-like class `Command`).
            List: arguments for the type of command (see comments in class `Command`)

        """

        if self.ctrl is not None:
            raise RuntimeError("[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False.")
        if not FIRMWARE_INSTALLED:
            raise RuntimeError("[ERROR] Module 'cffirmware' not installed.")

        iteration = int(time*self.CTRL_FREQ)

        #########################
        # REPLACE THIS (START) ##
        #########################

        # Handwritten solution for GitHub's example scenario.

        if iteration == 0:
            height = 0.75
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
            z = 1.5 
            yaw = 0.
            duration = 2.5

            command_type = Command(5)  # goTo.
            args = [x, y, z, yaw, duration, False]


        elif iteration == 23*self.CTRL_FREQ:
            x = self.initial_obs[0]
            y = self.initial_obs[2]
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
                   obs,
                   reward=None,
                   done=None,
                   info=None
                   ):
        """PID per-propeller thrusts with a simplified, software-only PID quadrotor controller.

        INSTRUCTIONS:
            Re-implement this function to return the target position and velocity.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's state [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            List: target position (len == 3).
            List: target velocity (len == 3).

        """

        if self.ctrl is None:
            raise RuntimeError("[ERROR] Attempting to use method 'cmdSimOnly' but Controller was created with 'use_firmware' = True.")

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

        # Store the last step's events.
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

    def _draw_trajectory(self,
                         initial_info
                         ):
        """Do not modify this.

        """
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
