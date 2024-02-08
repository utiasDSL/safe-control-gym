"""Write your proposed algorithm.
[NOTE]: The idea for the final project is to plan the trajectory based on a sequence of gates 
while considering the uncertainty of the obstacles. The students should show that the proposed 
algorithm is able to safely navigate a quadrotor to complete the task in both simulation and
real-world experiments.

Then run:

    $ python3 final_project.py --overrides ./getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS` and `REPLACE THIS (START)` in this file.

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
        1) planning
        2) cmdFirmware

"""
import numpy as np
import math
import pybullet as p

from collections import deque

try:
    from geo_controller import GeoController
except ImportError:
    from .geo_controller import GeoController

class Controller():
    """Template controller class.

    """

    def __init__(self,
                 circle_radius,
                 initial_obs,
                 initial_info,
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
                'symbolic_model', 'nominal_physical_parameters', 'nominal_gates_pos_and_type', etc.
            use_firmware (bool, optional): Choice between the on-board controll in `pycffirmware`
                or simplified software-only alternative.
            buffer_size (int, optional): Size of the data buffers used in method `learn()`.
            verbose (bool, optional): Turn on and off additional printouts and plots.

        """
        # [initial_info["x_reference"][0], initial_info["x_reference"][2], initial_info["x_reference"][4]]
        self.tolerance =  initial_info["tracking_tolerance"]
        self.radius = circle_radius
        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        # Store a priori scenario information.
        # plan the trajectory based on the information of the (1) gates and (2) obstacles. 
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        # Initialize a geometric tracking controller.
        self.ctrl = GeoController()

        # Save additonal environment parameters.
        self.KF = initial_info["quadrotor_kf"]

        # Reset counters and buffers.
        self.reset()
        self.interEpisodeReset()

        # perform trajectory planning
        self.planning(initial_info)

    def planning(self, initial_info):
        # Draw the boundary circle
        self.draw_circle(initial_info, self.radius - self.tolerance)
        self.draw_circle(initial_info, self.radius + self.tolerance)

    def draw_circle(self, initial_info, radius):
        init_pos = np.array([self.initial_obs[0], self.initial_obs[2], self.initial_obs[4]])
        bias_pos = np.array([self.radius, 0.0, 0.0])

        # self.omega = 2 * np.pi / self.duration
        duration = 4
        omega = 2 * np.pi / duration

        dt = 0.01
        nsample = int(duration / dt)
        time = dt * np.arange(nsample)
        ref_x = []
        ref_y = []
        ref_z = []
        for i in range(nsample):
            t = time[i]
            ref_x.append((radius) * math.cos(omega * t) - bias_pos[0] + init_pos[0])
            ref_y.append((radius) * math.sin(omega * t) - bias_pos[1] + init_pos[1])
            ref_z.append( - bias_pos[2] + init_pos[2])
        self.draw_trajectory(initial_info, np.array(ref_x), np.array(ref_y), np.array(ref_z))   

    def draw_trajectory(self, initial_info,
                        ref_x,
                        ref_y,
                        ref_z
                        ):
        """Draw a trajectory in PyBullet's GUI.

        """

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

    def computeAction(self, obs,target_p,target_v,target_a):
      """Compute the rotor speed using the geometric controller
        Args:
            control_timestep (float): The time step at which control is computed.
            cur_pos (ndarray): (3,1)-shaped array of floats containing the current position.
            cur_quat (ndarray): (4,1)-shaped array of floats containing the current orientation as a quaternion.
            cur_vel (ndarray): (3,1)-shaped array of floats containing the current velocity.
            cur_ang_vel (ndarray): (3,1)-shaped array of floats containing the current angular velocity.
            target_pos (ndarray): (3,1)-shaped array of floats containing the desired position.
            target_vel (ndarray): (3,1)-shaped array of floats containing the desired velocity.
            target_acc (ndarray): (3,1)-shaped array of floats containing the desired acceleration.
      """
      rpms, _, _ = self.ctrl.compute_control( self.CTRL_TIMESTEP,
                                              cur_pos=np.array([obs[0],obs[2],obs[4]]),
                                              cur_quat=np.array(p.getQuaternionFromEuler([obs[6],obs[7],obs[8]])),
                                              cur_vel=np.array([obs[1],obs[3],obs[5]]),
                                              cur_ang_vel=np.array([obs[9],obs[10],obs[11]]),
                                              target_pos=target_p,
                                              target_vel=target_v,
                                              target_acc=target_a
                                              )
      return self.KF * rpms**2

    def getRef(self,
              time,
              obs,
              reward=None,
              done=None,
              info=None
              ):
        """PID per-propeller thrusts with a simplified, software-only PID quadrotor controller.

        INSTRUCTIONS:
            You do NOT need to re-implement this method for the project.
            Only re-implement this method when `use_firmware` == False to return the target position and velocity.

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
            raise RuntimeError("[ERROR] Attempting to use method 'getRef' but Controller was created with 'use_firmware' = True.")

        # Get the desired speed 
        self.desired_speed = self.initial_obs[3]
        # Get the desired angular velocity
        self.omega = self.desired_speed / self.radius
        # Get the duration of completing one lap
        self.duration = 2 * np.pi / self.omega
        # Set the centre of the circle
        init_pos = np.array([self.initial_obs[0], self.initial_obs[2], self.initial_obs[4]])
        bias_pos = np.array([self.radius, 0.0, 0.0])

        omega = self.omega
        omega2 = omega * omega
        # Compute the reference pos, vel, and acc on the circel
        ref_pos = np.array([self.radius * math.cos(omega * time), self.radius * math.sin(omega * time), 0.0]) - bias_pos + init_pos
        ref_vel = np.array([-self.radius * omega * math.sin(omega * time), self.radius * omega * math.cos(omega * time), 0.0])
        ref_acc = np.array([-self.radius * omega2 * math.cos(omega * time), -self.radius * omega2 * math.sin(omega * time), 0.0])

        target_p = ref_pos
        target_v = ref_vel
        target_a = ref_acc

        return target_p, target_v, target_a

    def reset(self):
        """Initialize/reset data buffers and counters.

        Called once in __init__().

        """
        # Data buffers.
        self.action_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.obs_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.reward_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.done_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.info_buffer = deque([], maxlen=self.BUFFER_SIZE)

        # Counters.
        self.interstep_counter = 0
        self.interepisode_counter = 0

    # NOTE: this function is not used in the course project. 
    def interEpisodeReset(self):
        """Initialize/reset learning timing variables.

        Called between episodes in `getting_started.py`.

        """
        # Timing stats variables.
        self.interstep_learning_time = 0
        self.interstep_learning_occurrences = 0
        self.interepisode_learning_time = 0
