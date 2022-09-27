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

    They are in methods:
        1) __init__
        2) cmdFirmware
        3) interStepLearn (optional)
        4) interEpisodeLearn (optional)

"""
from bdb import set_trace
from tkinter import Y
import numpy as np

from collections import deque

try:
    from competition_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory
except ImportError:
    # Test import.
    from .competition_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory
from gym.spaces import Box

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
                'symbolic_model', 'nominal_physical_parameters', 'nominal_gates_pos_and_type', etc.
            use_firmware (bool, optional): Choice between the on-board controll in `pycffirmware`
                or simplified software-only alternative.
            buffer_size (int, optional): Size of the data buffers used in method `learn()`.
            verbose (bool, optional): Turn on and off additional printouts and plots.

        """
        # Save environment and conrol parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]


        # when level 1 this will be changed
        # [-0.9660992341982342, 0, -2.906546142854587, 0, 0.04960550660033104, 0, 0.02840173653255687, -0.05744735611733429, 0.07373743557600966, 0, 0, 0]
        # [-0.96609923  0.         -2.90654614  0.          0.04960551  0.        0.02840174           -0.05744736           0.07373744   0.      0.    0.        ]
        self.initial_obs = initial_obs

        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        # Store a priori scenario information.

        # when level 2 this will be changed
        # x y z r p y 
        # initial is [[0.5, -2.5, 0, 0, 0, -1.57, 0], [2, -1.5, 0, 0, 0, 0, 1], [0, 0.2, 0, 0, 0, 1.57, 1], [-0.5, 1.5, 0, 0, 0, 0, 0]]
        #  when target gate show in horizon , be like "Current target gate position: [0.6143489615604684, -2.634075935292277, 1.0, 0.0, 0.0, -1.6783756661588305]"
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        
        # when level 2 this will be changed
        # x y z r p y 
        # [[1.5, -2.5, 0, 0, 0, 0], [0.5, -1, 0, 0, 0, 0], [1.5, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0]]  
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"] 
        
        # Check for pycffirmware.
        if use_firmware:
            self.ctrl = None
        else:
            # Initialize a simple PID Controller ror debugging and test
            # Do NOT use for the IROS 2022 competition. 
            self.ctrl = PIDController()
            # Save additonal environment parameters.
            self.KF = initial_info["quadrotor_kf"]

        # Reset counters and buffers.
        self.reset()
        self.interEpisodeReset()

        #########################
        # REPLACE THIS (START) ##
        #########################

        self.net_work_freq=0.5   #  time gap
        self.state=np.zeros(7)
        state_dim = 7
        action_dim = 3
        max_action = 1.
        min_action = -1
        self.action_space=Box(np.array([-1,-1,-1],dtype=np.float64),np.array([1,1,1],dtype=np.float64))
        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "rew_discount": 0.99,
            "tau":0.005,
            "policy_noise": 0.2 * max_action,
            "noise_clip": 0.5 * max_action,
            "policy_freq": 2,
        }

        kwargs_safe = {
            "cost_discount": 0.99,
            "delta": 0.1,                     
        }
        from safetyplusplus_folder import safetyplusplus,replay_buffer,unconstrained

        # td3 policy
        self.policy = unconstrained.TD3(**kwargs)
        # self.policy.load("td3_2")
        self.replay_buffer = replay_buffer.SimpleReplayBuffer(state_dim, action_dim)

        # kwargs.update(kwargs_safe)
        # kwargs.update({'kappa':5})
        # self.policy = safetyplusplus.TD3Usl(**kwargs)
        # replay_buffer = replay_buffer.CostReplayBuffer(state_dim, action_dim)
        self.episode_reward = 0
        self.episode_cost = 0
        self.episode_timesteps = 0
        self.episode_num = 0
        self.cost_total = 0
        self.prev_cost = 0
        self.episode_iteration=-1

        self.pass_time = 1e6

        # env-based variable
        self.go_back=False
        self.pass_bool=False
        self.agent_type='passing'
        self.arrival_iteration =1e6
        self.goal_pos=[initial_info['x_reference'][0],initial_info['x_reference'][2],initial_info['x_reference'][4]]

        #########################
        # REPLACE THIS (END) ####
        #########################
    def get_state(self,obs,info):
        # state info : obs_info(3) + goal_info(3) + all_gate_info(1 + 16) + all_obstacle_info(12)     = 35
        # x,y,z  3 
        current_x=obs[0]
        current_y=obs[2]
        current_z=obs[4]

        # import pdb;pdb.set_trace()
        # [[x,y,z,(rpy all zero)]] 
        # 3 * obstacle num(4)
        #      [1.5, -2.5, 0, 0, 0, 0],[0.5, -1, 0, 0, 0, 0],[1.5, 0, 0, 0, 0, 0],[-1, 0, 0, 0, 0, 0]
        # ->   [1.5, -2.5, 1.05, 0.5, -1, 1.05,1.5, 0, 1.05, -1, 0, 1.05]
        all_obstacles_pos=np.array(self.NOMINAL_OBSTACLES)[:,0:3]
        for one_obstacle_info in all_obstacles_pos:
                one_obstacle_info[2] = 1.05  # quadrotor.py reset()
        all_obstacles_pos=all_obstacles_pos.flatten()

        #   [0.5, -2.5, 0, 0, 0, -1.57, 0],[2, -1.5, 0, 0, 0, 0, 1],[0, 0.2, 0, 0, 0, 1.57, 1],[-0.5, 1.5, 0, 0, 0, 0, 0]
        #-> [0.5,-2.5,1,-1.57 ,  2,-1.5,0.525,0  , 0,0.2,0.525,1.57,   -0.5,1.5,1,0 ]
        all_gates_pos=np.array(self.NOMINAL_GATES)
        for one_gete_info in all_gates_pos:
            one_gete_info[2]=1 if one_gete_info[6] == 0 else 0.525
        all_gates_pos=all_gates_pos[:,[0,1,2,5]].flatten()

        if info !={}:
            # 1 or 0  1
            current_target_gate_in_range= 1 if info['current_target_gate_in_range'] == True else 0

            # [x,y,z,r] exactly if current_target_gate_in_range==1 else [x,y,z,y] noisy  (r,p is zero ,ignored )
            current_target_gate_pos = info['current_target_gate_pos']
            if current_target_gate_pos[2]==0: #  means that  current_target_gate_in_range is False, add default height.
                current_target_gate_pos[2]=1 if info['current_target_gate_type'] == 0 else 0.525
            current_target_gate_pos=np.array(current_target_gate_pos)[[0,1,2,5]]

            # goal [x,y,z]
            if info['current_target_gate_id'] == -1 :
                current_goal_pos = self.goal_pos
            else :
                # TODO using the center of the gate.
                current_goal_pos=current_target_gate_pos[:3]
        else :
            current_target_gate_in_range= 0 
            current_target_gate_pos = np.zeros(4)
            current_goal_pos=np.zeros(3)
        state=np.array([current_x,current_y,current_z,current_goal_pos[0],current_goal_pos[1],current_goal_pos[2],current_target_gate_in_range])
        # state=np.append(state,all_obstacles_pos)
        # state=np.append(state,all_gates_pos)
        # state=np.append(state,current_target_gate_pos)
        return state
            
    def cmdFirmware(self,time,obs,reward=None,done=None,info=None,):
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this method to return the target position, velocity, acceleration, attitude, and attitude rates to be sent
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
        prt=False
        self.episode_iteration+=1
       
        if self.ctrl is not None:
            raise RuntimeError("[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False.")

        #########################
        # REPLACE THIS (START) ##
        #########################


        # whether goback
        if info != {} and info['current_target_gate_id'] == -1 and self.pass_bool==False:
            self.pass_time=self.episode_iteration
            self.pass_bool=True
            print(f"pass_all_gate_time : {self.pass_time}")
        if self.episode_iteration == self.pass_time + 2:
            self.agent_type='just_pass'
            self.go_back=True

        # begin with take off 
        if self.episode_iteration == 0:
            height = 1
            duration = 1.5
            command_type = Command(2)  # Take-off.
            args = [height, duration]
        # end with rule-based when have passed all the gate 
        elif self.go_back  :
            # up
            if self.agent_type=='just_pass':
                print("Just passed,going up")
                duration = 2
                self.arrival_iteration=self.episode_iteration+ duration *self.CTRL_FREQ
                command_type = Command(5)  # goTo.
                args = [[0, 0, 1], 0, duration, True]
                self.agent_type='going up'
            # go to goal place
            elif self.agent_type=='going up' and self.episode_iteration==self.arrival_iteration + 1 :
                print("going goal place")
                duration = 3
                self.arrival_iteration=self.episode_iteration+ duration *self.CTRL_FREQ
                command_type = Command(5)  # goTo.
                args = [[self.goal_pos[0], self.goal_pos[1], self.goal_pos[2]], 0, duration, False]
                self.agent_type='going goal place'

            #  arrival for 2s  or task_completed
            elif self.agent_type=='going goal place' and (self.episode_iteration == self.arrival_iteration + 2 * self.CTRL_FREQ + 1 or info['task_completed'] ==True):
                print("landing")
                height = 0.
                duration = 2
                self.arrival_iteration=self.episode_iteration+ duration *self.CTRL_FREQ
                command_type = Command(3)  # Land.
                args = [height, duration]
                self.agent_type='landing'

            # after land , exit
            elif self.agent_type=='landing' and self.episode_iteration==self.arrival_iteration + 1:
                command_type = Command(-1)  # Terminate command to be sent once trajectory is completed.
                args = []
            else :
                command_type = Command(0)  # None.
                args = []
        
        # using network to choose action
        elif self.episode_iteration >= 2 * self.CTRL_FREQ :
            
            if self.episode_iteration % (30*self.net_work_freq) ==0:
                command_type =  Command(1)  # "network"  # cmdFullState.
                if self.interepisode_counter <= 10:
                    self.state=self.get_state(obs,info)
                    action=self.action_space.sample() 
                else:
                    self.state=self.get_state(obs,info)
                    action = self.policy.select_action(self.state, exploration=True)  # array  delta_x , delta_y, delta_z
                action /= 10
                target_pos=self.state[[0,1,2]] + action
                target_vel = np.zeros(3)
                target_acc = np.zeros(3)
                target_yaw = 0.
                target_rpy_rates = np.zeros(3)
                args=[target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]
            # other time do nothing
            else :
                command_type = Command(0)  # None.
                args = []
        
        else :
            command_type = Command(0)  # None.
            args = []


        """ test 
        #----------------test action---------------------
        # if self.iteration == 0:
        #     z = 1
        #     yaw = 0.
        #     # duration 会影响时间
        #     duration = 2
        #     command_type = Command(5)  # goTo.
        #     args = [[0, 0, z], yaw, duration, True]

        # elif self.iteration > 0.5 *self.CTRL_FREQ and self.iteration < 1.5 *self.CTRL_FREQ :
        #     command_type = Command(4)  # Stop.
        #     args = []
        # else:
        #     command_type = Command(0)  # None.
        #     args = []
        #----------------test action---------------------
        """

        """ test  go to action
        #------------------------------------- 
        # go to not very exactly(exist error).  if not random , always this (getting_started.yaml)
        #         x                                   y                                z
        # even if send  command "only change y " from 60steps (2*CTRL_FREQ)  z will continue increase  ,meybe the problem about the Asynchronous 
        # init:  [-0.96609923  0.                  -2.90654614  0.                   0.04960551  0. 0.02840174 -0.05744736  0.07373744  0.          0.          0.        ]
        # 0-- [-0.96661885 -0.0279078           -2.90698836 -0.00873109            0.04799388 -0.05964756 0.02340614 -0.04682872  0.07295734 -0.60471722  1.66696169 -0.06040739]
        # 15- [-0.98957009  0.0177486           -2.91855744  0.01612694            0.08783306  0.40195587 -0.01363278  0.039587    0.07313598  0.1224719  -0.18560217 -0.0133112 ]
        # 30- [-9.74126116e-01 -6.12250160e-03  -2.90933867e+00  2.79855108e-02    5.77164080e-01  1.57968746e+00  1.13573342e-03  2.79637762e-04 7.34927365e-02 -3.14627467e-02  1.04491797e-01 -3.82468499e-05]
        # 45- [-0.96690019  0.02592517          -2.9244567  -0.04027908            1.58570938  2.21925943 -0.04590852  0.00406897  0.07356948 -0.20060068 -0.07108402 -0.00401296]
        # 60- [-0.97490167 -0.00690929          -2.90244781  0.03042487            2.5512753   1.51622867 0.03011076  0.02357646  0.07466138 -0.6881512  -0.51120803  0.0308384 ]
        # 75- [-0.95789148  0.04922068          -2.84188733  0.34668242            3.25083697  1.17400743 -0.23051172 -0.00670337  0.07419264 -1.46396087  0.74862298  0.10745695]
        # 90- [-0.96644482  0.03573701          -2.35380695  1.60398413            3.55250003 -0.0314102 -0.30132277  0.04541282  0.06064814 -0.13881515  1.02686106 -0.1751778 ]
        # 105-[-0.96541722  0.00741155          -1.35492773  2.16873779            3.27599927 -0.96907974 -0.02264677  0.00403805  0.07491015  0.86021552  1.42216285  0.04174302]
        # 120-[-9.69886183e-01  3.45681240e-02  -3.70208598e-01  1.58408557e+00    2.75920080e+00 -8.91757966e-01  1.95222977e-01  6.89915684e-04 7.17125806e-02 -2.54350558e-01  9.00435832e-01  2.10910125e-01]
        # 135-[-0.93853971  0.14409335          0.299702    1.2196867              2.35446502 -0.66440774 0.08730616  0.0404132   0.0763123   1.39507016 -0.06357986 -0.14670249]
        # if self.iteration == 0:
        #     z = 3
        #     yaw = 0.
        #     # duration 会影响时间
        #     duration = 3
        #     command_type = Command(5)  # goTo.
        #     args = [[0, 0, z], yaw, duration, True]
        # elif self.iteration == 2  *self.CTRL_FREQ:
        #     y = 3
        #     yaw = 0.
        #     # duration 会影响时间
        #     duration = 3
        #     command_type = Command(5)  # goTo.
        #     args = [[0, y, 0], yaw, duration, True]
        # elif self.iteration == 4  *self.CTRL_FREQ  + 1:
        #     x = 1
        #     yaw = 0.
        #     # duration 会影响时间
        #     duration = 3
        #     command_type = Command(5)  # goTo.
        #     args = [[x, 0, 0], yaw, duration, True]
        # else:
        #     command_type = Command(0)  # None.
        #     args = []
        #----------------test action end ---------------------
        """

        """test  go to  and stop action
        #----------------test  go to  and stop action---------------------
        # result :  stop can break the commmand but have to observe the rule about the intertia.
        # if self.iteration == 0:
        #     z = 1
        #     yaw = 0.
        #     # duration 会影响时间
        #     duration = 0.2
        #     command_type = Command(5)  # goTo.
        #     args = [[0, 0, z], yaw, duration, True]
        # elif self.iteration == 2  *self.CTRL_FREQ :
        #     command_type = Command(4)  # Stop.  
        #     args = []
        # elif self.iteration == 3  *self.CTRL_FREQ:
        #     y = -3
        #     yaw = 0.
        #     # duration 会影响时间
        #     duration = 5
        #     command_type = Command(5)  # goTo.
        #     args = [[0, y, 0], yaw, duration, True]
        # elif self.iteration == 5  *self.CTRL_FREQ  :
        #     command_type = Command(4)  # Stop.
        #     args = []
        # elif self.iteration == 6  *self.CTRL_FREQ  :
        #     x = -1
        #     yaw = 0.
        #     # duration 会影响时间
        #     duration = 5
        #     command_type = Command(5)  # goTo.
        #     args = [[x, 0, 0], yaw, duration, True]
        # else:
        #     command_type = Command(0)  # None.
        #     args = []
        #----------------test action end ---------------------
        """

        """test the min time and the max distance about the go to  action
        #----------------test   the min time and the max distance about the go to  action---------------------
        # result :  
        if self.iteration == 0:
            z = 0.15
            # duration 会影响时间
            duration = 0.33
            command_type = Command(5)  # goTo.
            args = [[0, 0, z], 0, duration, True]
        else:
            command_type = Command(0)  # None.
            args = []
        #----------------test action end ---------------------
        """

        """ tst the cmdFullState command  the next command will break the last command
        x=obs[0]
        y=obs[2]
        z=obs[4]
        if self.iteration == 0:
            height = 1
            duration = 2
            command_type = Command(2)  # Take-off.
            args = [height, duration]
        # 90-[-9.75560513e-01  3.98563938e-02 -2.90579683e+00 -2.46277672e-03 9.46152214e-01 -4.69072173e-02  5.22578314e-03  3.41655474e-02 5.11630895e-02 -2.64937344e-01  1.81993538e+00 -7.07496076e-01]
        elif self.iteration == 3 *self.CTRL_FREQ:
            target_pos = np.array([x+0.1,y,z])
            target_vel = np.zeros(3)
            target_acc = np.zeros(3)
            target_yaw = 0.
            target_rpy_rates = np.zeros(3)

            command_type = Command(1)  # cmdFullState.
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]
        
        # 105-[-9.09596873e-01  2.00324028e-01 -2.90993115e+00 -3.19762174e-02 9.08722645e-01 -6.57070694e-02 -4.07341352e-02 -3.57578141e-02 2.78315544e-03 -4.06931208e-01  1.00299233e+00  2.78734319e-01]
        elif self.iteration == 3.5 *self.CTRL_FREQ:
            target_pos = np.array([x, y+0.1, z])
            target_vel = np.zeros(3)
            target_acc = np.zeros(3)
            target_yaw = 0.
            target_rpy_rates = np.zeros(3)

            command_type = Command(1)  # cmdFullState.
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]
       
        # 120-[-9.09082306e-01 -8.87231485e-02 -2.84472628e+00  2.40919527e-01 8.64559996e-01  1.77237485e-02  3.32461760e-02  2.48394834e-02 2.12280677e-03  9.54833499e-01  1.31716466e+00  1.11926138e-01]
        elif self.iteration == 4 *self.CTRL_FREQ:
            target_pos = np.array([x, y, z+0.1])
            target_vel = np.zeros(3)
            target_acc = np.zeros(3)
            target_yaw = 0.
            target_rpy_rates = np.zeros(3)

            command_type = Command(1)  # cmdFullState.
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]
        else :
            command_type = Command(0)  # None.
            args = []
        #  135-[-0.90734541  0.07334976 -2.83234816 -0.10398782  0.90615406  0.03580154 -0.0083358  -0.00963472  0.00332174  0.67757902  0.5731973   0.09230706]
        """

        """
        # Example action : Handwritten solution for GitHub's getting_stated scenario.
        # # default action
        # if self.iteration == 0:
        #     height = 1
        #     duration = 2
        #     command_type = Command(2)  # Take-off.
        #     args = [height, duration]

        # elif self.iteration >= 3*self.CTRL_FREQ and self.iteration < 20*self.CTRL_FREQ:
        #     step = min(self.iteration-3*self.CTRL_FREQ, len(self.ref_x) -1)
        #     target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
        #     target_vel = np.zeros(3)
        #     target_acc = np.zeros(3)
        #     target_yaw = 0.
        #     target_rpy_rates = np.zeros(3)

        #     command_type = Command(1)  # cmdFullState.
        #     args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]

        # elif self.iteration == 20*self.CTRL_FREQ:
        #     command_type = Command(6)  # notify setpoint stop.
        #     args = []

        # elif self.iteration == 20*self.CTRL_FREQ+1:
        #     x = self.ref_x[-1]
        #     y = self.ref_y[-1]
        #     z = 1.5 
        #     yaw = 0.
        #     duration = 2.5

        #     command_type = Command(5)  # goTo.
        #     args = [[x, y, z], yaw, duration, False]

        # elif self.iteration == 23*self.CTRL_FREQ:
        #     x = self.initial_obs[0]
        #     y = self.initial_obs[2]
        #     z = 1.5
        #     yaw = 0.
        #     duration = 6

        #     command_type = Command(5)  # goTo.
        #     args = [[x, y, z], yaw, duration, False]

        # elif self.iteration == 30*self.CTRL_FREQ:
        #     height = 0.
        #     duration = 3

        #     command_type = Command(3)  # Land.
        #     args = [height, duration]

        # elif self.iteration == 33*self.CTRL_FREQ-1:
        #     command_type = Command(-1)  # Terminate command to be sent once trajectory is completed.
        #     args = []

        # else:
        #     command_type = Command(0)  # None.
        #     args = []
        """

        #########################
        # REPLACE THIS (END) ####
        #########################
        
        return command_type, args

    # NOT need to re-implement
    def cmdSimOnly(self,
                   time,
                   obs,
                   reward=None,
                   done=None,
                   info=None
                   ):
        """PID per-propeller thrusts with a simplified, software-only PID quadrotor controller.

        INSTRUCTIONS:
            You do NOT need to re-implement this method for the IROS 2022 Safe Robot Learning competition.
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
            raise RuntimeError("[ERROR] Attempting to use method 'cmdSimOnly' but Controller was created with 'use_firmware' = True.")

        self.iteration = int(time*self.CTRL_FREQ)

        #########################
        if self.iteration < len(self.ref_x):
            target_p = np.array([self.ref_x[self.iteration], self.ref_y[self.iteration], self.ref_z[self.iteration]])
        else:
            target_p = np.array([self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]])
        target_v = np.zeros(3)
        #########################

        return target_p, target_v

    @timing_step
    def interStepLearn(self,args,obs,reward,done,info):
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
        self.interstep_counter += 1

        
        #########################
        # REPLACE THIS (START) ##
        #########################


        # add experience when use network to decide
        if  self.episode_iteration> 2 * self.CTRL_FREQ  and (not self.go_back)  and self.episode_iteration % (30*self.net_work_freq) ==0:
            # 
            # Store the last step's events.
            target_pos=args[0]
            action=(target_pos-self.state[[0,1,2]]) * 10
            next_state=self.get_state(obs,info)
            # import pdb;pdb.set_trace()
            #  ok  buffer no problem
            if self.episode_iteration % 600 ==0  :
                print(f"add buffer:\nstate: {[float('{:.4f}'.format(i)) for i in self.state]}  \t\naction: {action} \t\nnext_state: {[float('{:.4f}'.format(i)) for i in next_state]} reward: {reward}")
                print("*********************************************")
            self.replay_buffer.add(self.state,action,next_state,reward,done)

            if self.interepisode_counter >= 20:
                self.policy.train(self.replay_buffer,batch_size=256,train_nums=int(1))
        #########################
        # REPLACE THIS (END) ####
        #########################

    @timing_ep
    def interEpisodeLearn(self,file_name):
        """Learning and controller updates called between episodes.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions, observations,
            rewards, done flags, and information dictionaries to learn, adapt, and/or re-plan.

        """
        self.interepisode_counter += 1
        
        #########################
        # REPLACE THIS (START) ##
        #########################

        # _ = self.action_buffer
        # _ = self.obs_buffer
        # _ = self.reward_buffer
        # _ = self.done_buffer
        # _ = self.info_buffer

        # if self.interepisode_counter >= 20 :
        #     self.policy.train(self.replay_buffer,batch_size=256,train_nums=(30 /self.net_work_freq ))

        if self.interepisode_counter >= 10 and self.interepisode_counter % 10 == 0 :
            self.policy.save(filename=file_name)
        #########################
        # REPLACE THIS (END) ####
        #########################

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

    def interEpisodeReset(self):
        """Initialize/reset learning timing variables.

        Called between episodes in `getting_started.py`.

        """
        # Timing stats variables.
        self.interstep_learning_time = 0
        self.interstep_learning_occurrences = 0
        self.interepisode_learning_time = 0
        self.episode_iteration = -1
