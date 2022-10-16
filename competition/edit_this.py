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
        3) interStepLearn (optional)
        4) interEpisodeLearn (optional)

"""
from bdb import set_trace
from tkinter import Y
import numpy as np
import time
from collections import deque

try:
    from competition_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory
except ImportError:
    # PyTest import.
    from .competition_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory
from gym.spaces import Box

from safetyplusplus_folder.slam import SLAM
from safetyplusplus_folder.plus_logger import SafeLogger
import random

file_name='1016_02_L1_S9_KnowSelf_PongPenality5Co20'
# file_name='L0_test'
test=False
sim_only=False
model_name='models/1200'
#########################
# REPLACE THIS (START) ##
#########################

# Optionally, create and import modules you wrote.
# Please refrain from importing large or unstable 3rd party packages.
try:
    import example_custom_utils as ecu
except ImportError:
    # PyTest import.
    from . import example_custom_utils as ecu

#########################
# REPLACE THIS (END) ####
#########################

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
        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"] 
        
        # Check for pycffirmware.
        if use_firmware:
            self.ctrl = None
        else:
            # Initialize a simple PID Controller for debugging and test.
            # Do NOT use for the IROS 2022 competition. 
            self.ctrl = PIDController()
            # Save additonal environment parameters.
            self.KF = initial_info["quadrotor_kf"]



        #########################
        # REPLACE THIS (START) ##
        #########################
        import torch
        torch.manual_seed(101)
        torch.cuda.manual_seed(101)
        torch.cuda.manual_seed_all(101)
        np.random.seed(101)
        random.seed(101)
        self.begin_train_seconds=1.5
        self.begin_net_infer_epo=80
        self.begin_train_epo=100
        self.net_work_freq=0.5     #  time gap  1  1s/次  0.5s/次   0.2m  400episode 
        max_action=2
        self.global_state_dim = 9
        self.set_offset=False
        self.batch_size=128
        
        # state-based 
        self.mass=initial_info['nominal_physical_parameters']['quadrotor_mass']
        self.local_state_dim=[5,23,23]
        self.current_all_state=[np.zeros(self.global_state_dim),np.zeros(self.local_state_dim)]
        self.last_all_state=[np.zeros(self.global_state_dim),np.zeros(self.local_state_dim)]
        
        self.m_slam = SLAM()
        self.m_slam.reset_occ_map()

        # action    
        action_dim = 3
        min_action=max_action * (-1)
        self.action_space=Box(np.array([min_action,min_action,min_action],dtype=np.float64),np.array([max_action,max_action,max_action],dtype=np.float64))
        self.action_space.seed(1)
        self.current_action=np.zeros(action_dim)
        self.last_action=np.zeros(action_dim)
        # network
        kwargs = {
            "state_dim": self.global_state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "rew_discount": 0.99,
            "tau":0.005,
            "policy_noise": 0.2 * max_action,
            "noise_clip": 0.5 * max_action,
            "policy_freq": 2,
        }
        from safetyplusplus_folder import safetyplusplus,replay_buffer,unconstrained
        # td3 policy
        self.policy = unconstrained.TD3(**kwargs)
        self.replay_buffer = replay_buffer.IrosReplayBuffer(self.global_state_dim,self.local_state_dim,action_dim,load=False)
        if test:
            self.policy.load(model_name)
            print("load ok ")

        # env-based variable
        self.info=None
        self.cur2goal_dis=0
        self.episode_reward = 0
        self.episode_cost = 0
        self.collisions_count=self.violations_count=0
        self.episode_iteration=-1
        self.target_gate_id=0
        self.goal_pos=[initial_info['x_reference'][0],initial_info['x_reference'][2],initial_info['x_reference'][4]]
        self.target_offset=np.array([0,0,0])
        self.trigger=False
        self.get_offset(info=None)
        self.rule_control_time=0
        # Reset counters and buffers.
        self.reset()
        self.interEpisodeReset()

        # logger
        self.logger_plus = SafeLogger(exp_name=file_name, env_name="compitition", seed=0,
                                fieldnames=['Eptime','EpRet', 'EpCost', 'collision_num','vilation_num','target_gate','dis'])   
        #########################
        # REPLACE THIS (END) ####
        #########################
        

    def get_offset(self,info):
        # init
        if info is None:
             self.target_offset=np.array([0.05,0,0])
        # step cross gate
        elif info['current_target_gate_id'] == 1 or info['current_target_gate_id'] == 3:
            self.target_offset=np.array([0,0.05,0])
        elif info['current_target_gate_id'] == 2:
            self.target_offset=np.array([-0.05,0,0])
        else:
            self.target_offset=np.array([0,0,0])

    def get_state(self,obs,info):
        # state info :+ obs_info(3) + goal_info(3) + gate_in_range(1)+target_gate_id(1) + mass(1) + pic_info(16)  
        # x,y,z  3 
        current_pos=[obs[0],obs[2],obs[4]]

        if info !={}:
            # goal [x,y,z]
            if info['current_target_gate_id'] == -1 :
                current_goal_pos = self.goal_pos
                current_target_gate_in_range= 0 
            else :
                # 1 or 0  1
                current_target_gate_in_range= 1 if info['current_target_gate_in_range'] == True else 0
                # [x,y,z,r] exactly if current_target_gate_in_range==1 else [x,y,z,y] noisy  (r,p is zero ,ignored )
                current_target_gate_pos = info['current_target_gate_pos']
                if current_target_gate_pos[2]==0: #  means that  current_target_gate_in_range is False, add default height.
                    current_target_gate_pos[2]=1 if info['current_target_gate_type'] == 0 else 0.525
                current_target_gate_pos=np.array(current_target_gate_pos)[[0,1,2,5]]
                current_goal_pos=current_target_gate_pos[:3]
            if self.set_offset:
                current_goal_pos += self.target_offset
        else :
            current_target_gate_in_range= 0 
            current_goal_pos=np.zeros(3)
        target_vector=[current_goal_pos[0]- current_pos[0],current_goal_pos[1]- current_pos[1],current_goal_pos[2]- current_pos[2]]
        # 10.09 V0.1
        global_state=np.array([current_pos[0], current_pos[1], current_pos[2],target_vector[0],target_vector[1],target_vector[2],
                               current_target_gate_in_range,info['current_target_gate_id'],self.mass])
        # global_state=np.array([current_pos[0], current_pos[1], current_pos[2],target_vector[0],target_vector[1],target_vector[2],self.mass])    
        local_state = self.m_slam.generate_3obs_img(obs,target_vector,name=self.episode_iteration,save=False if self.episode_iteration % 20!=0 else True)   
        return [global_state,local_state]
           
    def cmdFirmware(self,ctime,obs,reward=None,done=None,info=None,exploration=True):
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
        self.episode_iteration+=1
       
        if self.ctrl is not None:
            raise RuntimeError("[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False.")

        #########################
        # REPLACE THIS (START) ##
        #########################
        self.m_slam.update_occ_map(info)
        
        
        # begin with take off 
        if self.episode_iteration == 0:
            height = 1
            duration = 1.5
            command_type = Command(2)  # Take-off.
            args = [height, duration]
        elif info!={} and info['current_target_gate_id'] == -1 and self.episode_iteration % 5 ==0:
            self.rule_control_time+=1
            # navigate to the goal_pos.and stop
            command_type =  Command(1) 
            if self.rule_control_time <=6: 
                target_pos = np.array(self.goal_pos)
                target_pos[1] -=0.3
            else:
                target_pos = np.array(self.goal_pos)
            target_vel = np.zeros(3)
            target_acc = np.zeros(3)
            target_yaw = 0.
            target_rpy_rates = np.zeros(3)
            args=[target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]

            if  self.episode_iteration % (30*self.net_work_freq) ==0:
                all_state=self.get_state(obs,info)
                global_state=all_state[0]
                action=target_pos-global_state[[0,1,2]]
                action /= 10
                
                self.current_all_state=all_state
                self.current_action=action
        # using network to choose action
        elif self.episode_iteration >= self.begin_train_seconds * self.CTRL_FREQ and self.episode_iteration % (30*self.net_work_freq) ==0 :
            # cmdFullState
            command_type =  Command(1)  
            all_state=self.get_state(obs,info)
            global_state=all_state[0]
            if not test and self.interepisode_counter < self.begin_net_infer_epo:
                action= self.action_space.sample() 
            else:
                action = self.policy.select_action(all_state, exploration=False if test else True)  # array  delta_x , delta_y, delta_z
            action /= 10
            target_pos = global_state[[0,1,2]] + action
            self.current_all_state=all_state
            self.current_action=action
            target_vel = np.zeros(3)
            target_acc = np.zeros(3)
            target_yaw = 0.
            target_rpy_rates = np.zeros(3)
            args=[target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]
        # other time do nothing
        else :
            command_type = Command(0)  # None.
            args = []

        #########################
        # REPLACE THIS (END) ####
        #########################
        
        return command_type, args

    # NOT need to re-implement
    def cmdSimOnly(self,time,obs,reward=None,done=None,info=None):
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
        self.episode_iteration+=1
        if self.ctrl is None:
            raise RuntimeError("[ERROR] Attempting to use method 'cmdSimOnly' but Controller was created with 'use_firmware' = True.")

        self.iteration = int(time*self.CTRL_FREQ)
        self.m_slam.update_occ_map(info)

        #########################
        self.interstep_counter += 1
        if self.episode_iteration <= 40:
            target_p = np.array([-0.9,-2.9,1])
        else :
            all_state=self.get_state(obs,info)
            global_state=all_state[0]
            if self.interepisode_counter < 30:
                action= self.action_space.sample() 
            else:
                action = self.policy.select_action(all_state, exploration=True)  # array  delta_x , delta_y, delta_z
            action /= 10
            self.current_all_state=all_state
            self.current_action=action
            target_p = global_state[[0,1,2]] + action

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
        reward=0
        #########################
        # REPLACE THIS (START) ##
        #########################
        if not sim_only:
            # add experience when use network to decide
            if  self.episode_iteration == self.begin_train_seconds * self.CTRL_FREQ:
                self.last_all_state=self.current_all_state
                self.last_action = self.current_action

            if  self.episode_iteration> self.begin_train_seconds * self.CTRL_FREQ   :
                if self.episode_iteration % (30*self.net_work_freq) ==0:
                    last_pos= self.last_all_state[0][[0,1,2]]
                    current_pos=self.current_all_state[0][[0,1,2]]
                    current_local_space=self.current_all_state[1][2-1:2+2,11-1:11+2,11-1:11+2]

                    last2goal_vector= self.last_all_state[0][[3,4,5]]
                    last2cur_vector=current_pos-last_pos
                    cur2goal_vector=self.current_all_state[0][[3,4,5]]
                    cur2goal_dis=sum(cur2goal_vector * cur2goal_vector)
                    last2goal_dis=sum(last2goal_vector * last2goal_vector)

                    if self.target_gate_id == info['current_target_gate_id']:
                        reward +=( last2goal_dis - cur2goal_dis ) * 20
                    # cross the gate ,get the big reward
                    else:
                        reward = reward + 100 * ( (info['current_target_gate_id'] if info['current_target_gate_id']!=-1 else 4) +1)
                        print(f"STEP{self.episode_iteration} , step gate{self.target_gate_id}")
                        if info['current_target_gate_id']==-1:
                            print("step all gates")
                            self.trigger=True
                        self.get_offset(info)
                    if info['at_goal_position']:
                        reward += 100
                    if (current_local_space<0).any():
                        reward -= 5    
                    if info['constraint_violation']:
                        reward -= 15
                    if info["collision"][1]:
                        reward -= 20
                        self.collisions_count += 1 
                        self.episode_cost+=1
                    if 'constraint_values' in info and info['constraint_violation'] == True:
                        self.violations_count += 1
                        self.episode_cost+=1
                    if info['constraint_violation'] and (  test or self.interepisode_counter> 100 ) :
                        print(f"step{self.episode_iteration} , constraint_violation : current pos : {current_pos}")
                    if info["collision"][1] and (  test or self.interepisode_counter> 100 ):
                        print(info["collision"])   
                    # cmdFullState
                    self.replay_buffer.add(self.last_all_state[0],self.last_all_state[1],self.last_action * 10 ,self.current_all_state[0],self.current_all_state[1],reward,done)
                    if self.episode_iteration % 900 ==0  and not test:
                        print(f"Step{self.episode_iteration} add buffer:\nlast_pos:{last_pos} aim vector: {last2goal_vector} ")
                        print(f"action_infer: {self.last_action * 10}\t lastDis-CurDis:{( last2goal_dis - cur2goal_dis )}")
                        print(f"last2cur_pos_vector: {last2cur_vector } \t reward: {reward}")
                        print(f"target_gate_id:{info['current_target_gate_id']} ; pos: {info['current_target_gate_pos']} ; distance : {cur2goal_dis}")
                        print("*************************************************************************************")
                    self.episode_reward+=reward

                    # ready for next update
                    self.last_all_state=self.current_all_state
                    self.last_action=self.current_action
                    self.target_gate_id= info['current_target_gate_id']
                    self.cur2goal_dis=cur2goal_dis
                    self.info=info
                    
                    
                else :
                    pass
            # change_Train_Num Better
            if self.interepisode_counter > self.begin_train_epo  and not test :
                # self.policy.train(self.replay_buffer,batch_size=min(128,64*int(self.interepisode_counter/100+1)),train_nums=int(1))   
                self.policy.train(self.replay_buffer,batch_size=self.batch_size,train_nums=int(1)) 
        #########################
        # REPLACE THIS (END) ####
        #########################

    @timing_ep
    def interEpisodeLearn(self):
        """Learning and controller updates called between episodes.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions, observations,
            rewards, done flags, and information dictionaries to learn, adapt, and/or re-plan.

        """
        self.interepisode_counter += 1
        
        #########################
        # REPLACE THIS (START) ##
        #########################

        if self.interepisode_counter % 300 == 0 or self.interepisode_counter==1000:
            self.policy.save(filename=f"{self.logger_plus.log_dir}/{self.interepisode_counter}")

        print(f"Episode Num: {self.interepisode_counter}  step Num : {self.episode_iteration} ,Reward: {self.episode_reward:.3f} Cost: {self.episode_cost:.3f} violation: {self.violations_count:.3f}  collision:{self.collisions_count:.3f} ,")
        print(f"gates_passed:{self.info['current_target_gate_id']},at_goal_position : {self.info['at_goal_position']}  task_completed: {self.info['task_completed']}")
        self.logger_plus.update([self.episode_reward, self.episode_cost,self.collisions_count,self.violations_count,self.info['current_target_gate_id'],self.cur2goal_dis], total_steps=self.interepisode_counter)

        #########################
        # REPLACE THIS (END) ####
        #########################

    def reset(self):
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
        self.episode_reward = 0
        self.episode_cost = 0
        self.collisions_count=self.violations_count=0
        self.episode_iteration=-1
        # env-based variable
        self.target_gate_id=0
        self.m_slam.reset_occ_map()
        self.current_all_state=[np.zeros(self.global_state_dim),np.zeros(self.local_state_dim)]
        self.last_all_state=[np.zeros(self.global_state_dim),np.zeros(self.local_state_dim)]
        self.target_offset=np.array([0,0,0])
        self.cur2goal_dis=0
        self.info=None
        self.trigger=False
        self.rule_control_time=0

