import numpy as np 



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
                
    else :
        current_target_gate_in_range= 0 
        current_target_gate_pos = np.zeros(4)
        current_goal_pos=np.zeros(3)
    state=np.array([current_x,current_y,current_z,current_goal_pos[0]-current_x,current_goal_pos[1]-current_y,current_goal_pos[2]-current_z,current_target_gate_in_range,info['current_target_gate_id']])
    # state=np.append(state,all_obstacles_pos)
    # state=np.append(state,all_gates_pos)
    # state=np.append(state,current_target_gate_pos)
    return state
