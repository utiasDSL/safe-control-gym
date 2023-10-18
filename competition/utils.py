import numpy as np 



def get_state(obs,info,NOMINAL_OBSTACLES,NOMINAL_GATES):
    # state info : obs_info(3) + goal_info(3) + all_gate_info(1 + 16) + all_obstacle_info(12)     = 35
    # x,y,z  3 
    current_x=obs[0]
    current_y=obs[2]
    current_z=obs[4]

    # obstacle info
    # all_obstacles_pos=np.array(NOMINAL_OBSTACLES)[:,0:3]
    # for one_obstacle_info in all_obstacles_pos:
    #         one_obstacle_info[2] = 1.05  # quadrotor.py reset()
    
    # agent_pos = np.array([current_x, current_y, current_z])
    # dis = [sum((i - agent_pos) ** 2) for i in all_obstacles_pos]
    # visual_index = [i for i in range(len(dis)) if dis[i] < 0.75]
    # visual_obstacles = obstacles[visual_index]

    # obstacles_state = np.zeros(8)
    # x_agent = agent_pos[0]
    # y_agent = agent_pos[1]
    # for obstacle in visual_obstacles:
    #     x = obstacle[0]
    #     y = obstacle[1]
    #     delta_x = x - x_agent
    #     delta_y = y - y_agent
    #     if delta_y * delta_x == 0:
    #         if delta_y == 0:
    #             if delta_x >= 0:
    #                 obstacles_state[2] = 1
    #             else:
    #                 obstacles_state[6] = 1
    #         elif delta_x == 0:
    #             if delta_y >= 0:
    #                 obstacles_state[0] = 1
    #             else:
    #                 obstacles_state[4] = 1
    #     else:
    #         if abs(delta_x) / abs(delta_y) >= 1.71:  # gen hao 3
    #             if delta_x > 0:
    #                 obstacles_state[2] = 1
    #             else:
    #                 obstacles_state[6] = 1
    #         elif abs(delta_x) / abs(delta_y) <= 1 / 1.71:  # gen hao 3
    #             if delta_y > 0:
    #                 obstacles_state[0] = 1
    #             else:
    #                 obstacles_state[4] = 1
    #         else:
    #             if delta_y * delta_x > 0:
    #                 if delta_y > 0 and delta_x > 0:
    #                     obstacles_state[1] = 1
    #                 elif delta_y < 0 and delta_x < 0:
    #                     obstacles_state[5] = 1
    #             else:
    #                 if delta_y > 0 and delta_x < 0:
    #                     obstacles_state[7] = 1
    #                 elif delta_y < 0 and delta_x > 0:
    #                     obstacles_state[3] = 1


    #   [0.5, -2.5, 0, 0, 0, -1.57, 0],[2, -1.5, 0, 0, 0, 0, 1],[0, 0.2, 0, 0, 0, 1.57, 1],[-0.5, 1.5, 0, 0, 0, 0, 0]
    #-> [0.5,-2.5,1,-1.57 ,  2,-1.5,0.525,0  , 0,0.2,0.525,1.57,   -0.5,1.5,1,0 ]
    # all_gates_pos=np.array(NOMINAL_GATES)
    # for one_gate_info in all_gates_pos:
    #     one_gate_info[2]=1 if one_gate_info[6] == 0 else 0.525
    # all_gates_pos=all_gates_pos[:,[0,1,2,5]].flatten()

    if info !={}:
        # goal [x,y,z]
        if info['current_target_gate_id'] == -1 :
            current_goal_pos = goal_pos
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
