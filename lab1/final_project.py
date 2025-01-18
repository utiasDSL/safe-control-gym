"""Base script.

Run as:

    $ python3 final_project.py --overrides ./getting_started.yaml

Look for instructions in `README.md` and `edit_this.py`.

"""
import time
import inspect
import numpy as np
import pybullet as p

from functools import partial
from rich.tree import Tree
from rich import print

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import sync

try:
    from project_utils import Command, thrusts
    from edit_this import Controller
except ImportError:
    # Test import.
    from .project_utils import Command, thrusts
    from .edit_this import Controller

try:
    import pycffirmware
except ImportError:
    FIRMWARE_INSTALLED = False
else:
    FIRMWARE_INSTALLED = True
finally:
    print("Module 'cffirmware' available:", FIRMWARE_INSTALLED)


def run(test=False):
    """The main function creating, running, and closing an environment.

    """

    # Start a timer.
    START = time.time()

    # Load configuration.
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()

    # Testing (without pycffirmware).
    if test:
        config['use_firmware'] = False
        config['verbose'] = False
        config.quadrotor_config['ctrl_freq'] = 60
        config.quadrotor_config['pyb_freq'] = 240
        config.quadrotor_config['gui'] = False

    # Check firmware configuration.
    if config.use_firmware and not FIRMWARE_INSTALLED:
        raise RuntimeError("[ERROR] Module 'cffirmware' not installed.")
    CTRL_FREQ = config.quadrotor_config['ctrl_freq']
    CTRL_DT = 1/CTRL_FREQ

    # Create environment.
    # [INSTRUCTIONS:] 
    # When the simulation env. reset, we can get the current observation and information 
    # of the simulation env.
    if config.use_firmware:
        FIRMWARE_FREQ = 500
        assert(config.quadrotor_config['pyb_freq'] % FIRMWARE_FREQ == 0), "pyb_freq must be a multiple of firmware freq"
        # The env.step is called at a firmware_freq rate, but this is not as intuitive to the end user, and so 
        # we abstract the difference. This allows ctrl_freq to be the rate at which the user sends ctrl signals, 
        # not the firmware. 
        config.quadrotor_config['ctrl_freq'] = FIRMWARE_FREQ
        env_func = partial(make, 'quadrotor', **config.quadrotor_config)
        firmware_wrapper = make('firmware',
                    env_func, FIRMWARE_FREQ, CTRL_FREQ
                    ) 
        obs, info = firmware_wrapper.reset()      
        info['ctrl_timestep'] = CTRL_DT
        info['ctrl_freq'] = CTRL_FREQ
        env = firmware_wrapper.env
    else:
        env = make('quadrotor', **config.quadrotor_config)
        # Reset the environment, obtain the initial observations and info dictionary.
        obs, info = env.reset()
    
    # Create controller.
    # [INSTRUCTIONS:] 
    # vicon_obs indicates the initial observation (initial state) from Vicon.
    # obs = {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r}.
    # vicon_obs = {x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0}.
    vicon_obs = [obs[0], 0, obs[2], 0, obs[4], 0, obs[6], obs[7], obs[8], 0, 0, 0]

    # NOTE: students can get access to the information of the gates and obstacles 
    #       when creating the controller object. 
    ctrl = Controller(vicon_obs, info, config.use_firmware, verbose=config.verbose)

    # Create counters
    episodes_count = 1
    cumulative_reward = 0
    collisions_count = 0
    collided_objects = set()
    violations_count = 0
    episode_start_iter = 0
    time_label_id = p.addUserDebugText("", textPosition=[0, 0, 1],physicsClientId=env.PYB_CLIENT)
    num_of_gates = len(config.quadrotor_config.gates)
    stats = []

    # Wait for keyboard input to start.
    # input("Press any key to start")

    # Initial printouts.
    if config.verbose:
        print('\tInitial observation [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0]: ' + str(obs))
        print('\tControl timestep: ' + str(info['ctrl_timestep']))
        print('\tControl frequency: ' + str(info['ctrl_freq']))
        print('\tMaximum episode duration: ' + str(info['episode_len_sec']))
        print('\tNominal quadrotor mass and inertia: ' + str(info['nominal_physical_parameters']))
        print('\tGates properties: ' + str(info['gate_dimensions']))
        print('\tObstacles properties: ' + str(info['obstacle_dimensions']))
        print('\tNominal gates positions [x, y, z, r, p, y, type]: ' + str(info['nominal_gates_pos_and_type']))
        print('\tNominal obstacles positions [x, y, z, r, p, y]: ' + str(info['nominal_obstacles_pos']))
        print('\tFinal target hover position [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r]: ' + str(info['x_reference']))
        print('\tDistribution of the error on the initial state: ' + str(info['initial_state_randomization']))
        print('\tDistribution of the error on the inertial properties: ' + str(info['inertial_prop_randomization']))
        print('\tDistribution of the error on positions of gates and obstacles: ' + str(info['gates_and_obs_randomization']))
        print('\tDistribution of the disturbances: ' + str(info['disturbances']))
        print('\tA priori symbolic model:')
        print('\t\tState: ' + str(info['symbolic_model'].x_sym).strip('vertcat'))
        print('\t\tInput: ' + str(info['symbolic_model'].u_sym).strip('vertcat'))
        print('\t\tDynamics: ' + str(info['symbolic_model'].x_dot).strip('vertcat'))
        print('Input constraints lower bounds: ' + str(env.constraints.input_constraints[0].lower_bounds))
        print('Input constraints upper bounds: ' + str(env.constraints.input_constraints[0].upper_bounds))
        print('State constraints active dimensions: ' + str(config.quadrotor_config.constraints[1].active_dims))
        print('State constraints lower bounds: ' + str(env.constraints.state_constraints[0].lower_bounds))
        print('State constraints upper bounds: ' + str(env.constraints.state_constraints[0].upper_bounds))
        print('\tSymbolic constraints: ')
        for fun in info['symbolic_constraints']:
            print('\t' + str(inspect.getsource(fun)).strip('\n'))

    # Run an experiment.
    ep_start = time.time()
    first_ep_iteration = True
    for i in range(config.num_episodes*CTRL_FREQ*env.EPISODE_LEN_SEC):
        # label for if the trajectory is complete
        complete = False
        # Elapsed sim time.
        curr_time = (i-episode_start_iter)*CTRL_DT

        # Print episode time in seconds on the GUI.
        time_label_id = p.addUserDebugText("Ep. time: {:.2f}s".format(curr_time),
                                           textPosition=[0, 0, 1.5],
                                           textColorRGB=[1, 0, 0],
                                           lifeTime=3*CTRL_DT,
                                           textSize=1.5,
                                           parentObjectUniqueId=0,
                                           parentLinkIndex=-1,
                                           replaceItemUniqueId=time_label_id,
                                           physicsClientId=env.PYB_CLIENT)

        # Compute control input.
        if config.use_firmware:
            # [INSTRUCTIONS:] 
            # vicon_obs provides the state measurements from Vicon
            vicon_obs = [obs[0], 0, obs[2], 0, obs[4], 0, obs[6], obs[7], obs[8], 0, 0, 0]
                # obs = {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r}.
                # vicon_obs = {x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0}.
            if first_ep_iteration:
                action = np.zeros(4)
                reward = 0
                done = False
                info = {}
                first_ep_iteration = False
            command_type, args = ctrl.cmdFirmware(curr_time, vicon_obs, reward, done, info)
            
            # --- debug 
            # print(vicon_obs)

            # Select interface.
            if command_type == Command.FULLSTATE:
                firmware_wrapper.sendFullStateCmd(*args, curr_time)
            elif command_type == Command.TAKEOFF:
                firmware_wrapper.sendTakeoffCmd(*args)
            elif command_type == Command.LAND:
                firmware_wrapper.sendLandCmd(*args)
            elif command_type == Command.STOP:
                firmware_wrapper.sendStopCmd()
                # indicate the trajectory is complete
                complete = True
            elif command_type == Command.GOTO:
                firmware_wrapper.sendGotoCmd(*args)
            elif command_type == Command.NOTIFYSETPOINTSTOP:
                firmware_wrapper.notifySetpointStop()
            elif command_type == Command.NONE:
                pass
            else:
                raise ValueError("[ERROR] Invalid command_type.")

            # Step the environment.
            obs, reward, done, info, action = firmware_wrapper.step(curr_time, action)
        else:
            if first_ep_iteration:
                reward = 0
                done = False
                info = {}
                first_ep_iteration = False
            target_pos, target_vel = ctrl.cmdSimOnly(curr_time, obs, reward, done, info)
            action = thrusts(ctrl.ctrl, ctrl.CTRL_TIMESTEP, ctrl.KF, obs, target_pos, target_vel)
            obs, reward, done, info = env.step(action)

        # Add up reward, collisions, violations.
        cumulative_reward += reward
        if info["collision"][1]:
            collisions_count += 1
            collided_objects.add(info["collision"][0])
        if 'constraint_values' in info and info['constraint_violation'] == True:
            violations_count += 1

        # Printouts.
        if config.verbose and i%int(CTRL_FREQ/2) == 0:
            print('\n'+str(i)+'-th step.')
            print('\tApplied action: ' + str(action))
            print('\tObservation: ' + str(obs))
            print('\tReward: ' + str(reward) + ' (Cumulative: ' + str(cumulative_reward) +')')
            print('\tDone: ' + str(done))
            print('\tCurrent target gate ID: ' + str(info['current_target_gate_id']))
            print('\tCurrent target gate type: ' + str(info['current_target_gate_type']))
            print('\tCurrent target gate in range: ' + str(info['current_target_gate_in_range']))
            print('\tCurrent target gate position: ' + str(info['current_target_gate_pos']))
            print('\tAt goal position: ' + str(info['at_goal_position']))
            print('\tTask completed: ' + str(info['task_completed']))
            if 'constraint_values' in info:
                print('\tConstraints evaluations: ' + str(info['constraint_values']))
                print('\tConstraints violation: ' + str(bool(info['constraint_violation'])))
            print('\tCollision: ' + str(info["collision"]))
            print('\tTotal collisions: ' + str(collisions_count))
            print('\tCollided objects (history): ' + str(collided_objects))      

        # Synchronize the GUI.
        if config.quadrotor_config.gui:
            sync(i-episode_start_iter, ep_start, CTRL_DT)

        # If an episode is complete, reset the environment.
        if done or complete:
            # Append episode stats.
            if info['current_target_gate_id'] == -1:
                gates_passed = num_of_gates
            else:
                gates_passed = info['current_target_gate_id']
            if config.quadrotor_config.done_on_collision and info["collision"][1]:
                termination = 'COLLISION'
            elif config.quadrotor_config.done_on_completion and info['task_completed']:
                termination = 'TASK COMPLETION'
            elif config.quadrotor_config.done_on_violation and info['constraint_violation']:
                termination = 'CONSTRAINT VIOLATION'
            else:
                termination = 'MAX EPISODE DURATION'
            if ctrl.interstep_learning_occurrences != 0:
                interstep_learning_avg = ctrl.interstep_learning_time/ctrl.interstep_learning_occurrences
            else:
                interstep_learning_avg = ctrl.interstep_learning_time
            episode_stats = [
                '[yellow]Flight time (s): '+str(curr_time),
                '[yellow]Reason for termination: '+termination,
                '[green]Gates passed: '+str(gates_passed),
                '[green]Total reward: '+str(cumulative_reward),
                '[red]Number of collisions: '+str(collisions_count),
                '[red]Number of constraint violations: '+str(violations_count),
                '[white]Total and average interstep learning time (s): '+str(ctrl.interstep_learning_time)+', '+str(interstep_learning_avg),
                '[white]Interepisode learning time (s): '+str(ctrl.interepisode_learning_time),
                ]
            stats.append(episode_stats)
            # break the loop when the trajectory is complete
            break 

    # Close the environment and print timing statistics.
    env.close()
    elapsed_sec = time.time() - START
    print(str("\n{:d} iterations (@{:d}Hz) and {:d} episodes in {:.2f} sec, i.e. {:.2f} steps/sec for a {:.2f}x speedup.\n"
          .format(i,
                  env.CTRL_FREQ,
                  config.num_episodes,
                  elapsed_sec,
                  i/elapsed_sec,
                  (i*CTRL_DT)/elapsed_sec
                  )
          ))

    # Print episodes summary.
    tree = Tree("Summary")
    for idx, ep in enumerate(stats):
        ep_tree = tree.add('Episode ' + str(idx+1))
        for val in ep:
            ep_tree.add(val)
    print('\n\n')
    print(tree)
    print('\n\n')

if __name__ == "__main__":
    run()
