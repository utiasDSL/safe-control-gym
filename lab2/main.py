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
    from planner import Controller
except ImportError:
    # Test import.
    from .project_utils import Command, thrusts
    from .planner import Controller

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

    CTRL_FREQ = config.quadrotor_config['ctrl_freq']
    CTRL_DT = 1/CTRL_FREQ

    # Create environment.
    # [INSTRUCTIONS:] 
    # When the simulation env. reset, we can get the current observation and information 
    # of the simulation env.
    env = make('quadrotor', **config.quadrotor_config)
    # Reset the environment, obtain the initial observations and info dictionary.
    obs, info = env.reset()

    # Create controller.
    # [INSTRUCTIONS:] 
    # vicon_obs indicates the initial observation (initial state) from Vicon.
    # obs = {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r}.
    # vicon_obs = {x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0}.
    vicon_obs = [obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[7], obs[8], 0, 0, 0]

    # NOTE: students can get access to the information of the gates and obstacles 
    #       when creating the controller object. 

    # the circle radius is the same as init_x: 3.0 in the config file
    circle_radius = obs[0] # m

    ctrl = Controller(circle_radius, vicon_obs, info, verbose=config.verbose)
    
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
        curr_time = (i-episode_start_iter)*CTRL_DT # CTRL_DT = 0.02

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

        # Compute control commands
        if first_ep_iteration:
            reward = 0
            done = False
            info = {}
            first_ep_iteration = False
        # Get reference pos, vel, acc from the circle trajectory
        target_pos, target_vel, target_acc = ctrl.getRef(curr_time, obs, reward, done, info)
        # TODO: implement the geometric controller in the computeAction function
        action = ctrl.computeAction(obs, target_pos, target_vel, target_acc)
        # Get new observation after taking the computed actions
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
