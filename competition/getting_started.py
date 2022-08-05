"""Demo script.

Example:

    $ python3 getting_started.py --overrides ./getting_started.yaml

"""
import os
import time
from functools import partial
import yaml
import inspect
import numpy as np
import pybullet as p
import casadi as cs
import matplotlib.pyplot as plt

from safe_control_gym.utils.utils import str2bool
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.envs.gym_pybullet_drones.Logger import Logger

from edit_this import Controller, Command

def main():
    """The main function creating, running, and closing an environment.

    """

    # Start a timer.
    START = time.time()

    # Create an environment
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()
    try: 
        raise
        import cffirmware
        env_func = partial(make, 'quadrotor', **config.quadrotor_config)
        firmware_wrapper = make('firmware',
                    env_func,
                    ) 
        env = firmware_wrapper.env
        firmware_exists = True
    except:
        env = make('quadrotor', **config.quadrotor_config)
        firmware_exists = False
    
    # Reset the environment, obtain and print the initial observations.
    initial_obs, initial_info = env.reset()
    obs = initial_obs

    # Dynamics info
    print('\nPyBullet dynamics info:')
    print('\t' + str(p.getDynamicsInfo(bodyUniqueId=env.DRONE_IDS[0], linkIndex=-1, physicsClientId=env.PYB_CLIENT)))
    print('\nInitial reset.')
    print('\tInitial observation: ' + str(initial_obs))

    # Create maze.
    for obstacle in config.obstacles:
        p.loadURDF(os.path.join(env.URDF_DIR, "obstacle.urdf"),
                   obstacle[0:3],
                   p.getQuaternionFromEuler(obstacle[3:6]),
                   physicsClientId=env.PYB_CLIENT)
    for gate in config.gates:
        p.loadURDF(os.path.join(env.URDF_DIR, "portal.urdf"),
                   gate[0:3],
                   p.getQuaternionFromEuler(gate[3:6]),
                   physicsClientId=env.PYB_CLIENT)

    # Create a controller.
    ctrl = Controller(env)

    if firmware_exists:
        firmware_wrapper.update_initial_state(obs)
    action = np.zeros(4)

    # Set episode counters.
    num_episodes = 2
    episodes_count = 1

    # Create a logger.
    logger = Logger(logging_freq_hz=env.CTRL_FREQ)

    # Run an experiment.
    for i in range(num_episodes*env.CTRL_FREQ*env.EPISODE_LEN_SEC):

        # Step by keyboard input.
        # _ = input('Press any key to continue.')

        # Compute control input.
        if firmware_exists:
            command_type, args = ctrl.getCmd(i*(1/env.CTRL_FREQ), obs)

            if command_type == Command.NONE:
                pass
            elif command_type == Command.FULLSTATE:
                firmware_wrapper.sendFullStateCmd(*args)
            elif command_type == Command.TAKEOFF:
                firmware_wrapper.sendTakeoffCmd(*args)
            elif command_type == Command.LAND:
                firmware_wrapper.sendLandCmd(*args)
            elif command_type == Command.STOP:
                firmware_wrapper.sendStopCmd()
            elif command_type == Command.GOTO:
                firmware_wrapper.sendGotoCmd(*args)

            # Step the environment and print all returned information.
            obs, reward, done, info, action = firmware_wrapper.step(i, action)
        else:
            action = ctrl.cmdFullState(i%(env.CTRL_FREQ*env.EPISODE_LEN_SEC), obs)
            obs, reward, done, info = env.step(action)

        # Print outs.
        if i%10 == 0:
            print('\n'+str(i)+'-th step.')
            out = '\tApplied action: ' + str(action)
            print(out)
            out = '\tObservation: ' + str(obs)
            print(out)
            out = '\tReward: ' + str(reward)
            print(out)
            out = '\tDone: ' + str(done)
            print(out)
            if 'constraint_values' in info:
                out = '\tConstraints evaluations: ' + str(info['constraint_values'])
                print(out)
                out = '\tConstraints violation: ' + str(bool(info['constraint_violation']))
                print(out)

        # Log data
        pos = [obs[0],obs[2],obs[4]]
        rpy = [obs[6],obs[7],obs[8]]
        vel = [obs[1],obs[3],obs[5]]
        ang_vel = [obs[9],obs[10],obs[11]]
        logger.log(drone=0,
                   timestamp=i/env.CTRL_FREQ,
                   state=np.hstack([pos, np.zeros(4), rpy, vel, ang_vel, np.sqrt(action/env.KF)]),
                   )

        # If an episode is complete, reset the environment.
        if done:
            # Plot logging.
            logger.plot()

            # # CSV safe
            logger.save_as_csv("pid-episode-"+str(episodes_count))

            # Create a new logger.
            logger = Logger(logging_freq_hz=env.CTRL_FREQ)

            # Reset the environment
            episodes_count += 1
            new_initial_obs, new_initial_info = env.reset()
            print(str(episodes_count)+'-th reset.')
            print('Reset obs' + str(new_initial_obs))
            print('Reset info' + str(new_initial_info))

            # Create maze.
            for obstacle in config.obstacles:
                p.loadURDF(os.path.join(env.URDF_DIR, "obstacle.urdf"),
                           obstacle[0:3],
                           p.getQuaternionFromEuler(obstacle[3:6]),
                           physicsClientId=env.PYB_CLIENT)
            for gate in config.gates:
                p.loadURDF(os.path.join(env.URDF_DIR, "portal.urdf"),
                           gate[0:3],
                           p.getQuaternionFromEuler(gate[3:6]),
                           physicsClientId=env.PYB_CLIENT)

    # Close the environment and print timing statistics.
    env.close()
    elapsed_sec = time.time() - START
    out = str("\n{:d} iterations (@{:d}Hz) and {:d} episodes in {:.2f} seconds, i.e. {:.2f} steps/sec for a {:.2f}x speedup.\n\n"
          .format(i, env.CTRL_FREQ, num_episodes, elapsed_sec, i/elapsed_sec, (i*env.CTRL_TIMESTEP)/elapsed_sec))
    print(out)

if __name__ == "__main__":
    main()
