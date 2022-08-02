"""3D quadrotor example script.

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
    # Set iterations and episode counter.
    num_episodes = 1
    ITERATIONS = int(15000)
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
    

    # Create a logger.
    logger = Logger(logging_freq_hz=env.CTRL_FREQ)

    # Run an experiment.
    obs = initial_obs
    if firmware_exists:
        firmware_wrapper.update_initial_state(obs)
    action = np.zeros(4)
    for i in range(ITERATIONS):
        # time.sleep(0.4)

        # Step by keyboard input
        # _ = input('Press any key to continue.')

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
            action = ctrl.cmdFullState(i, obs)
            obs, reward, done, info = env.step(action)

        #
        # print('\n'+str(i)+'-th step.')
        out = '\tApplied action: ' + str(action)
        print(out)
        out = '\tObservation: ' + str(obs)
        # print(out)
        out = '\tReward: ' + str(reward)
        # print(out)
        out = '\tDone: ' + str(done)
        # print(out)
        if 'constraint_values' in info:
            out = '\tConstraints evaluations: ' + str(info['constraint_values'])
            # print(out)
            out = '\tConstraints violation: ' + str(bool(info['constraint_violation']))
            # print(out)

        # Log data
        pos = [obs[0],obs[2],obs[4]]
        rpy = [obs[6],obs[7],obs[8]]
        vel = [obs[1],obs[3],obs[5]]
        ang_vel = [obs[9],obs[10],obs[11]]
        logger.log(drone=0,
                   timestamp=i/env.CTRL_FREQ,
                   state=np.hstack([pos, np.zeros(4), rpy, vel, ang_vel, np.sqrt(action/env.KF)]),
                   # control=np.hstack([ref_x[i], ref_y[i], ref_z[i], np.zeros(9)])
                   )

        # If an episode is complete, reset the environment.
        if done:
            num_episodes += 1
            new_initial_obs, new_initial_info = env.reset()
            print(str(num_episodes)+'-th reset.', 7)
            print('Reset obs' + str(new_initial_obs), 2)
            print('Reset info' + str(new_initial_info), 0)

            # Plot logging.
            logger.plot()

            # # CSV safe
            logger.save_as_csv("pid-episode-"+str(num_episodes-1))

            # Create a new logger.
            logger = Logger(logging_freq_hz=env.CTRL_FREQ)

    # Close the environment and print timing statistics.
    env.close()
    elapsed_sec = time.time() - START
    out = str("\n{:d} iterations (@{:d}Hz) and {:d} episodes in {:.2f} seconds, i.e. {:.2f} steps/sec for a {:.2f}x speedup.\n\n"
          .format(ITERATIONS, env.CTRL_FREQ, num_episodes, elapsed_sec, ITERATIONS/elapsed_sec, (ITERATIONS*env.CTRL_TIMESTEP)/elapsed_sec))
    print(out)

if __name__ == "__main__":
    main()
