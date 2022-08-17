"""Demo script.

Run as:

    $ python3 getting_started.py --overrides ./getting_started.yaml

"""
import time
from functools import partial
import numpy as np

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.envs.gym_pybullet_drones.Logger import Logger

from edit_this import Controller, Command

try:
    import cffirmware
except ImportError:
    FIRMWARE_INSTALLED = False
else:
    FIRMWARE_INSTALLED = True
finally:
    print("Module 'cffirmware' available:", FIRMWARE_INSTALLED)


def main():
    """The main function creating, running, and closing an environment.

    """

    # Start a timer.
    START = time.time()

    # Load configuration.
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()
    if config.use_firmware and not FIRMWARE_INSTALLED:
        raise RuntimeError("[ERROR] Module 'cffirmware' not installed.")

    # Create environment.
    if config.use_firmware:
        pyb_freq = config.quadrotor_config['pyb_freq']
        ctrl_freq = config.quadrotor_config['ctrl_freq']
        firmware_freq = config.quadrotor_config['firmware_freq']

        assert(pyb_freq % firmware_freq == 0), "pyb_freq must be a multiple of firmware freq"

        # The env.step is called at a firmware_freq rate, but this is not as intuitive to the end user, and so 
        # we abstract the difference. This allows ctrl_freq to be the rate at which the user sends ctrl signals, 
        # not the firmware. 
        config.quadrotor_config['ctrl_freq'] = firmware_freq

        env_func = partial(make, 'quadrotor', **config.quadrotor_config)
        firmware_wrapper = make('firmware',
                    env_func, firmware_freq, ctrl_freq
                    ) 
        env = firmware_wrapper.env
        action = np.zeros(4)
    else:
        env = make('quadrotor', **config.quadrotor_config)
        ctrl_freq = env.CTRL_FREQ

    # Reset the environment, obtain the initial observations and info dictionary.
    obs, info = env.reset()

    # Create controller.
    ctrl = Controller(obs, info, config.use_firmware)

    # Initialize firmware.
    if config.use_firmware:
        firmware_wrapper.update_initial_state(obs)

    # Set episode counters.
    num_episodes = 2
    episodes_count = 1

    # Create a logger.
    logger = Logger(logging_freq_hz=env.CTRL_FREQ)

    # Run an experiment.
    for i in range(num_episodes*ctrl_freq*env.EPISODE_LEN_SEC):

        # Step by keyboard input.
        # _ = input('Press any key to continue.')

        # Elapsed sim time.
        curr_time = (i%(ctrl_freq*env.EPISODE_LEN_SEC))*env.CTRL_TIMESTEP

        # Compute control input.
        if config.use_firmware:
            command_type, args = ctrl.cmdFirmware(curr_time, obs)

            if command_type == Command.FULLSTATE:
                firmware_wrapper.sendFullStateCmd(*args)
            elif command_type == Command.TAKEOFF:
                firmware_wrapper.sendTakeoffCmd(*args)
            elif command_type == Command.LAND:
                firmware_wrapper.sendLandCmd(*args)
            elif command_type == Command.STOP:
                firmware_wrapper.sendStopCmd()
            elif command_type == Command.GOTO:
                firmware_wrapper.sendGotoCmd(*args)
            elif command_type == Command.NONE:
                pass
            else:
                raise ValueError("[ERROR] Invalid command_type.")

            # Step the environment.
            obs, reward, done, info, action = firmware_wrapper.step(i%(env.CTRL_FREQ*env.EPISODE_LEN_SEC), action)
        else:
            action = ctrl.cmdSimOnly(curr_time, obs)
            obs, reward, done, info = env.step(action)

        if info["collision"][1]:
            collisions_count += 1
            collided_objects.add(info["collision"][0])

        # Printouts.
        if i%100 == 0:
            print('\n'+str(i)+'-th step.')
            print('\tApplied action: ' + str(action))
            print('\tObservation: ' + str(obs))
            print('\tReward: ' + str(reward))
            print('\tDone: ' + str(done))
            if 'constraint_values' in info:
                print('\tConstraints evaluations: ' + str(info['constraint_values']))
                print('\tConstraints violation: ' + str(bool(info['constraint_violation'])))
            print('\tCollisions: ' + str(collisions_count))
            print('\tCollided objects: ' + str(collided_objects))

        # Log data.
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
            logger.plot(comment="get_start-episode-"+str(episodes_count))

            # CSV save.
            logger.save_as_csv(comment="get_start-episode-"+str(episodes_count))

            # Create a new logger.
            logger = Logger(logging_freq_hz=env.CTRL_FREQ)

            episodes_count += 1
            if episodes_count > config.num_episodes:
                break
            collisions_count = 0
            collided_objects = set()

            # Reset the environment.
            new_initial_obs, new_initial_info = env.reset()
            print(str(episodes_count)+'-th reset.')
            print('Reset obs' + str(new_initial_obs))
            print('Reset info' + str(new_initial_info))

            # Re-initialize firmware.
            if config.use_firmware:
                firmware_wrapper.update_initial_state(new_initial_obs)
                action = np.zeros(4)

    # Close the environment and print timing statistics.
    env.close()
    elapsed_sec = time.time() - START
    print(str("\n{:d} iterations (@{:d}Hz) and {:d} episodes in {:.2f} sec, i.e. {:.2f} steps/sec for a {:.2f}x speedup.\n\n"
          .format(i,
                  env.CTRL_FREQ,
                  config.num_episodes,
                  elapsed_sec,
                  i/elapsed_sec,
                  (i*env.CTRL_TIMESTEP)/elapsed_sec
                  )))

if __name__ == "__main__":
    main()
