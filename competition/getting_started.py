"""Demo script.

Run as:

    $ python3 getting_started.py --overrides ./getting_started.yaml

"""
import time
import numpy as np

from functools import partial

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.envs.gym_pybullet_drones.Logger import Logger

from edit_this import Controller, Command

try:
    import pycffirmware
except ImportError:
    FIRMWARE_INSTALLED = False
else:
    FIRMWARE_INSTALLED = True
finally:
    print("Module 'cffirmware' available:", FIRMWARE_INSTALLED)


def main():
    """The main function creating, running, and closing an environment over N episodes.

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
        if "observation" in config.quadrotor_config.disturbances:
            # pass
            # raise NotImplementedError("Observation noise not supported with firmware wrapper.")
            del config.quadrotor_config.disturbances['observation']

        pyb_freq = config.quadrotor_config['pyb_freq']
        ctrl_freq = config.quadrotor_config['ctrl_freq']
        firmware_freq = config.quadrotor_config['firmware_freq']
        ctrl_dt = 1/ctrl_freq

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
        ctrl_dt = env.CTRL_TIMESTEP

    # Reset the environment, obtain the initial observations and info dictionary.
    obs, info = env.reset()

    # Initialize firmware.
    if config.use_firmware:
        info['ctrl_timestep'] = ctrl_dt
        info['ctrl_freq'] = ctrl_freq
        
        # Create controller.
        ctrl = Controller(obs, info, config.use_firmware, verbose=config.verbose)
        firmware_wrapper.update_initial_state(obs)
    else:
        # Create controller.
        ctrl = Controller(obs, info, config.use_firmware)

    # Create a logger and counters
    logger = Logger(logging_freq_hz=ctrl_freq)
    episodes_count = 1
    cumulative_reward = 0
    collisions_count = 0
    collided_objects = set()

    # Run an experiment.
    for i in range(config.num_episodes*ctrl_freq*env.EPISODE_LEN_SEC):

        # Step by keyboard input.
        # _ = input()

        # Elapsed sim time.
        curr_time = (i%(ctrl_freq*env.EPISODE_LEN_SEC))*ctrl_dt

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
            obs, reward, done, info, action = firmware_wrapper.step(curr_time, action)
        else:
            action = ctrl.cmdSimOnly(curr_time, obs)
            obs, reward, done, info = env.step(action)

        # Update the controller internal state and models.
        ctrl.learn(action, obs, reward, done, info)

        cumulative_reward += reward
        if info["collision"][1]:
            collisions_count += 1
            collided_objects.add(info["collision"][0])

        # Printouts.
        if config.verbose and i%100 == 0:
            print('\n'+str(i)+'-th step.')
            print('\tApplied action: ' + str(action))
            print('\tObservation: ' + str(obs))
            print('\tReward: ' + str(reward) + ' (Cumulative: ' + str(cumulative_reward) +')')
            print('\tDone: ' + str(done))
            if 'constraint_values' in info:
                print('\tConstraints evaluations: ' + str(info['constraint_values']))
                print('\tConstraints violation: ' + str(bool(info['constraint_violation'])))
            print('\tCurrent target gate: ' + str(info['current_target_gate']))
            print('\tAt goal position: ' + str(info['at_goal_position']))
            print('\tCollisions: ' + str(collisions_count))
            print('\tCollided objects: ' + str(collided_objects))

        # Log data.
        pos = [obs[0],obs[2],obs[4]]
        rpy = [obs[6],obs[7],obs[8]]
        vel = [obs[1],obs[3],obs[5]]
        bf_rates = [obs[9],obs[10],obs[11]]
        logger.log(drone=0,
                   timestamp=i/ctrl_freq,
                   state=np.hstack([pos, np.zeros(4), rpy, vel, bf_rates, np.sqrt(action/env.KF)]),
                   )

        # If an episode is complete, reset the environment.
        if done:
            # Plot logging.
            logger.plot(comment="get_start-episode-"+str(episodes_count))

            # CSV save.
            logger.save_as_csv(comment="get_start-episode-"+str(episodes_count))

            # Create a new logger.
            logger = Logger(logging_freq_hz=ctrl_freq)

            episodes_count += 1
            if episodes_count > config.num_episodes:
                break
            cumulative_reward = 0
            collisions_count = 0
            collided_objects = set()

            # Reset the environment.
            new_initial_obs, new_initial_info = env.reset()
            if config.verbose:
                print(str(episodes_count)+'-th reset.')
                print('Reset obs' + str(new_initial_obs))
                print('Reset info' + str(new_initial_info))

            # Re-initialize firmware.
            if config.use_firmware:
                firmware_wrapper.reset()
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
                  (i*ctrl_dt)/elapsed_sec
                  )))

if __name__ == "__main__":
    main()
