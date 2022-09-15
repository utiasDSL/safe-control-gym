"""Base script.

Run as:

    $ python3 getting_started.py --overrides ./getting_started.yaml

Look for instructions in `README.md` and `edit_this.py`.

"""
import time
import numpy as np
import pybullet as p
import argparse
import importlib

from functools import partial

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import sync
from safe_control_gym.envs.gym_pybullet_drones.Logger import Logger


try:
    import pycffirmware
except ImportError:
    FIRMWARE_INSTALLED = False
else:
    FIRMWARE_INSTALLED = True
finally:
    print("Module 'cffirmware' available:", FIRMWARE_INSTALLED)


def run(run_type, test=False):
    """The main function creating, running, and closing an environment over N episodes.

    """
    mod = importlib.import_module(f'{run_type}.edit_this')

    # Start a timer.
    START = time.time()

    # Load configuration.
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge(config_override=[f"{run_type}/getting_started.yaml"])

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
    if config.use_firmware:
        FIRMWARE_FREQ = 500
        assert(config.quadrotor_config['pyb_freq'] % FIRMWARE_FREQ == 0), "pyb_freq must be a multiple of firmware freq"
        # The env.step is called at a firmware_freq rate, but this is not as intuitive to the end user, and so 
        # we abstract the difference. This allows ctrl_freq to be the rate at which the user sends ctrl signals, 
        # not the firmware. 
        config.quadrotor_config['ctrl_freq'] = FIRMWARE_FREQ
        env_func = partial(make, 'quadrotor', **config.quadrotor_config)
        firmware_wrapper = make('firmware',
                    env_func, FIRMWARE_FREQ, CTRL_FREQ, verbose=True
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
    vicon_obs = [obs[0], 0, obs[2], 0, obs[4], 0, obs[6], obs[7], obs[8], 0, 0, 0]
        # obs = {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r}.
        # vicon_obs = {x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0}.
    ctrl = mod.Controller(vicon_obs, info, config.use_firmware, verbose=config.verbose)

    # Create a logger and counters
    logger = Logger(logging_freq_hz=CTRL_FREQ)
    episodes_count = 1
    cumulative_reward = 0
    collisions_count = 0
    collided_objects = set()
    episode_start_iter = 0
    text_label_id = p.addUserDebugText("", textPosition=[0, 0, 1],physicsClientId=env.PYB_CLIENT)

    # Wait for keyboard input to start.
    # input("Press any key to start")

    # Run an experiment.
    ep_start = time.time()
    first_ep_iteration = True
    for i in range(config.num_episodes*CTRL_FREQ*env.EPISODE_LEN_SEC):

        # Step by keyboard input.
        # _ = input("Press any key to continue")

        # Elapsed sim time.
        curr_time = (i-episode_start_iter)*CTRL_DT

        # Print episode time in seconds on the GUI.
        text_label_id = p.addUserDebugText("Ep. time: {:.2f}s".format(curr_time),
                                           textPosition=[0, 0, 1.5],
                                           textColorRGB=[1, 0, 0],
                                           lifeTime=3*CTRL_DT,
                                           textSize=1.5,
                                           parentObjectUniqueId=0,
                                           parentLinkIndex=-1,
                                           replaceItemUniqueId=text_label_id,
                                           physicsClientId=env.PYB_CLIENT)

        # Compute control input.
        if config.use_firmware:
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

            # Select interface.
            if command_type == mod.Command.FULLSTATE:
                firmware_wrapper.sendFullStateCmd(*args, curr_time)
            elif command_type == mod.Command.TAKEOFF:
                firmware_wrapper.sendTakeoffCmd(*args)
            elif command_type == mod.Command.LAND:
                firmware_wrapper.sendLandCmd(*args)
            elif command_type == mod.Command.STOP:
                firmware_wrapper.sendStopCmd()
            elif command_type == mod.Command.GOTO:
                firmware_wrapper.sendGotoCmd(*args)
            elif command_type == mod.Command.NOTIFYSETPOINTSTOP:
                firmware_wrapper.notifySetpointStop(*args)
            elif command_type == mod.Command.NONE:
                pass
            elif command_type == mod.Command.FINISHED:
                break
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
            action = ctrl._thrusts(obs, target_pos, target_vel)
            obs, reward, done, info = env.step(action)

        # Update the controller internal state and models.
        ctrl.interStepLearn(action, obs, reward, done, info)

        # Add up reward and collisions.
        cumulative_reward += reward
        if info["collision"][1]:
            collisions_count += 1
            collided_objects.add(info["collision"][0])

        # Printouts.
        if config.verbose and i%int(CTRL_FREQ/2) == 0:
            print('\n'+str(i)+'-th step.')
            print('\tApplied action: ' + str(action))
            print('\tObservation: ' + str(obs))
            print('\tReward: ' + str(reward) + ' (Cumulative: ' + str(cumulative_reward) +')')
            print('\tDone: ' + str(done))
            if 'constraint_values' in info:
                print('\tConstraints evaluations: ' + str(info['constraint_values']))
                print('\tConstraints violation: ' + str(bool(info['constraint_violation'])))
            print('\tCurrent target gate ID: ' + str(info['current_target_gate_id']))
            print('\tCurrent target gate in range: ' + str(info['current_target_gate_in_range']))
            print('\tCurrent target gate position: ' + str(info['current_target_gate_pos']))
            print('\tAt goal position: ' + str(info['at_goal_position']))
            print('\tCollisions: ' + str(collisions_count))
            print('\tCollided objects: ' + str(collided_objects))

        # Log data.
        pos = [obs[0],obs[2],obs[4]]
        rpy = [obs[6],obs[7],obs[8]]
        vel = [obs[1],obs[3],obs[5]]
        bf_rates = [obs[9],obs[10],obs[11]]
        logger.log(drone=0,
                   timestamp=i/CTRL_FREQ,
                   state=np.hstack([pos, np.zeros(4), rpy, vel, bf_rates, np.sqrt(action/env.KF)])
                   )

        # Synchronize the GUI.
        if config.quadrotor_config.gui:
            sync(i-episode_start_iter, ep_start, CTRL_DT)

        # If an episode is complete, reset the environment.
        if done:
            # Plot logging (comment as desired).
            if not test:
                logger.plot(comment="get_start-episode-"+str(episodes_count))

            # CSV save.
            logger.save_as_csv(comment="get_start-episode-"+str(episodes_count))

            # Create a new logger.
            logger = Logger(logging_freq_hz=CTRL_FREQ)

            # Update the controller internal state and models.
            ctrl.interEpisodeLearn()

            # Reset/update counters.
            episodes_count += 1
            if episodes_count > config.num_episodes:
                break
            cumulative_reward = 0
            collisions_count = 0
            collided_objects = set()

            # Reset the environment.
            if config.use_firmware:
                # Re-initialize firmware.
                new_initial_obs, new_initial_info = firmware_wrapper.reset()
            else:
                new_initial_obs, new_initial_info = env.reset()
            first_ep_iteration = True

            # ctrl._draw_trajectory(new_initial_info)

            if config.verbose:
                print(str(episodes_count)+'-th reset.')
                print('Reset obs' + str(new_initial_obs))
                print('Reset info' + str(new_initial_info))
            
            episode_start_iter = i+1
            ep_start = time.time()

    # Close the environment and print timing statistics.
    env.close()
    elapsed_sec = time.time() - START
    print(str("\n{:d} iterations (@{:d}Hz) and {:d} episodes in {:.2f} sec, i.e. {:.2f} steps/sec for a {:.2f}x speedup.\n\n"
          .format(i,
                  env.CTRL_FREQ,
                  config.num_episodes,
                  elapsed_sec,
                  i/elapsed_sec,
                  (i*CTRL_DT)/elapsed_sec
                  )
          ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--run", type=str)
    args = parser.parse_known_args()[0]

    run(args.run)
