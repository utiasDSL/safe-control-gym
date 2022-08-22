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
from pycrazyswarm import * 


def main():
    """The main function creating, running, and closing an environment.

    """
    swarm = CrazySwarm() 
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    # Start a timer.
    START = time.time()

    # Load configuration.
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()
    ctrl_freq = config.quadrotor_config['ctrl_freq']

    # Get gate and obstacle position for minimal reformatting 
    env = make('quadrotor', **config.quadrotor_config)
    _, info = env.reset()
    env.close()

    # Create controller.
    ctrl = Controller(cf.position, info, config.use_firmware)

    # Create a logger and counters
    logger = Logger(logging_freq_hz=ctrl_freq)

    # Run an experiment.
    start_time = timeHelper.time()
    while not timeHelper.isShutdown():

        # Step by keyboard input.
        # _ = input()

        # Elapsed flight time.
        t = timeHelper.time() - start_time

        pos = cf.position() 
        # Compute control input.
        if config.use_firmware:
            command_type, args = ctrl.cmdFirmware(t, None, vicon_pos=pos)

            if command_type == Command.FULLSTATE:
                cf.cmdFullState(*args)
            elif command_type == Command.TAKEOFF:
                cf.takeoff(*args)
            elif command_type == Command.LAND:
                cf.land(*args)
            elif command_type == Command.STOP:
                cf.stop()
            elif command_type == Command.GOTO:
                cf.goTo(*args)
            elif command_type == Command.NONE:
                pass
            else:
                raise ValueError("[ERROR] Invalid command_type.")
        else:
            # Only available in sim. This will raise NotImplementedError
            action = ctrl.cmdSimOnly(t, obs)
            obs, reward, done, info = env.step(action)


        # Log data.
        # pos = [obs[0],obs[2],obs[4]]
        # rpy = [obs[6],obs[7],obs[8]]
        # vel = [obs[1],obs[3],obs[5]]
        # ang_vel = [obs[9],obs[10],obs[11]]
        logger.log(drone=0,
                   timestamp=t,
                   state=pos,
                   )

        # If an episode is complete, reset the environment.
        if done:
            # Plot logging.
            logger.plot(comment="get_start-episode-")

            # CSV save.
            logger.save_as_csv(comment="get_start-episode-")

            # Create a new logger.
            logger = Logger(logging_freq_hz=ctrl_freq)

            break


    # Close the environment and print timing statistics.
    elapsed_sec = time.time() - START
    print(str("\n{:d}s flight time (@{:d}Hz).\n\n"
          .format(elapsed_sec,
                  ctrl_freq,
                  )))

if __name__ == "__main__":
    main()
