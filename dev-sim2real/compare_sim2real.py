"""Base script.

Run as:

    $ python3 getting_started.py --overrides ./getting_started.yaml

Look for instructions in `README.md` and `edit_this.py`.

"""
import time
import sys
LOG_TIME = time.time()
LOG_NAME = "logs/sim2real_comparison"
sys.stdout = open(f"{LOG_NAME}_{LOG_TIME}.txt", "a")
import yaml
import argparse
from functools import partial
from enum import Enum
import importlib

import numpy as np
import pybullet as p
from scipy.optimize import basinhopping, differential_evolution

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import sync
from safe_control_gym.envs.gym_pybullet_drones.Logger import Logger
import numpy as np

from sim_data_utils import load_average_run
from trial_data_utils import align_data
import pycffirmware



def run_trial(config, controller, trajectory):
    """The main function creating, running, and closing an environment over N episodes.

    """
    # Overwrite config values to speed up optimization
    config.quadrotor_config['gui'] = False
    # Sets initial conditions to match vicon reading 
    config.quadrotor_config.init_state.init_x = trajectory[0,1]
    config.quadrotor_config.init_state.init_y = trajectory[0,2]
    config.quadrotor_config.init_state.init_z = trajectory[0,3] + 0.01

    roll, pitch, yaw = p.getEulerFromQuaternion(trajectory[0,4:])
    config.quadrotor_config.init_state.init_phi = roll
    config.quadrotor_config.init_state.init_theta = pitch
    config.quadrotor_config.init_state.init_psi = yaw

    config.verbose = False
    assert(config.use_firmware), "use_firmware must be true to fit firmware."

    # Check firmware configuration.
    CTRL_FREQ = config.quadrotor_config['ctrl_freq']
    CTRL_DT = 1/CTRL_FREQ

    # Create environment.
    FIRMWARE_FREQ = 500
    assert(config.quadrotor_config['pyb_freq'] % FIRMWARE_FREQ == 0), "pyb_freq must be a multiple of firmware freq"
    # The env.step is called at a firmware_freq rate, but this is not as intuitive to the end user, and so 
    # we abstract the difference. This allows ctrl_freq to be the rate at which the user sends ctrl signals, 
    # not the firmware. 
    config.quadrotor_config['ctrl_freq'] = FIRMWARE_FREQ
    env_func = partial(make, 'quadrotor', **config.quadrotor_config)
    firmware_wrapper = make('firmware',
                env_func, FIRMWARE_FREQ, CTRL_FREQ, verbose=False
                ) 
    firmware_wrapper.env.GROUND_PLANE_Z = trajectory[0,3]
    obs, info = firmware_wrapper.reset()
    info['ctrl_timestep'] = CTRL_DT
    info['ctrl_freq'] = CTRL_FREQ
    env = firmware_wrapper.env

    # Create controller.
    vicon_obs = [obs[0], 0, obs[2], 0, obs[4], 0, obs[6], obs[7], obs[8], 0, 0, 0]
    ctrl = controller.Controller(vicon_obs, info, config.use_firmware, verbose=False)

    # Create a logger and counters
    logger = Logger(logging_freq_hz=CTRL_FREQ)
    episodes_count = 1
    cumulative_reward = 0
    collisions_count = 0
    collided_objects = set()
    episode_start_iter = 0

    # Run an experiment.
    first_ep_iteration = True
    for i in range(config.num_episodes*CTRL_FREQ*env.EPISODE_LEN_SEC):

        # Elapsed sim time.
        curr_time = (i-episode_start_iter)*CTRL_DT

        # Compute control input.
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
        if command_type == controller.Command.FULLSTATE:
            firmware_wrapper.sendFullStateCmd(*args, curr_time)
        elif command_type == controller.Command.TAKEOFF:
            firmware_wrapper.sendTakeoffCmd(*args)
        elif command_type == controller.Command.LAND:
            firmware_wrapper.sendLandCmd(*args)
        elif command_type == controller.Command.STOP:
            firmware_wrapper.sendStopCmd()
        elif command_type == controller.Command.GOTO:
            firmware_wrapper.sendGotoCmd(*args)
        elif command_type == controller.Command.NOTIFYSETPOINTSTOP:
            firmware_wrapper.notifySetpointStop(*args)
        elif command_type == controller.Command.NONE:
            pass
        elif command_type == controller.Command.FINISHED:
            break
        else:
            raise ValueError("[ERROR] Invalid command_type.")

        # Step the environment.
        obs, reward, done, info, action = firmware_wrapper.step(curr_time, action)

        # If an episode is complete, reset the environment.
        if done:
            episodes_count += 1
            if firmware_wrapper._error:
                env.close()
                return float('inf')
            if episodes_count > config.num_episodes:
                break
            
            firmware_wrapper.reset()
            first_ep_iteration = True
            
            episode_start_iter = i+1


    # Close the environment and print timing statistics.
    env.close()


    states = np.array(firmware_wrapper.states)

    aligned_states = align_data([states])[:len(trajectory)]

    sub = np.subtract(trajectory[:,1:4],aligned_states[:,1:4])

    square = np.square(sub)

    dist = np.sqrt(np.mean(square, 1))

    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt

    from trial_data_utils import get_data

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    xline = trajectory[:,1]
    yline = trajectory[:,2]
    zline = trajectory[:,3]
    ax.plot3D(xline, yline, zline, label="real_world")

    xline = aligned_states[:,1]
    yline = aligned_states[:,2]
    zline = aligned_states[:,3]
    ax.plot3D(xline, yline, zline, label="sim")

    ax.legend()
    ax.set_xlim3d(-3.5, 3.5)
    ax.set_ylim3d(-3.5, 3.5)
    ax.set_zlim3d(0, 2.5)

    plt.show()

    # rmse = np.sqrt(np.square(np.subtract(trajectory[:,1:4],aligned_states[:,1:4])).mean())
    return np.mean(dist)

    
trial_counter = 0
def run(trajectories, controllers):
    global trial_counter
    ret = 0
    start = time.time()

    for run in trajectories.keys():
        # print("Running", run)
        # Load configuration.
        CONFIG_FACTORY = ConfigFactory()
        config = CONFIG_FACTORY.merge(config_override=[f"{run}/getting_started.yaml"])
        dist = run_trial(config, controllers[run], trajectories[run])
        print(f"{run} with avg dist {dist}m")
        ret += dist

    ret /= len(trajectories.keys())

    print(f"[{trial_counter}] Score {ret} achieved in {time.time() - start:.3}s", file=open(f"{LOG_NAME}_{LOG_TIME}.txt", "a"))
    trial_counter += 1
    return ret


if __name__ == "__main__":
    '''
    ACTION_DELAY
    SENSOR_DELAY
    GYRO_SCALE
    GRYO_CONST
    '''
    trajectories = {
        'ellipse': load_average_run('ellipse'),
        'line': load_average_run('line'),
        'outward_spiral': load_average_run('outward_spiral'),
        'outward_spiral_varying_z': load_average_run('outward_spiral_varying_z'),
        'slalom': load_average_run('slalom'),
        'zig_zag_climb': load_average_run('zig_zag_climb'),
        'zig_zag_fall': load_average_run('zig_zag_fall')
    }

    controllers = {
        'ellipse': importlib.import_module('ellipse.edit_this'),
        'line': importlib.import_module('line.edit_this'),
        'outward_spiral': importlib.import_module('outward_spiral.edit_this'),
        'outward_spiral_varying_z': importlib.import_module('outward_spiral_varying_z.edit_this'),
        'slalom': importlib.import_module('slalom.edit_this'),
        'zig_zag_climb': importlib.import_module('zig_zag_climb.edit_this'),
        'zig_zag_fall': importlib.import_module('zig_zag_fall.edit_this'),
    }

    run(trajectories, controllers)
    