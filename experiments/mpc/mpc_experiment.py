"""A quadrotor trajectory tracking example.

Notes:
    Includes and uses PID control.

Run as:

    $ python3 ./pid_experiment.py --task quadrotor --algo pid --overrides ./config_pid_quadrotor.yaml

"""
import time
import pybullet as p
from functools import partial
import numpy as np

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

def main():
    """The main function creating, running, and closing an environment.

    """

    # Create an environment
    CONFIG_FACTORY = ConfigFactory()               
    config = CONFIG_FACTORY.merge()
    
    # Set max_steps and episode counter.
    max_steps = int(config.quadrotor_config['episode_len_sec']*config.quadrotor_config['ctrl_freq'])
    
    # Start a timer.
    START = time.time()
    
    # Create controller.
    env_func = partial(make,
                    config.task,
                    **config.quadrotor_config
                    )
    ctrl = make(config.algo,
                env_func,
                **config.algo_config
                )
                
    if config.quadrotor_config.task == 'traj_tracking':
        reference_traj = ctrl.reference

        # Plot trajectory.
        for i in range(0, reference_traj.shape[0], 10):
            p.addUserDebugLine(lineFromXYZ=[reference_traj[i-10,0], 0, reference_traj[i-10,2]],
                                lineToXYZ=[reference_traj[i,0], 0, reference_traj[i,2]],
                                lineColorRGB=[1, 0, 0],
                                physicsClientId=ctrl.env.PYB_CLIENT)

    # Run the experiment.
    results = ctrl.run(max_steps=max_steps)
    ctrl.close()

    N = len(results['obs']) - 1
    # Plot the experiment.
    for i in range(N):
        # Step the environment and print all returned information.
        obs, reward, done, info, action = results['obs'][i], results['reward'][i], results['done'][i], results['info'][i], results['action'][i]
        
        # Print the last action and the information returned at each step.
        print(i, '-th step.')
        print(action, '\n', obs, '\n', reward, '\n', done, '\n', info, '\n')

    # Calculate the maximum disturbance:
    D = results['obs'][0].shape[0]
    disturbances = np.zeros((N, D))
    for i in range(N):
            xkp1 = results['horizon_states'][i][:,1]
            obs = results['obs'][i + 1]
            disturbances[i,:] = obs - xkp1
    print('min disturbance: {}'.format(np.min(disturbances, axis=0)))            
    print('max disturbance: {}'.format(np.max(disturbances, axis=0)))
    print('mean disturbance: {}'.format(np.mean(disturbances, axis=0)))
    print('std dev disturbance: {}'.format(np.std(disturbances, axis=0)))
    wmin = np.mean(disturbances, axis=0) - 3 * np.std(disturbances, axis=0)
    wmax = np.mean(disturbances, axis=0) + 3 * np.std(disturbances, axis=0)
    print('wmin: {}'.format(wmin))
    print('wmax: {}'.format(wmax))


if __name__ == "__main__":
    main()