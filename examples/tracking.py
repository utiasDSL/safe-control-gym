"""A quadrotor trajectory tracking example.

Notes:
    Includes and uses PID control.

Run as:

    $ python3 tracking.py --overrides ./tracking.yaml

"""
import time
import pybullet as p
from functools import partial

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
    
    for i in range(3):
        # Start a timer.
        START = time.time()
        
        if i == 1:
            config.quadrotor_config['task_info']['trajectory_type'] = 'circle'
        elif i == 2:
            config.quadrotor_config['task_info']['trajectory_type'] = 'square'
                
        # Create controller.
        env_func = partial(make,
                           'quadrotor',
                           **config.quadrotor_config
                           )
        ctrl = make('pid',
                    env=env_func,
                    )
                    
        reference_traj = ctrl.reference

        # Plot trajectory.
        for i in range(0, reference_traj.shape[0], 10):
            p.addUserDebugLine(lineFromXYZ=[reference_traj[i-10,0], 0, reference_traj[i-10,2]],
                               lineToXYZ=[reference_traj[i,0], 0, reference_traj[i,2]],
                               lineColorRGB=[1, 0, 0],
                               physicsClientId=ctrl.env.PYB_CLIENT)

        # Run the experiment.
        results = ctrl.run(max_steps=max_steps)
                
        # Plot the experiment.
        for i in range(max_steps):
            # Step the environment and print all returned information.
            obs, reward, done, info, action = results['obs'][i], results['reward'][i], results['done'][i], results['info'][i], results['action'][i]
            
            # Print the last action and the information returned at each step.
            print(i, '-th step.')
            print(action, '\n', obs, '\n', reward, '\n', done, '\n', info, '\n')    

        ctrl.close()            

        elapsed_sec = time.time() - START
        print("\n{:d} iterations (@{:d}Hz) and {:d} episodes in {:.2f} seconds, i.e. {:.2f} steps/sec for a {:.2f}x speedup.\n"
              .format(max_steps, config.quadrotor_config.ctrl_freq, 1, elapsed_sec, max_steps/elapsed_sec, (max_steps*(1. / config.quadrotor_config.ctrl_freq))/elapsed_sec))


if __name__ == "__main__":
    main()