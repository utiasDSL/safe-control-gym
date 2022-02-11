"""A quadrotor trajectory tracking example.

Notes:
    Includes and uses PID control.

Run as:

    $ python3 tracking.py --task quadrotor --algo pid --overrides ./tracking.yaml

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
    
    # Set iterations and episode counter.
    ITERATIONS = int(config.quadrotor_config['episode_len_sec']*config.quadrotor_config['ctrl_freq'])
    
    for i in range(3):
        # Start a timer.
        START = time.time()
        
        if i == 1:
            config.quadrotor_config['task_info']['trajectory_type'] = 'circle'
        elif i == 2:
            config.quadrotor_config['task_info']['trajectory_type'] = 'square'
                
        # Create controller.
        env_func = partial(make,
                       config.task,
                       **config.quadrotor_config
                       )
        ctrl = make(config.algo,
                    env_func,
                    )
                    
        reference_traj = ctrl.reference

        # Plot trajectory.
        for i in range(0, reference_traj.shape[0], 10):
            p.addUserDebugLine(lineFromXYZ=[reference_traj[i-10,0], 0, reference_traj[i-10,2]],
                                lineToXYZ=[reference_traj[i,0], 0, reference_traj[i,2]],
                                lineColorRGB=[1, 0, 0],
                                physicsClientId=ctrl.env.PYB_CLIENT)

        # Run the experiment.
        results = ctrl.run( iterations=ITERATIONS)
                
        # Plot the experiment.
        for i in range(ITERATIONS):
            # Step the environment and print all returned information.
            obs, reward, done, info, action = results['obs'][i], results['reward'][i], results['done'][i], results['info'][i], results['action'][i]
            
            # Print the last action and the information returned at each step.
            print(i, '-th step.')
            print(action, '\n', obs, '\n', reward, '\n', done, '\n', info, '\n')    

        ctrl.close()            

        elapsed_sec = time.time() - START
        print("\n{:d} iterations (@{:d}Hz) and {:d} episodes in {:.2f} seconds, i.e. {:.2f} steps/sec for a {:.2f}x speedup.\n"
              .format(ITERATIONS, config.quadrotor_config.ctrl_freq, 1, elapsed_sec, ITERATIONS/elapsed_sec, (ITERATIONS*(1. / config.quadrotor_config.ctrl_freq))/elapsed_sec))


if __name__ == "__main__":
    main()