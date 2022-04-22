"""A quadrotor trajectory tracking example.

Notes:
    Includes and uses PID control.

Run as:

    $ python3 ./pid_experiment.py --task quadrotor --algo pid --overrides ./config_pid_quadrotor.yaml

"""
import time
import pickle
import numpy as np
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
    max_steps = int(config.task_config['episode_len_sec']*config.task_config['ctrl_freq'])
    
    # Create controller.
    env_func = partial(make,
                    config.task,
                    **config.task_config
                    )
    ctrl = make(config.algo,
                env_func,
                **config.algo_config
                )

    train_env = env_func(gui=False, randomized_init=True, init_state=None, disturbances=None) # training without disturbances
    # train_env = env_func(gui=False, randomized_init=True, init_state=None) # training with disturbances
    ctrl.learn(env=train_env)

    all_results = {'init_state': [], 'final_state': [], 'iters': [], 'violations': [], 'success': [], 'time': []}
                
    num_tests = 100
    for iter in range(num_tests):
        # Run the experiment.
        START = time.time()
        if config.algo == 'lqr':
            results = ctrl.run()
            results = results['ep_results'][0]
        else:
            results = ctrl.run(max_steps=max_steps)

        elapsed_time = time.time() - START
        print('Iteration #', iter)
        ctrl.reset()

        if config.algo == 'tube_mpc':
            print('Original Violations: ', sum(results['original_violations']))
            print('Tightened Violations: ', sum(results['violations']))
            violations = sum(results['original_violations'])
        else:
            violations = 0
            upper_state = np.array(config.task_config.constraints[0].upper_bounds)
            lower_state = np.array(config.task_config.constraints[0].lower_bounds)
            upper_input = np.array(config.task_config.constraints[1].upper_bounds)
            lower_input = np.array(config.task_config.constraints[1].lower_bounds)
            for i in range(len(results['obs'])-1):
                violations += not np.all(np.logical_and(lower_state < results['obs'][i+1], results['obs'][i+1] < upper_state))
                violations += not np.all(np.logical_and(lower_input < results['action'][i], results['action'][i] < upper_input))

            print('Violations: ', violations)

        success = results['info'][-1]['goal_reached']
        all_results['init_state'].append(results['obs'][0])
        all_results['time'].append(elapsed_time)
        all_results['iters'].append(len(results['obs']))
        all_results['success'].append(success)
        all_results['violations'].append(violations)
        all_results['final_state'].append(np.vstack(results['obs'])[-1, :])

        print('Success', success)
        print('Final state', np.vstack(results['obs'])[-1, :])
        print('Num Iters', len(results['obs']))
    
    ctrl.close()

    with open('results_tube.pkl', "wb") as f:
        pickle.dump(all_results, f)

    print("NUM SUCCESSES:", sum(all_results['success']))
    print("NUM VIOLATIONS:", sum(all_results['violations']))
    print("AVG ITERATIONS:", sum(all_results['iters'])/len(all_results['iters']))


if __name__ == "__main__":
    main()