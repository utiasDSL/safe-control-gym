"""A quadrotor trajectory tracking example.

Notes:
    Includes and uses PID control.

Run as:

    $ python3 ./pid_experiment.py --task quadrotor --algo pid --overrides ./config_pid_quadrotor.yaml

"""
import time
import pickle
import numpy as np
from functools import partial
import matplotlib.pyplot as plt


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

    all_results = {'init_state': [], 'final_state': [], 'iters': [], 'violations': [], 'success': [], 'time': [], 'rmse': []}
                
    num_tests = 1
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

        violations = sum([results['info'][i]['constraint_violation'] for i in range(len(results['info']))])

        if config.task_config.task == 'stabilization':
            success = results['info'][-1]['goal_reached']
            mse = np.square(np.subtract(np.vstack(results['obs'])[1:, :], ctrl.env.X_GOAL)).mean() 
            rmse = mse**0.5
        else:
            success = bool(np.linalg.norm(results['obs'][-1][[0, 2]] - config.task_config.task_info.stabilization_goal) < config.task_config.task_info.stabilization_goal_tolerance)
            if np.vstack(results['obs'])[1:, :].shape == ctrl.env.X_GOAL.shape:
                mse = np.square(np.subtract(np.vstack(results['obs'])[1:, :], ctrl.env.X_GOAL)).mean() 
                rmse = mse**0.5
            else:
                rmse = float('inf')
                success = False

        result_obs = np.vstack(results['obs'])
        all_results['rmse'].append(rmse)
        all_results['init_state'].append(results['obs'][0])
        all_results['time'].append(elapsed_time)
        all_results['iters'].append(len(results['obs']))
        all_results['success'].append(success)
        all_results['violations'].append(violations)
        all_results['final_state'].append(result_obs[-1, :])

        fig, ax = plt.subplots()
        ax.plot(result_obs[:, 2], result_obs[:, 3], 'r--', label='Trajectory')
        ax.axvline(x=-0.2, color='k', lw=2, label='Limit')
        ax.scatter(result_obs[0, 2], result_obs[0, 3], color='g', marker='o', s=100, label='Initial State')
        ax.axvline(x=0.2, color='k', lw=2)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$\dot{\theta}$")
        ax.set_box_aspect(0.5)
        ax.legend(loc='upper right')
        plt.show()

        print('Success', success)
        print('Final state', result_obs[-1, :])
        print('Num Iters', len(results['obs']))
    
    ctrl.close()

    with open(f'results/{config.task_config.task}_{config.algo}.pkl', "wb") as f:
        pickle.dump(all_results, f)

    print("NUM SUCCESSES:", sum(all_results['success']))
    print("NUM VIOLATIONS:", sum(all_results['violations']))
    print("AVG ITERATIONS:", sum(all_results['iters'])/len(all_results['iters']))
    print("RMSE:", sum(all_results['rmse'])/len(all_results['rmse']))
    print("FAILED:", all_results['rmse'].count(float('inf')))


if __name__ == "__main__":
    main()