'''A PID example on a quadrotor. '''

import os
import pickle
from functools import partial

from safe_control_gym.experiment import Experiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


def run(gui=True, n_episodes=2, n_steps=None, save_data=True):
    '''The main function running PID experiments.

    Args:
        gui (bool): Whether to display the gui and plot graphs.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): The total number of steps to execute.
        save_data (bool): Whether to save the collected experiment data.
    '''

    # Create an environment
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()

    # Create controller.
    env_func = partial(make,
                    config.task,
                    **config.task_config
                    )
    env = env_func(gui=gui)
    ctrl = make(config.algo,
                env_func,
                )

    # Run the experiment.
    experiment = Experiment(env, ctrl)
    trajs_data, metrics = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps)
    experiment.close()

    if save_data:
        results = {'trajs_data': trajs_data, 'metrics': metrics}
        path_dir = os.path.dirname('./temp-data/')
        os.makedirs(path_dir, exist_ok=True)
        with open(f'./temp-data/pid_data_{config.task_config.task}.pkl', 'wb') as file:
            pickle.dump(results, file)

    iterations = len(trajs_data['action'][0])
    for i in range(iterations):
        # Step the environment and print all returned information.
        obs, reward, done, info, action = trajs_data['obs'][0][i], trajs_data['reward'][0][i], trajs_data['done'][0][i], trajs_data['info'][0][i], trajs_data['action'][0][i]

        # Print the last action and the information returned at each step.
        print(i, '-th step.')
        print(action, '\n', obs, '\n', reward, '\n', done, '\n', info, '\n')

    elapsed_sec = trajs_data['timestamp'][0][-1] - trajs_data['timestamp'][0][0]
    print('\n{:d} iterations (@{:d}Hz) and {:d} episodes in {:.2f} seconds, i.e. {:.2f} steps/sec for a {:.2f}x speedup.\n'
            .format(iterations, config.task_config.ctrl_freq, 1, elapsed_sec, iterations/elapsed_sec, (iterations*(1. / config.task_config.ctrl_freq))/elapsed_sec))


if __name__ == '__main__':
    run()
