'''A LQR and iLQR example. '''

import os
import pickle
from functools import partial
from collections import defaultdict

from safe_control_gym.experiment import Experiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


def run(gui=True, training=True, n_episodes=10, n_steps=None, save_data=True):
    '''The main function creating, running, and closing an environment. '''

    # Create an environment
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()

    # Create controller.
    env_func = partial(make,
                       config.task,
                       **config.task_config
                       )
    random_env = env_func(gui=False)
    ctrl = make(config.algo,
                env_func,
                **config.algo_config
                )

    all_trajs = defaultdict(list)
    n_episodes = 1 if n_episodes is None else n_episodes

    # Run the experiment.
    for _ in range(n_episodes):
        # Get initial state and create environments
        init_state, _ = random_env.reset()
        static_env = env_func(gui=gui, randomized_init=False, init_state=init_state)
        static_train_env = env_func(gui=False, randomized_init=False, init_state=init_state)

        # Create experiment, train, and run evaluation
        experiment = Experiment(env=static_env, ctrl=ctrl, train_env=static_train_env)
        if training:
            experiment.launch_training()
        if n_steps is None:
            trajs_data, _ = experiment.run_evaluation(training=True, n_episodes=1)
        else:
            trajs_data, _ = experiment.run_evaluation(training=True, n_steps=n_steps)

        # Close environments
        static_env.close()
        static_train_env.close()

        # Merge in new trajectory data
        for key in trajs_data.keys():
            all_trajs[key] += trajs_data[key]

    random_env.close()
    metrics = experiment.compute_metrics(all_trajs)
    all_trajs = dict(all_trajs)

    if save_data:
        results = {'trajs_data': all_trajs, 'metrics': metrics}
        path_dir = os.path.dirname('./temp-data/')
        os.makedirs(path_dir, exist_ok=True)
        with open(f'./temp-data/{config.algo}_data_{config.task_config.task}.pkl', 'wb') as file:
            pickle.dump(results, file)

    print('FINAL METRICS - ' + ', '.join([f'{key}: {value}' for key, value in metrics.items()]))


if __name__ == '__main__':
    run()
