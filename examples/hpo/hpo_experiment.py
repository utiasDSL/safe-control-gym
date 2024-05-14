'''Template hyperparameter optimization/hyperparameter evaluation script.'''
import os
from functools import partial

import yaml

from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.hyperparameters.hpo import HPO
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import set_device_from_config, set_dir_from_config, set_seed_from_config


def hpo(config):
    '''Hyperparameter optimization.

    Usage:
        * to start HPO, use with `--func hpo`.
    '''

    # Experiment setup.
    if config.hpo_config.hpo:
        set_dir_from_config(config)
    set_seed_from_config(config)
    set_device_from_config(config)

    # HPO
    hpo = HPO(config.algo,
              config.task,
              config.sampler,
              config.load_study,
              config.output_dir,
              config.task_config,
              config.hpo_config,
              **config.algo_config)

    if config.hpo_config.hpo:
        hpo.hyperparameter_optimization()
        print('Hyperparameter optimization done.')


def train(config):
    '''Training for a given set of hyperparameters.

    Usage:
        * to start training, use with `--func train`.
    '''
    # Override algo_config with given yaml file
    if config.opt_hps == '':
        # if no opt_hps file is given
        pass
    else:
        # if opt_hps file is given
        with open(config.opt_hps, 'r') as f:
            opt_hps = yaml.load(f, Loader=yaml.FullLoader)
        for hp in opt_hps:
            if isinstance(config.algo_config[hp], list) and not isinstance(opt_hps[hp], list):
                config.algo_config[hp] = [opt_hps[hp]] * len(config.algo_config[hp])
            else:
                config.algo_config[hp] = opt_hps[hp]
    # Experiment setup.
    set_dir_from_config(config)
    set_seed_from_config(config)
    set_device_from_config(config)

    # Define function to create task/env.
    env_func = partial(make, config.task, output_dir=config.output_dir, **config.task_config)
    # Create the controller/control_agent.
    # Note:
    # eval_env will take config.seed * 111 as its seed
    # env will take config.seed as its seed
    control_agent = make(config.algo,
                         env_func,
                         training=True,
                         checkpoint_path=os.path.join(config.output_dir, 'model_latest.pt'),
                         output_dir=config.output_dir,
                         use_gpu=config.use_gpu,
                         seed=config.seed,
                         **config.algo_config)
    control_agent.reset()

    eval_env = env_func(seed=config.seed * 111)
    # Run experiment
    experiment = BaseExperiment(eval_env, control_agent)
    experiment.launch_training()
    results, metrics = experiment.run_evaluation(n_episodes=1, n_steps=None, done_on_max_steps=True)
    control_agent.close()

    return eval_env.X_GOAL, results, metrics


MAIN_FUNCS = {'hpo': hpo, 'train': train}


if __name__ == '__main__':
    # Make config.
    fac = ConfigFactory()
    fac.add_argument('--func', type=str, default='train', help='main function to run.')
    fac.add_argument('--opt_hps', type=str, default='', help='yaml file as a result of HPO.')
    fac.add_argument('--load_study', type=bool, default=False, help='whether to load study from a previous HPO.')
    fac.add_argument('--sampler', type=str, default='TPESampler', help='which sampler to use in HPO.')
    # merge config
    config = fac.merge()

    # Execute.
    func = MAIN_FUNCS.get(config.func, None)
    if func is None:
        raise Exception(f'Main function {config.func} not supported.')
    func(config)
