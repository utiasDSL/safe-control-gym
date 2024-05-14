'''Sampler for hyperparameters for different algorithms

Reference:
    * stable baselines3 https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/hyperparams_opt.py
'''

from typing import Any, Dict

import optuna

# define the categorical choice or real interval for each hyperparameter
PPO_dict = {
    'categorical': {
        'hidden_dim': [8, 16, 32, 64, 128, 256],
        'activation': ['tanh', 'relu', 'leaky_relu'],
        'gamma': [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999],
        'gae_lambda': [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0],
        'clip_param': [0.1, 0.2, 0.3, 0.4],
        'opt_epochs': [1, 5, 10, 20],
        'mini_batch_size': [32, 64, 128],
        'rollout_steps': [50, 100, 150, 200],
        'max_env_steps': [30000, 72000, 216000],  # to make sure having the checkpoint at these steps [30000, 72000, 216000]
    },
    'float': {  # note that in float type, you must specify the upper and lower bound
        'target_kl': [0.00000001, 0.8],
        'entropy_coef': [0.00000001, 0.1],
        'actor_lr': [1e-5, 1],
        'critic_lr': [1e-5, 1],
    }
}
SAC_dict = {
    'categorical': {
        'hidden_dim': [32, 64, 128, 256, 512],
        'activation': ['tanh', 'relu', 'leaky_relu'],
        'gamma': [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999],
        'train_interval': [10, 100, 1000],  # should be divisible by max_env_steps
        'train_batch_size': [32, 64, 128, 256, 512],
        'max_env_steps': [30000, 72000, 216000],  # to make sure having the checkpoint at these steps [30000, 72000, 216000]
        'warm_up_steps': [500, 1000, 2000, 4000],
    },
    'float': {  # note that in float type, you must specify the upper and lower bound
        'tau': [0.005, 1.0],
        'actor_lr': [1e-5, 1],
        'critic_lr': [1e-5, 1],
    }
}

GPMPC_dict = {
    'categorical': {
        'horizon': [10, 15, 20, 25, 30, 35],
        'kernel': ['Matern', 'RBF'],
        'n_ind_points': [30, 40, 50],  # number should lower 0.8 * MIN(num_samples) if 0,2 is test_data_ratio
        'num_epochs': [4, 5, 6, 7, 8],
        'num_samples': [70, 75, 80, 85],
        'optimization_iterations': [2800, 3000, 3200],  # to make sure having the same checkpoint at these steps [30000, 54000, 72000]
    },
    'float': {  # note that in float type, you must specify the upper and lower bound
        'learning_rate': [5e-4, 0.5],
    }
}


def ppo_sampler(hps_dict: Dict[str, Any], trial: optuna.Trial) -> Dict[str, Any]:
    '''Sampler for PPO hyperparameters.

    Args:
        hps_dict: the dict of hyperparameters that will be optimized over
        trial: budget variable
    '''

    # TODO: conditional hyperparameters

    # model args
    hidden_dim = trial.suggest_categorical('hidden_dim', PPO_dict['categorical']['hidden_dim'])
    activation = trial.suggest_categorical('activation', PPO_dict['categorical']['activation'])

    # loss args
    gamma = trial.suggest_categorical('gamma', PPO_dict['categorical']['gamma'])
    # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    gae_lambda = trial.suggest_categorical('gae_lambda', PPO_dict['categorical']['gae_lambda'])
    clip_param = trial.suggest_categorical('clip_param', PPO_dict['categorical']['clip_param'])
    # Limit the KL divergence between updates
    target_kl = trial.suggest_float('target_kl', PPO_dict['float']['target_kl'][0], PPO_dict['float']['target_kl'][1], log=True)
    # Entropy coefficient for the loss calculation
    entropy_coef = trial.suggest_float('entropy_coef', PPO_dict['float']['entropy_coef'][0], PPO_dict['float']['entropy_coef'][1], log=True)

    # optim args
    opt_epochs = trial.suggest_categorical('opt_epochs', PPO_dict['categorical']['opt_epochs'])
    mini_batch_size = trial.suggest_categorical('mini_batch_size', PPO_dict['categorical']['mini_batch_size'])
    actor_lr = trial.suggest_float('actor_lr', PPO_dict['float']['actor_lr'][0], PPO_dict['float']['actor_lr'][1], log=True)
    critic_lr = trial.suggest_float('critic_lr', PPO_dict['float']['critic_lr'][0], PPO_dict['float']['critic_lr'][1], log=True)
    # The maximum value for the gradient clipping
    # max_grad_norm = trial.suggest_categorical('max_grad_norm', [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])

    # The number of steps to run for each environment per update
    # Note: rollout_steps * n_envs should be greater than mini_batch_size
    # The value is set in this way for the sake of evluation and checkpoint
    # e.g. total_steps will be incremented by 4 *[50, 100, 150, 250] = [200, 400, 600, 1000]
    # then eval_inverval can be set to 6000 if we want to capture learning efficiency and intermediate performance
    rollout_steps = trial.suggest_categorical('rollout_steps', PPO_dict['categorical']['rollout_steps'])
    max_env_steps = trial.suggest_categorical('max_env_steps', PPO_dict['categorical']['max_env_steps'])

    # Orthogonal initialization
    # ortho_init = False
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])

    hps_suggestion = {
        'hidden_dim': hidden_dim,
        'activation': activation,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'clip_param': clip_param,
        'target_kl': target_kl,
        'entropy_coef': entropy_coef,
        'opt_epochs': opt_epochs,
        'mini_batch_size': mini_batch_size,
        'actor_lr': actor_lr,
        'critic_lr': critic_lr,
        # 'max_grad_norm': max_grad_norm, (currently not implemented in PPO controller)
        'max_env_steps': max_env_steps,
        'rollout_steps': rollout_steps,
    }

    assert len(hps_suggestion) == len(hps_dict), ValueError('We are optimizing over different number of HPs as you listed.')

    return hps_suggestion


def sac_sampler(hps_dict: Dict[str, Any], trial: optuna.Trial) -> Dict[str, Any]:
    '''Sampler for SAC hyperparameters.

    Args:
        hps_dict: the dict of hyperparameters that will be optimized over
        trial: budget variable
    '''

    # TODO: conditional hyperparameters

    # model args
    hidden_dim = trial.suggest_categorical('hidden_dim', SAC_dict['categorical']['hidden_dim'])
    activation = trial.suggest_categorical('activation', SAC_dict['categorical']['activation'])

    # loss args
    gamma = trial.suggest_categorical('gamma', SAC_dict['categorical']['gamma'])
    tau = trial.suggest_float('tau', SAC_dict['float']['tau'][0], SAC_dict['float']['tau'][1], log=False)

    # optim args
    train_interval = trial.suggest_categorical('train_interval', SAC_dict['categorical']['train_interval'])
    train_batch_size = trial.suggest_categorical('train_batch_size', SAC_dict['categorical']['train_batch_size'])
    actor_lr = trial.suggest_float('actor_lr', SAC_dict['float']['actor_lr'][0], SAC_dict['float']['actor_lr'][1], log=True)
    critic_lr = trial.suggest_float('critic_lr', SAC_dict['float']['critic_lr'][0], SAC_dict['float']['critic_lr'][1], log=True)

    max_env_steps = trial.suggest_categorical('max_env_steps', SAC_dict['categorical']['max_env_steps'])
    warm_up_steps = trial.suggest_categorical('warm_up_steps', SAC_dict['categorical']['warm_up_steps'])

    hps_suggestion = {
        'hidden_dim': hidden_dim,
        'activation': activation,
        'gamma': gamma,
        'tau': tau,
        'train_interval': train_interval,
        'train_batch_size': train_batch_size,
        'actor_lr': actor_lr,
        'critic_lr': critic_lr,
        'max_env_steps': max_env_steps,
        'warm_up_steps': warm_up_steps,
    }

    assert len(hps_suggestion) == len(hps_dict), ValueError('We are optimizing over different number of HPs as you listed.')

    return hps_suggestion


def gpmpc_sampler(hps_dict: Dict[str, Any], trial: optuna.Trial) -> Dict[str, Any]:
    '''Sampler for PPO hyperparameters.

    Args:
        hps_dict: the dict of hyperparameters that will be optimized over
        trial: budget variable
    '''

    horizon = trial.suggest_categorical('horizon', GPMPC_dict['categorical']['horizon'])
    kernel = trial.suggest_categorical('kernel', GPMPC_dict['categorical']['kernel'])
    n_ind_points = trial.suggest_categorical('n_ind_points', GPMPC_dict['categorical']['n_ind_points'])
    num_epochs = trial.suggest_categorical('num_epochs', GPMPC_dict['categorical']['num_epochs'])
    num_samples = trial.suggest_categorical('num_samples', GPMPC_dict['categorical']['num_samples'])

    # get dimensions of the dynamics
    d = len(hps_dict['learning_rate'])
    assert d == len(hps_dict['optimization_iterations']), 'The number of optimization iterations must be the same as the number of learning rates.'

    # use same setting for all dimensions for simplicity
    optimization_iterations, learning_rate = [], []

    optimization_iterations = d * [trial.suggest_categorical('optimization_iterations', GPMPC_dict['categorical']['optimization_iterations'])]
    learning_rate = d * [trial.suggest_float('learning_rate', GPMPC_dict['float']['learning_rate'][0], GPMPC_dict['float']['learning_rate'][1], log=True)]

    hps_suggestion = {
        'horizon': horizon,
        'kernel': kernel,
        'n_ind_points': n_ind_points,
        'num_epochs': num_epochs,
        'num_samples': num_samples,
        'optimization_iterations': optimization_iterations,
        'learning_rate': learning_rate,
    }

    assert len(hps_suggestion) == len(hps_dict), ValueError('We are optimizing over different number of HPs as you listed.')

    return hps_suggestion


HYPERPARAMS_SAMPLER = {
    'ppo': ppo_sampler,
    'sac': sac_sampler,
    'gp_mpc': gpmpc_sampler,
}

HYPERPARAMS_DICT = {
    'ppo': PPO_dict,
    'sac': SAC_dict,
    'gp_mpc': GPMPC_dict,
}
