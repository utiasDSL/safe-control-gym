""" Utils for Optuna hyperparameter optimization. """

import optuna
from typing import Dict, Any

from safe_control_gym.hyperparameters.hpo_search_space import PPO_dict, SAC_dict, GPMPC_dict, iLQR_dict, iLQR_SF_dict
from safe_control_gym.hyperparameters.hpo_search_space import get_scale

def ppo_sampler(hps_dict: Dict[str, Any], trial: optuna.Trial, state_dim: int, action_dim: int) -> Dict[str, Any]:
    """Sampler for PPO hyperparameters.

    args:
        hps_dict: the dict of hyperparameters that will be optimized over
        trial: budget variable

    """

    # TODO: conditional hyperparameters

    # model args
    hidden_dim = trial.suggest_categorical('hidden_dim', PPO_dict['hidden_dim'])
    activation = trial.suggest_categorical('activation', PPO_dict['activation'])

    # loss args
    gamma = trial.suggest_categorical('gamma', PPO_dict['gamma'])
    # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    gae_lambda = trial.suggest_categorical('gae_lambda', PPO_dict['gae_lambda'])
    clip_param = trial.suggest_categorical('clip_param', PPO_dict['clip_param'])
    # Limit the KL divergence between updates
    target_kl = trial.suggest_float('target_kl', PPO_dict['target_kl'][0], PPO_dict['target_kl'][1])
    # Entropy coefficient for the loss calculation
    entropy_coef = trial.suggest_float('entropy_coef', PPO_dict['entropy_coef'][0], PPO_dict['entropy_coef'][1], log=True)

    # optim args
    opt_epochs = trial.suggest_categorical('opt_epochs', PPO_dict['opt_epochs'])
    mini_batch_size = trial.suggest_categorical('mini_batch_size', PPO_dict['mini_batch_size'])
    actor_lr = trial.suggest_float('actor_lr', PPO_dict['actor_lr'][0], PPO_dict['actor_lr'][1], log=True)
    critic_lr = trial.suggest_float('critic_lr', PPO_dict['critic_lr'][0], PPO_dict['critic_lr'][1], log=True)
    max_env_steps = trial.suggest_categorical('max_env_steps', PPO_dict['max_env_steps'])


    # cost parameters
    state_weight = [
        trial.suggest_float(f'rew_state_weight_{i}', PPO_dict['rew_state_weight'][0], PPO_dict['rew_state_weight'][1])
        for i in range(state_dim)
    ]
    action_weight = [
        trial.suggest_float(f'rew_action_weight_{i}', PPO_dict['rew_action_weight'][0], PPO_dict['rew_action_weight'][1])
        for i in range(action_dim)
    ]

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
        'max_env_steps': max_env_steps,
        'rew_state_weight': state_weight,
        'rew_action_weight': action_weight,
    }

    return hps_suggestion


def sac_sampler(hps_dict: Dict[str, Any], trial: optuna.Trial, state_dim: int, action_dim: int) -> Dict[str, Any]:
    """Sampler for SAC hyperparameters.

    args:
        hps_dict: the dict of hyperparameters that will be optimized over
        trial: budget variable

    """

    # TODO: conditional hyperparameters

    # model args
    hidden_dim = trial.suggest_categorical('hidden_dim', SAC_dict['hidden_dim'])
    activation = trial.suggest_categorical('activation', SAC_dict['activation'])

    # loss args
    gamma = trial.suggest_categorical('gamma', SAC_dict['gamma'])
    tau = trial.suggest_float('tau', SAC_dict['tau'][0], SAC_dict['tau'][1], log=False)

    # optim args
    train_interval = trial.suggest_categorical('train_interval', SAC_dict['train_interval'])
    train_batch_size = trial.suggest_categorical('train_batch_size', SAC_dict['train_batch_size'])
    actor_lr = trial.suggest_float('actor_lr', SAC_dict['actor_lr'][0], SAC_dict['actor_lr'][1], log=True)
    critic_lr = trial.suggest_float('critic_lr', SAC_dict['critic_lr'][0], SAC_dict['critic_lr'][1], log=True)
    entropy_lr = trial.suggest_float('entropy_lr', SAC_dict['entropy_lr'][0], SAC_dict['entropy_lr'][1], log=True)
    init_temperature = trial.suggest_float('init_temperature', SAC_dict['init_temperature'][0], SAC_dict['init_temperature'][1], log=False)

    max_env_steps = trial.suggest_categorical('max_env_steps', SAC_dict['max_env_steps'])
    warm_up_steps = trial.suggest_categorical('warm_up_steps', SAC_dict['warm_up_steps'])
    max_buffer_size = trial.suggest_categorical('max_buffer_size', SAC_dict['max_buffer_size'])

    # cost parameters
    state_weight = [
        trial.suggest_float(f'rew_state_weight_{i}', SAC_dict['rew_state_weight'][0], SAC_dict['rew_state_weight'][1])
        for i in range(state_dim)
    ]
    action_weight = [
        trial.suggest_float(f'rew_action_weight_{i}', SAC_dict['rew_action_weight'][0], SAC_dict['rew_action_weight'][1])
        for i in range(action_dim)
    ]

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
        'rew_state_weight': state_weight,
        'rew_action_weight': action_weight,
    }

    return hps_suggestion


def gpmpc_sampler(hps_dict: Dict[str, Any], trial: optuna.Trial, state_dim: int, action_dim: int) -> Dict[str, Any]:
    """Sampler for PPO hyperparameters.

    args:
        hps_dict: the dict of hyperparameters that will be optimized over
        trial: budget variable

    """

    horizon = trial.suggest_categorical('horizon', GPMPC_dict['horizon'])
    kernel = trial.suggest_categorical('kernel', GPMPC_dict['kernel'])
    n_ind_points = trial.suggest_categorical('n_ind_points', GPMPC_dict['n_ind_points'])
    num_epochs = trial.suggest_categorical('num_epochs', GPMPC_dict['num_epochs'])
    num_samples = trial.suggest_categorical('num_samples', GPMPC_dict['num_samples'])

    # get dimensions of the dynamics
    d = len(hps_dict['learning_rate'])
    assert d == len(hps_dict['optimization_iterations']), 'The number of optimization iterations must be the same as the number of learning rates.'

    # use same setting for all dimensions for simplicity
    optimization_iterations, learning_rate = [], []

    optimization_iterations = d * [trial.suggest_categorical('optimization_iterations', GPMPC_dict['optimization_iterations'])]
    learning_rate = d * [trial.suggest_float('learning_rate', GPMPC_dict['learning_rate'][0], GPMPC_dict['learning_rate'][1], log=True)]

    # objective
    state_weight = [
        trial.suggest_float(f'q_mpc_{i}', GPMPC_dict['q_mpc'][0], GPMPC_dict['q_mpc'][1])
        for i in range(state_dim)
    ]
    action_weight = [
        trial.suggest_float(f'r_mpc_{i}', GPMPC_dict['r_mpc'][0], GPMPC_dict['r_mpc'][1])
        for i in range(action_dim)
    ]
    
    hps_suggestion = {
        'horizon': horizon,
        'kernel': kernel,
        'n_ind_points': n_ind_points,
        'num_epochs': num_epochs,
        'num_samples': num_samples,
        'optimization_iterations': optimization_iterations,
        'learning_rate': learning_rate,
        'q_mpc': state_weight,
        'r_mpc': action_weight,
    }

    return hps_suggestion

def ilqr_sampler(hps_dict: Dict[str, Any], trial: optuna.Trial, state_dim: int, action_dim: int) -> Dict[str, Any]:
    """Sampler for iLQR hyperparameters.

    args:
        hps_dict: the dict of hyperparameters that will be optimized over
        trial: budget variable

    """

    max_iterations = trial.suggest_categorical('max_iterations', iLQR_dict['max_iterations']['values'])
    lamb_factor = trial.suggest_categorical('lamb_factor', iLQR_dict['lamb_factor']['values'])
    lamb_max = trial.suggest_categorical('lamb_max', iLQR_dict['lamb_max']['values'])
    epsilon = trial.suggest_categorical('epsilon', iLQR_dict['epsilon']['values'])

    # cost parameters
    state_weight = [
        trial.suggest_float(f'q_lqr_{i}', iLQR_dict['q_lqr']['values'][0], iLQR_dict['q_lqr']['values'][1], log=get_scale(iLQR_dict['q_lqr']))
        for i in range(state_dim)
    ]
    action_weight = [
        trial.suggest_float(f'r_lqr_{i}', iLQR_dict['r_lqr']['values'][0], iLQR_dict['r_lqr']['values'][1], log=get_scale(iLQR_dict['r_lqr']))
        for i in range(action_dim)
    ]

    hps_suggestion = {
        'max_iterations': max_iterations,
        'lamb_factor': lamb_factor,
        'lamb_max': lamb_max,
        'epsilon': epsilon,
        'q_lqr': state_weight,
        'r_lqr': action_weight,
    }

    return hps_suggestion

def ilqr_sf_sampler(hps_dict: Dict[str, Any], trial: optuna.Trial, state_dim: int, action_dim: int) -> Dict[str, Any]:
    """Sampler for iLQR hyperparameters with safety filter.

    args:
        hps_dict: the dict of hyperparameters that will be optimized over
        trial: budget variable

    """

    max_iterations = trial.suggest_categorical('max_iterations', iLQR_SF_dict['max_iterations'])
    lamb_factor = trial.suggest_categorical('lamb_factor', iLQR_SF_dict['lamb_factor'])
    lamb_max = trial.suggest_categorical('lamb_max', iLQR_SF_dict['lamb_max'])
    epsilon = trial.suggest_categorical('epsilon', iLQR_SF_dict['epsilon'])

    # safety filter
    horizon = trial.suggest_categorical('horizon', iLQR_SF_dict['horizon'])
    n_samples = trial.suggest_categorical('n_samples', iLQR_SF_dict['n_samples'])

    # cost parameters
    state_weight = [
        trial.suggest_float(f'q_lqr_{i}', iLQR_SF_dict['q_lqr'][0], iLQR_SF_dict['q_lqr'][1])
        for i in range(state_dim)
    ]
    action_weight = [
        trial.suggest_float(f'r_lqr_{i}', iLQR_SF_dict['r_lqr'][0], iLQR_SF_dict['r_lqr'][1])
        for i in range(action_dim)
    ]

    # safety filter
    tau = trial.suggest_float('tau', iLQR_SF_dict['tau'][0], iLQR_SF_dict['tau'][1], log=False)
    sf_state_weight = [
        trial.suggest_float(f'q_lin_{i}', iLQR_SF_dict['q_lin'][0], iLQR_SF_dict['q_lin'][1])
        for i in range(state_dim)
    ]
    sf_action_weight = [
        trial.suggest_float(f'r_lin_{i}', iLQR_SF_dict['r_lin'][0], iLQR_SF_dict['r_lin'][1])
        for i in range(action_dim)
    ]
    hps_suggestion = {
        'max_iterations': max_iterations,
        'lamb_factor': lamb_factor,
        'lamb_max': lamb_max,
        'epsilon': epsilon,
        'horizon': horizon,
        'n_samples': n_samples,
        'state_weight': state_weight,
        'action_weight': action_weight,
        'tau': tau,
        'q_lin': sf_state_weight,
        'r_lin': sf_action_weight,
    }

    return hps_suggestion

HYPERPARAMS_SAMPLER = {
    'ppo': ppo_sampler,
    'sac': sac_sampler,
    'gp_mpc': gpmpc_sampler,
    'gpmpc_acados': gpmpc_sampler,
    'ilqr': ilqr_sampler,
    'ilqr_sf': ilqr_sf_sampler,
}