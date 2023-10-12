'''RARL Utils.'''

import numpy as np

from safe_control_gym.envs.env_wrappers.vectorized_env.vec_env_utils import _flatten_obs, _unflatten_obs


def split_obs_by_adversary(obs, indices_splits):
    '''Splits and groups the observations by adversaries.

    Args:
        obs (np.array | dict): batch of observations, (N,*obs_dim).
        indices_splits (np.array): shape (n,),
            first occurence indices of the obs batch for each adversary.

    Returns:
        list: groups of observations for adversaries.
    '''
    obs_list = _unflatten_obs(obs)
    start = indices_splits
    end = np.concatenate([indices_splits[1:], [len(obs_list)]])
    obs_groups = [_flatten_obs(obs_list[i:j]) for i, j in zip(start, end)]
    return obs_groups
