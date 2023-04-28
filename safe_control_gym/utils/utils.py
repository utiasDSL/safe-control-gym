'''Miscellaneous utility functions.'''

import argparse
import datetime
import json
import os
import random
import subprocess
import sys

import gymnasium as gym
import munch
import yaml
import imageio
import numpy as np
import torch


def mkdirs(*paths):
    '''Makes a list of directories.'''

    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def eval_token(token):
    '''Converts string token to int, float or str.'''
    if token.isnumeric():
        return int(token)
    try:
        return float(token)
    except TypeError:
        return token


def read_file(file_path, sep=','):
    '''Loads content from a file (json, yaml, csv, txt).

    For json & yaml files returns a dict.
    Ror csv & txt returns list of lines.
    '''
    if len(file_path) < 1 or not os.path.exists(file_path):
        return None
    # load file
    f = open(file_path, 'r')
    if 'json' in file_path:
        data = json.load(f)
    elif 'yaml' in file_path:
        data = yaml.load(f, Loader=yaml.FullLoader)
    else:
        sep = sep if 'csv' in file_path else ' '
        data = []
        for line in f.readlines():
            line_post = [eval_token(t) for t in line.strip().split(sep)]
            # if only sinlge item in line
            if len(line_post) == 1:
                line_post = line_post[0]
            if len(line_post) > 0:
                data.append(line_post)
    f.close()
    return data


def merge_dict(source_dict, update_dict):
    '''Merges updates into source recursively.'''
    for k, v in update_dict.items():
        if k in source_dict and isinstance(source_dict[k], dict) and isinstance(
                v, dict):
            merge_dict(source_dict[k], v)
        else:
            source_dict[k] = v


def get_time():
    '''Gets current timestamp (as string).'''
    start_time = datetime.datetime.now()
    time = str(start_time.strftime('%Y_%m_%d-%X'))
    return time


def get_random_state():
    '''Snapshots the random state at any moment.'''
    return {
        'random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state()
    }


def set_random_state(state_dict):
    '''Resets the random state for experiment restore.'''
    random.setstate(state_dict['random'])
    np.random.set_state(state_dict['numpy'])
    torch.torch.set_rng_state(state_dict['torch'])


def set_seed(seed, cuda=False):
    '''General seeding function for reproducibility.'''
    assert seed is not None, 'Error in set_seed(...), provided seed not valid'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_dir_from_config(config):
    '''Creates a output folder for experiment (and save config files).

    Naming format: {root (e.g. results)}/{tag (exp id)}/{seed}_{timestamp}_{git commit id}
    '''
    # Make run folder (of a seed run for an experiment)
    seed = str(config.seed) if config.seed is not None else '-'
    timestamp = str(datetime.datetime.now().strftime('%b-%d-%H-%M-%S'))
    try:
        commit_id = subprocess.check_output(
            ['git', 'describe', '--tags', '--always']
        ).decode('utf-8').strip()
        commit_id = str(commit_id)
    except BaseException:
        commit_id = '-'
    run_dir = f'seed{seed}_{timestamp}_{commit_id}'
    # Make output folder.
    config.output_dir = os.path.join(config.output_dir, config.tag, run_dir)
    mkdirs(config.output_dir)
    # Save config.
    with open(os.path.join(config.output_dir, 'config.yaml'), 'w') as file:
        yaml.dump(munch.unmunchify(config), file, default_flow_style=False)
    # Save command.
    with open(os.path.join(config.output_dir, 'cmd.txt'), 'a') as file:
        file.write(' '.join(sys.argv) + '\n')


def set_seed_from_config(config):
    '''Sets seed, only set if seed is provided.'''
    seed = config.seed
    if seed is not None:
        set_seed(seed, cuda=config.use_gpu)


def set_device_from_config(config):
    '''Sets device, using GPU is set to `cuda` for now, no specific GPU yet.'''
    use_cuda = config.use_gpu and torch.cuda.is_available()
    config.device = 'cuda' if use_cuda else 'cpu'


def save_video(name, frames, fps=20):
    '''Convert list of frames (H,W,C) to a video.

    Args:
        name (str): path name to save the video.
        frames (list): frames of the video as list of np.arrays.
        fps (int, optional): frames per second.
    '''
    assert '.gif' in name or '.mp4' in name, 'invalid video name'
    vid_kwargs = {'fps': fps}
    h, w, c = frames[0].shape
    video = np.stack(frames, 0).astype(np.uint8).reshape(-1, h, w, c)
    imageio.mimsave(name, video, **vid_kwargs)


def str2bool(val):
    '''Converts a string into a boolean.

    Args:
        val (str|bool): Input value (possibly string) to interpret as boolean.

    Returns:
        bool: Interpretation of `val` as True or False.
    '''
    if isinstance(val, bool):
        return val
    elif val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('[ERROR] in str2bool(), a Boolean value is expected')


def unwrap_wrapper(env, wrapper_class):
    '''Retrieve a ``VecEnvWrapper`` object by recursively searching.'''
    env_tmp = env
    while isinstance(env_tmp, gym.Wrapper):
        if isinstance(env_tmp, wrapper_class):
            return env_tmp
        env_tmp = env_tmp.env
    return None


def is_wrapped(env, wrapper_class):
    '''Check if a given environment has been wrapped with a given wrapper.'''
    return unwrap_wrapper(env, wrapper_class) is not None
