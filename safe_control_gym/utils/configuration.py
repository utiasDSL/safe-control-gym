'''Configuration utilities.'''

import argparse
import os
import warnings

import munch
from dict_deep import deep_set

from safe_control_gym.utils.registration import get_config
from safe_control_gym.utils.utils import merge_dict, read_file


class ConfigFactory:
    '''Manager class that's in charge of experiment configs.'''

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Benchmark')
        self.add_arguments()
        self.base_dict = dict(
            tag='temp',
            seed=None,
            use_gpu=False,
            output_dir='results',
            restore=None,
        )

    def add_argument(self, *args, **kwargs):
        '''Extends to new arguments.'''
        self.parser.add_argument(*args, **kwargs)

    def add_arguments(self):
        '''Registers base arguments (for experiment bookkeeping).'''
        self.add_argument('--tag', type=str, help='id of the experiment')
        self.add_argument('--seed', type=int, help='random seed, default is no seed/None')
        # self.add_argument('--device', type=str, help='cpu or cuda(gpu)')
        self.add_argument('--use_gpu', action='store_true', help='added to use gpu (if available)')
        self.add_argument('--output_dir', type=str, help='output saving folder')
        self.add_argument('--restore', type=str, help='folder to reload from')
        # Need to explicitly provide from command line (if training for the 1st time).
        self.add_argument('--algo', type=str, help='algorithm/controller')
        self.add_argument('--task', type=str, help='task/environment')
        self.add_argument('--safety_filter', type=str, help='safety filter')
        self.add_argument('--overrides',
                          nargs='+',
                          type=str,
                          help='override config files')
        self.add_argument('--kv_overrides',
                          nargs='+',
                          type=str,
                          help='override key-value pairs')

    def merge(self, config_override=None):
        '''Creates experiment config object from command line and config files.'''
        config_dict = self.base_dict
        args, _ = self.parser.parse_known_args()
        if config_override is not None:
            args.overrides = config_override

        if args.restore:
            # Restore for continual training or evaluation.
            restore_path = os.path.join(args.restore, 'config.yaml')
            config_dict.update(read_file(restore_path))
        elif args.algo and args.task:
            # Start fresh training.
            config_dict['algo_config'] = get_config(args.algo)
            config_dict['task_config'] = get_config(args.task)
            if args.safety_filter:
                config_dict['sf_config'] = get_config(args.safety_filter)
        else:
            warnings.warn('No agent/task config given.')
        if args.use_gpu:
            config_dict['use_gpu'] = args.use_gpu
        # Experiment-specific overrides, e.g. training hyperparameters.
        if args.overrides:
            for f in args.overrides:
                merge_dict(config_dict, read_file(f))
        if args.kv_overrides:
            kv_dict = {}
            for kv in args.kv_overrides:
                k, v = kv.split('=')
                try:
                    v = eval(v)  # String as a python expression.
                except BaseException:
                    pass  # Normal python string.
                deep_set(kv_dict, k.strip(), v)
            merge_dict(config_dict, kv_dict)
        # Command line overrides (e.g. retains `restore` field).
        cmdline_dict = {k: v for k, v in args.__dict__.items() if v is not None}
        config_dict.update(cmdline_dict)
        # Allow attribute-style access.
        return munch.munchify(config_dict)
