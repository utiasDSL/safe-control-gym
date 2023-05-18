'''Logging utilities.'''

import logging
import os
from collections import defaultdict

import imageio
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class StdoutLogger:
    '''Channel print content to std out and log file.'''

    def __init__(self, logger_name, log_dir, level=logging.INFO):
        logger = logging.getLogger(logger_name)
        formatter = logging.Formatter('%(asctime)s : %(message)s')
        # Log to file ('w' to overwrite, 'a' to keep appending).
        log_file = os.path.join(log_dir, 'std_out.txt')
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        # Log to std out.
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.setLevel(level)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        self.logger = logger
        self.file_handler = file_handler

    def info(self, msg):
        '''Chain print message to logger.'''
        self.logger.info(msg)

    def close(self):
        '''Free log file.'''
        self.file_handler.close()


class FileLogger:
    '''Logger for saving statistics and other outputs to text files.

    Based on https://github.com/michaelrzhang/logger

    Initializes the log directory and creates log files given by name in arguments.
    Can be used to append future log values to each file.
    '''

    def __init__(self, log_dir):
        # Creates folder for logging stats
        self.log_dir = os.path.join(log_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        # name of stats being tracked
        self.log_names = []

    def init_logfile(self, name, xlabel='step'):
        '''Makes text file for logging the stat.'''
        fname = self.get_log_fname(name)
        # Already exist due to restore.
        if os.path.exists(fname):
            return
        dir_name = os.path.dirname(os.path.realpath(fname))
        os.makedirs(dir_name, exist_ok=True)
        with open(fname, 'w') as log_file:
            log_file.write(f'{xlabel},{name}\n')

    def get_log_fname(self, name):
        '''Gets log file name for the stat.'''
        return os.path.join(self.log_dir, f'{name}.log')

    def log(self, name, value, step):
        '''Logs the stat to its corresponding text file.'''
        if name not in self.log_names:
            # Initialize only if not done so already.
            self.init_logfile(name)
            self.log_names.append(name)
        fname = self.get_log_fname(name)
        with open(fname, 'a') as log_file:
            log_file.write(f'{step},{value}\n')

    def restore(self, step):
        '''Resets all log files to ignore lines after `step`.'''
        # Find all stats log files.
        log_files = []
        for res, _, files in os.walk(self.log_dir):
            for each_file in files:
                if '.log' in files:
                    log_files.append(os.path.join(res, each_file))
        for fname in log_files:
            with open(fname, 'r') as file:
                lines = file.readlines()
            # Find which line to start purging.
            stop_idx = None
            for i, each_line in enumerate(lines):
                temp = each_line.strip().split(',')
                idx = int(temp[0].strip())
                stop_idx = i
                # Skip header.
                if i == 0:
                    continue
                # First invalid line.
                if idx > step:
                    break
            # Overwrite log file with only valid lines.
            lines = lines[:stop_idx]
            with open(fname, 'w') as file:
                for each_line in lines:
                    file.write(each_line)


class ExperimentLogger:
    '''A hybrid logger.'''

    def __init__(self,
                 log_dir,
                 log_std_out=True,
                 log_file_out=False,
                 use_tensorboard=False
                 ):
        '''Initializes loggers.

        Args:
            log_dir (str): name of folder to save logs.
            log_std_out (bool): if to save terminal logs.
            log_file_out (bool): if to write data logs to text files.
            use_tensorboard (bool): if to use tensorboard.
        '''
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        # Container for a log period.
        self.stats_buffer = defaultdict(list)
        # Terminal logging.
        self.log_std_out = log_std_out
        if log_std_out:
            self.std_out_logger = StdoutLogger('Benchmark', log_dir)
        # Text file logging.
        self.log_file_out = log_file_out
        if log_file_out:
            self.file_logger = FileLogger(log_dir)
        # Tensorboard logging.
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.tb_logger = SummaryWriter(log_dir=log_dir)

    def load(self, step):
        '''Resume from experiment, but ignores any logs after `step`.'''
        if self.log_file_out:
            self.file_logger.restore(step)
        if self.use_tensorboard:
            self.tb_logger = SummaryWriter(log_dir=self.log_dir,
                                           purge_step=step)

    def close(self):
        '''Cleans up logging resources.'''
        if self.log_std_out:
            self.std_out_logger.close()
        if self.use_tensorboard:
            self.tb_logger.close()

    def info(self, msg):
        '''Logs a message to std output.'''
        if self.log_std_out:
            self.std_out_logger.info(msg)
        else:
            print(msg)

    def add_scalar(self,
                   name,
                   val,
                   step,
                   store=True,
                   write=True,
                   write_tb=True):
        '''Logs a scalar data.'''
        # Add to buffer (to be logged to terminal).
        if store:
            self.stats_buffer[name].append(val)
        # Log to text file.
        if self.log_file_out and write:
            self.file_logger.log(name, val, step)
        # Log to tensorboard.
        if self.use_tensorboard and write_tb:
            self.tb_logger.add_scalar(name, val, step)

    def add_scalars(self,
                    data,
                    step,
                    prefix=None,
                    store=True,
                    write=True,
                    write_tb=True):
        '''Logs a group of scalars.'''
        assert isinstance(data, dict)
        for name, val in data.items():
            # Scalars under the same name group.
            full_name = prefix + '/' + name if prefix else name
            self.add_scalar(full_name, val, step, store, write, write_tb)

    def dump_scalars(self):
        '''Produce a summary of stats within the log period (from buffer).

        Currently only dump to terminal as a table summary,
        can dump to a CSV file in the future,
        but feels repetitive & less flexible than `add_scalar(..., write=True)`.
        '''
        keys, values = [], []
        tag = None
        # Important: sorted keys are important for consistency betwen log steps.
        for key, val_list in sorted(self.stats_buffer.items()):
            if len(val_list) == 1:
                # Left align.
                val_str = '{:<8.3g}'.format(val_list[0])
            else:
                val_np = np.asarray(val_list)
                val_str = '{:.3f} +/- {:.3f}'.format(val_np.mean(), val_np.std())
            # Find tag and add it to the dict.
            if key.find('/') > 0:
                tag = key[:key.find('/') + 1]
                trunc_tag = self._truncate(tag)
                if trunc_tag not in keys:
                    keys.append(trunc_tag)
                    values.append('')
            # Remove tag from key.
            if tag is not None and tag in key:
                key = str('   ' + key[len(tag):])
            keys.append(self._truncate(key))
            values.append(self._truncate(val_str))
        # Find max widths.
        if len(keys) == 0:
            print('Tried to write empty key-value dict')
            return
        else:
            key_width = max(map(len, keys))
            val_width = max(map(len, values))
        # Write out the data.
        dashes = '-' * (key_width + val_width + 7)
        lines = [dashes]
        for key, value in zip(keys, values):
            key_space = ' ' * (key_width - len(key))
            val_space = ' ' * (val_width - len(value))
            lines.append('| {}{} | {}{} |'.format(key, key_space, value, val_space))
        lines.append(dashes)
        summary = '\n' + '\n'.join(lines) + '\n'
        self.info(summary)
        self.stats_buffer.clear()

    def _truncate(self, string, max_length=23):
        if len(string) > max_length:
            return string[:max_length - 3] + '...'
        else:
            return string

    def log_video(self, name, video, fps=20):
        '''Saves a video for evaluation, video: list of np.arrays of shape (H,W,C).'''
        vid_kargs = {'fps': fps}
        vid_name = f'{self.log_dir}/{name}'
        imageio.mimsave(vid_name, video, **vid_kargs)
