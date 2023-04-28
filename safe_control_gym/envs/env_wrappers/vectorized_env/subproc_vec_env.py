'''Subprocess vectorized environments.

See also:
    * https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
    * https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/subproc_vec_env.py
'''

import copy
import multiprocessing as mp

import numpy as np

from safe_control_gym.utils.utils import get_random_state, set_random_state
from safe_control_gym.envs.env_wrappers.vectorized_env.vec_env import VecEnv
from safe_control_gym.envs.env_wrappers.vectorized_env.vec_env_utils import _flatten_list, _flatten_obs, CloudpickleWrapper, clear_mpi_env_vars


class SubprocVecEnv(VecEnv):
    '''Multiprocess envs.'''

    def __init__(self, env_fns, spaces=None, context='spawn', n_workers=1):
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.n_workers = n_workers
        assert nenvs % n_workers == 0, 'Number of envs must be divisible by number of workers to run in series'
        env_fns = np.array_split(env_fns, self.n_workers)
        # Context is necessary for multiprocessing with CUDA, see pytorch.org/docs/stable/notes/multiprocessing.html
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(self.n_workers)])
        self.ps = [
            ctx.Process(target=worker,
                        args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote,
                 env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = True  # If the main process crashes, we should not cause things to hang.
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces_spec', None))
        observation_space, action_space = self.remotes[0].recv().x
        self.viewer = None
        VecEnv.__init__(self, nenvs, observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        actions = np.array_split(actions, self.n_workers)
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        results = _flatten_list(results)
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), {'n': infos}

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        results = _flatten_list(results)
        obs, infos = zip(*results)
        return _flatten_obs(obs), {'n': infos}

    def get_images(self):
        '''Called by parent `render` to support tiling images.'''
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        imgs = _flatten_list(imgs)
        return imgs

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def _assert_not_closed(self):
        assert not self.closed, 'Trying to operate on a SubprocVecEnv after calling close()'

    def get_env_random_state(self):
        for remote in self.remotes:
            remote.send(('get_random_state', None))
        worker_random_states = [remote.recv().x for remote in self.remotes]
        return worker_random_states

    def set_env_random_state(self, worker_random_states):
        for remote, random_state in zip(self.remotes, worker_random_states):
            remote.send(('set_random_state', random_state))

    def get_attr(self, attr_name, indices=None):
        '''Return attribute from vectorized environment (see base class).'''
        target_remotes, remote_env_indices = self._get_target_envs(indices)
        for remote, env_indices in zip(target_remotes, remote_env_indices):
            remote.send(('get_attr', (env_indices, attr_name)))
        return _flatten_list([remote.recv() for remote in target_remotes])

    def set_attr(self, attr_name, values, indices=None):
        '''Set attribute inside vectorized environments (see base class).'''
        target_remotes, remote_env_indices, splits = self._get_target_envs(
            indices)
        value_splits = []
        for i in range(len(splits) - 1):
            start, end = splits[i], splits[i + 1]
            value_splits.append(values[start:end])

        for remote, env_indices, value_split in zip(target_remotes,
                                                    remote_env_indices,
                                                    value_splits):
            remote.send(('set_attr', (env_indices, attr_name, value_split)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self,
                   method_name,
                   method_args=None,
                   method_kwargs=None,
                   indices=None):
        '''Call instance methods of vectorized environments.'''
        target_remotes, remote_env_indices, splits = self._get_target_envs(indices)
        method_arg_splits, method_kwarg_splits = [], []
        for i in range(len(splits) - 1):
            start, end = splits[i], splits[i + 1]
            if method_args is None:
                method_arg_splits.append([[]] * len(end - start))
            else:
                method_arg_splits.append(method_args[start:end])
            if method_kwargs is None:
                method_kwarg_splits.append([{}] * len(end - start))
            else:
                method_kwarg_splits.append(method_kwargs[start:end])

        for remote, env_indices, method_arg_split, method_kwarg_split in zip(
                target_remotes, remote_env_indices, method_arg_splits,
                method_kwarg_splits):
            remote.send(('env_method', (env_indices, method_name,
                                        method_arg_split, method_kwarg_split)))
        return _flatten_list([remote.recv() for remote in target_remotes])

    def _get_target_envs(self, indices):
        '''
        Example:
            n_workers: 3
            current envs: [0,1,2,3,4,5]
            remote_envs: [0,1], [2,3], [4,5]
            target_envs: [1,1,3,4]

            remote_indices: [0,0,1,1] -> [0,1]
            splits: [0,2] -> [0,2,4]
            remote_env_indices: [1,1,0,1] -> [1,1], [0,1]
        '''

        assert indices is None or sorted(
            indices) == indices, 'Indices must be ordered'
        indices = self._get_indices(indices)
        remote_indices = [idx // self.n_workers for idx in indices]
        remote_env_indices = [idx % self.n_workers for idx in indices]
        remote_indices, splits = np.unique(np.array(remote_indices), return_index=True)
        target_remotes = [self.remotes[idx] for idx in remote_indices]
        remote_env_indices = np.split(np.array(remote_env_indices), splits[1:])
        remote_env_indices = remote_env_indices.tolist()
        splits = np.append(splits, [len(indices)])
        return target_remotes, remote_env_indices, splits


def worker(remote, parent_remote, env_fn_wrappers):
    '''Worker func to execute vec_env commands.'''
    def step_env(env, action):
        ob, reward, done, info = env.step(action)
        if done:
            end_obs = copy.deepcopy(ob)
            end_info = copy.deepcopy(info)
            ob, info = env.reset()
            info['terminal_observation'] = end_obs
            info['terminal_info'] = end_info
        return ob, reward, done, info
    parent_remote.close()
    envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
    try:
        while True:
            cmd, data = remote.recv()
            # Branch out for requests.
            if cmd == 'step':
                remote.send(
                    [step_env(env, action) for env, action in zip(envs, data)])
            elif cmd == 'reset':
                remote.send([env.reset() for env in envs])
            elif cmd == 'render':
                remote.send([env.render(mode='rgb_array') for env in envs])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces_spec':
                remote.send(
                    CloudpickleWrapper(
                        (envs[0].observation_space, envs[0].action_space)))
            elif cmd == 'get_random_state':
                remote.send(CloudpickleWrapper(get_random_state()))
            elif cmd == 'set_random_state':
                set_random_state(data)
                # Placeholder for the return.
                remote.send(True)
            elif cmd == 'get_attr':
                env_indices, attr_name = data
                target_envs = [envs[idx] for idx in env_indices]
                remote.send([getattr(env, attr_name) for env in target_envs])
            elif cmd == 'set_attr':
                env_indices, attr_name, values = data
                target_envs = [envs[idx] for idx in env_indices]
                remote.send([
                    setattr(env, attr_name, value)
                    for env, value in zip(target_envs, values)
                ])
            elif cmd == 'env_method':
                env_indices, name, args_list, kwargs_list = data
                target_envs = [envs[idx] for idx in env_indices]
                methods = [getattr(env, name) for env in target_envs]
                remote.send([
                    method(*args, **kwargs) for method, args, kwargs in zip(
                        methods, args_list, kwargs_list)
                ])
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    except Exception as e:
        print('Environment runner process failed...')
        print(str(e))
    finally:
        for env in envs:
            env.close()
