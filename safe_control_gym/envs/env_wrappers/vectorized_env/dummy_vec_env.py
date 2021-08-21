import copy
import numpy as np

from safe_control_gym.envs.env_wrappers.vectorized_env.vec_env import VecEnv
from safe_control_gym.envs.env_wrappers.vectorized_env.vec_env_utils import _flatten_list, _flatten_obs
from safe_control_gym.utils.utils import get_random_state, set_random_state


class DummyVecEnv(VecEnv):
    """Single thread env (allow multiple envs sequentially).
    
    """

    def __init__(self,
                 env_fns
                 ):
        """
        
        """
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.actions = None
        self.closed = False

    def step_async(self,
                   actions
                   ):
        """
        
        """
        self.actions = actions

    def step_wait(self):
        """
        
        """
        results = []
        for i in range(self.num_envs):
            obs, rew, done, info = self.envs[i].step(self.actions[i])
            if done:
                end_obs = copy.deepcopy(obs)
                end_info = copy.deepcopy(info)
                obs, info = self.envs[i].reset()
                info["terminal_observation"] = end_obs
                info["terminal_info"] = end_info
            results.append([obs, rew, done, info])
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.array(rews), np.array(dones), {"n": infos}

    def reset(self):
        """
        
        """
        results = []
        for env in self.envs:
            results.append(env.reset())
        obs, infos = zip(*results)
        return _flatten_obs(obs), {"n": infos}

    def close(self):
        """
        
        """
        for env in self.envs:
            env.close()
        if self.viewer is not None:
            self.viewer.close()
        self.closed = True

    def get_images(self):
        """
        
        """
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self,
               mode='human'
               ):
        """
        
        """
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

    def get_env_random_state(self):
        """
        
        """
        return [get_random_state()]

    def set_env_random_state(self,
                             worker_random_states
                             ):
        """

        """
        set_random_state(worker_random_states[0])

    def get_attr(self,
                 attr_name,
                 indices=None
                 ):
        """Return attribute from vectorized environment (see base class).

        """
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self,
                 attr_name,
                 values,
                 indices=None
                 ):
        """Set attribute inside vectorized environments (see base class).

        """
        target_envs = self._get_target_envs(indices)
        assert len(target_envs) == len(values)
        for env_i, val_i in zip(target_envs, values):
            setattr(env_i, attr_name, val_i)

    def env_method(self,
                   method_name,
                   method_args=None,
                   method_kwargs=None,
                   indices=None):
        """Call instance methods of vectorized environments.

        """
        target_envs = self._get_target_envs(indices)
        if method_args is None:
            method_args = [[]] * len(target_envs)
        if method_kwargs is None:
            method_kwargs = [{}] * len(target_envs)
        assert len(target_envs) == len(method_args) and len(target_envs) == len(
            method_kwargs)
        return [
            getattr(env_i, method_name)(*args, **kwargs) for env_i, args, kwargs
            in zip(target_envs, method_args, method_kwargs)
        ]

    def _get_target_envs(self,
                         indices
                         ):
        """

        """
        assert indices is None or sorted(
            indices) == indices, "Indices must be ordered"
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]
