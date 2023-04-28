'''Adapted from OpenAI Baselines.

See also:
    * https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_env.py
    * https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/base_vec_env.py
'''

from abc import ABC, abstractmethod

from safe_control_gym.envs.env_wrappers.vectorized_env.vec_env_utils import tile_images


class VecEnv(ABC):
    '''An abstract asynchronous, vectorized environment.

    Used to batch data from multiple copies of an environment, so that each observation becomes a
    batch of observations, and expected action is a batch of actions to be applied per-environment.
    '''
    closed = False
    viewer = None
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 num_envs,
                 observation_space,
                 action_space
                 ):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        '''Reset all the environments and return an array of observations, or a dict of observation arrays.

        If step_async is still doing work, that work will be cancelled and step_wait() should not
        be called until step_async() is invoked again.
        '''
        pass

    @abstractmethod
    def step_async(self,
                   actions
                   ):
        '''Tell all the environments to start taking a step with the given actions.

        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is already pending.
        '''
        pass

    @abstractmethod
    def step_wait(self):
        '''Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
            - obs: an array of observations, or a dict of arrays of observations.
            - rews: an array of rewards
            - dones: an array of 'episode done' booleans
            - infos: a sequence of info objects
        '''
        pass

    def close_extras(self):
        '''Clean up the  extra resources. Only runs when not self.closed.'''
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self,
             actions
             ):
        '''Step the environments synchronously.'''
        self.step_async(actions)
        return self.step_wait()

    def render(self,
               mode='human'
               ):
        '''Display environment via a viewer.'''
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        '''Return RGB images from each environment.'''
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer

    @abstractmethod
    def get_attr(self, attr_name, indices=None):
        '''Return attribute from vectorized environment.'''
        pass

    @abstractmethod
    def set_attr(self, attr_name, values, indices=None):
        '''Set attribute inside vectorized environments.'''
        pass

    @abstractmethod
    def env_method(self,
                   method_name,
                   method_args=None,
                   method_kwargs=None,
                   indices=None):
        '''Call instance methods of vectorized environments.'''
        raise NotImplementedError()

    def _get_indices(self,
                     indices
                     ):
        '''Convert a flexibly-typed reference to environment indices to an implied list of indices.'''
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return indices


class VecEnvWrapper(VecEnv):
    '''An environment wrapper that applies to an entire batch of environments at once.'''

    def __init__(self,
                 venv,
                 observation_space=None,
                 action_space=None
                 ):
        self.venv = venv
        super().__init__(num_envs=venv.num_envs,
                         observation_space=observation_space or venv.observation_space,
                         action_space=action_space or venv.action_space)

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self, mode='human'):
        return self.venv.render(mode=mode)

    def get_images(self):
        return self.venv.get_images()

    def __getattr__(self,
                    name
                    ):
        if name.startswith('_'):
            raise AttributeError(
                f'attempted to get missing private attribute \'{name}\'')
        return getattr(self.venv, name)

    def get_attr(self,
                 attr_name,
                 indices=None
                 ):
        return self.venv.get_attr(attr_name, indices)

    def set_attr(self,
                 attr_name,
                 values,
                 indices=None
                 ):
        return self.venv.set_attr(attr_name, values, indices)

    def env_method(self,
                   method_name,
                   method_args=None,
                   method_kwargs=None,
                   indices=None):
        return self.venv.env_method(method_name,
                                    method_args=method_args,
                                    method_kwargs=method_kwargs,
                                    indices=indices)
