'''Disturbances.'''

import numpy as np


class Disturbance:
    '''Base class for disturbance or noise applied to inputs or dyanmics.'''

    def __init__(self,
                 env,
                 dim,
                 mask=None,
                 **kwargs
                 ):
        self.dim = dim
        self.mask = mask
        if mask is not None:
            self.mask = np.asarray(mask)
            assert self.dim == len(self.mask)

    def reset(self,
              env
              ):
        pass

    def apply(self,
              target,
              env
              ):
        '''Default is identity.'''
        return target

    def seed(self, env):
        '''Reset seed from env.'''
        self.np_random = env.np_random


class DisturbanceList:
    '''Combine list of disturbances as one.'''

    def __init__(self,
                 disturbances
                 ):
        '''Initialization of the list of disturbances.'''
        self.disturbances = disturbances

    def reset(self,
              env
              ):
        '''Sequentially reset disturbances.'''
        for disturb in self.disturbances:
            disturb.reset(env)

    def apply(self,
              target,
              env
              ):
        '''Sequentially apply disturbances.'''
        disturbed = target
        for disturb in self.disturbances:
            disturbed = disturb.apply(disturbed, env)
        return disturbed

    def seed(self, env):
        '''Reset seed from env.'''
        for disturb in self.disturbances:
            disturb.seed(env)


class ImpulseDisturbance(Disturbance):
    '''Impulse applied during a short time interval.

    Examples:
        * single step, square (duration=1, decay_rate=1): ______|-|_______
        * multiple step, square (duration>1, decay_rate=1): ______|-----|_____
        * multiple step, triangle (duration>1, decay_rate<1): ______/\\_____
    '''

    def __init__(self,
                 env,
                 dim,
                 mask=None,
                 magnitude=1,
                 step_offset=None,
                 duration=1,
                 decay_rate=1,
                 **kwargs
                 ):
        super().__init__(env, dim, mask)
        self.magnitude = magnitude
        self.step_offset = step_offset
        self.max_step = int(env.EPISODE_LEN_SEC / env.CTRL_TIMESTEP)
        # Specify shape of the impulse.
        assert duration >= 1
        assert decay_rate > 0 and decay_rate <= 1
        self.duration = duration
        self.decay_rate = decay_rate

    def reset(self,
              env
              ):
        if self.step_offset is None:
            self.current_step_offset = self.np_random.randint(self.max_step)
        else:
            self.current_step_offset = self.step_offset
        self.current_peak_step = int(self.current_step_offset + self.duration / 2)

    def apply(self,
              target,
              env
              ):
        noise = 0
        if env.ctrl_step_counter >= self.current_step_offset:
            peak_offset = np.abs(env.ctrl_step_counter - self.current_peak_step)
            if peak_offset < self.duration / 2:
                decay = self.decay_rate**peak_offset
            else:
                decay = 0
            noise = self.magnitude * decay
        if self.mask is not None:
            noise *= self.mask
        disturbed = target + noise
        return disturbed


class StepDisturbance(Disturbance):
    '''Constant disturbance at all time steps (but after offset).

    Applied after offset step (randomized or given): _______|---------
    '''

    def __init__(self,
                 env,
                 dim,
                 mask=None,
                 magnitude=1,
                 step_offset=None,
                 **kwargs
                 ):
        super().__init__(env, dim, mask)
        self.magnitude = magnitude
        self.step_offset = step_offset
        self.max_step = int(env.EPISODE_LEN_SEC / env.CTRL_TIMESTEP)

    def reset(self,
              env
              ):
        if self.step_offset is None:
            self.current_step_offset = self.np_random.randint(self.max_step)
        else:
            self.current_step_offset = self.step_offset

    def apply(self,
              target,
              env
              ):
        noise = 0
        if env.ctrl_step_counter >= self.current_step_offset:
            noise = self.magnitude
        if self.mask is not None:
            noise *= self.mask
        disturbed = target + noise
        return disturbed


class UniformNoise(Disturbance):
    '''i.i.d uniform noise ~ U(low, high) per time step.'''

    def __init__(self, env, dim, mask=None, low=0.0, high=1.0, **kwargs):
        super().__init__(env, dim, mask)

        # uniform distribution bounds
        if isinstance(low, float):
            self.low = np.asarray([low] * self.dim)
        elif isinstance(low, list):
            self.low = np.asarray(low)
        else:
            raise ValueError('[ERROR] UniformNoise.__init__(): low must be specified as a float or list.')

        if isinstance(high, float):
            self.high = np.asarray([high] * self.dim)
        elif isinstance(low, list):
            self.high = np.asarray(high)
        else:
            raise ValueError('[ERROR] UniformNoise.__init__(): high must be specified as a float or list.')

    def apply(self, target, env):
        noise = self.np_random.uniform(self.low, self.high, size=self.dim)
        if self.mask is not None:
            noise *= self.mask
        disturbed = target + noise
        return disturbed


class WhiteNoise(Disturbance):
    '''I.i.d Gaussian noise per time step.'''

    def __init__(self,
                 env,
                 dim,
                 mask=None,
                 std=1.0,
                 **kwargs
                 ):
        super().__init__(env, dim, mask)
        # I.i.d gaussian variance.
        if isinstance(std, float):
            self.std = np.asarray([std] * self.dim)
        elif isinstance(std, list):
            self.std = np.asarray(std)
        else:
            raise ValueError('[ERROR] WhiteNoise.__init__(): std must be specified as a float or list.')
        assert self.dim == len(self.std), 'std shape should be the same as dim.'

    def apply(self,
              target,
              env
              ):
        noise = self.np_random.normal(0, self.std, size=self.dim)
        if self.mask is not None:
            noise *= self.mask
        disturbed = target + noise
        return disturbed


class BrownianNoise(Disturbance):
    '''Simple random walk noise.'''

    def __init__(self):
        super().__init__()


class PeriodicNoise(Disturbance):
    '''Sinuisodal noise.'''

    def __init__(self,
                 env,
                 dim,
                 mask=None,
                 scale=1.0,
                 frequency=1.0,
                 **kwargs
                 ):
        super().__init__(env, dim)
        # Sine function parameters.
        self.scale = scale
        self.frequency = frequency

    def apply(self,
              target,
              env
              ):
        phase = self.np_random.uniform(low=-np.pi, high=np.pi, size=self.dim)
        t = env.pyb_step_counter * env.PYB_TIMESTEP
        noise = self.scale * np.sin(2 * np.pi * self.frequency * t + phase)
        if self.mask is not None:
            noise *= self.mask
        disturbed = target + noise
        return disturbed


class StateDependentDisturbance(Disturbance):
    '''Time varying and state varying, e.g. friction.

    Here to provide an explicit form, can also enable friction in simulator directly.
    '''

    def __init__(self,
                 env,
                 dim,
                 mask=None,
                 **kwargs
                 ):
        super().__init__()


DISTURBANCE_TYPES = {'impulse': ImpulseDisturbance,
                     'step': StepDisturbance,
                     'uniform': UniformNoise,
                     'white_noise': WhiteNoise,
                     'periodic': PeriodicNoise,
                     }


def create_disturbance_list(disturbance_specs, shared_args, env):
    '''Creates a DisturbanceList from yaml disturbance specification.

    Args:
        disturbance_specs (list): List of dicts defining the disturbances info.
        shared_args (dict): args shared across the disturbances in the list.
        env (BenchmarkEnv): Env for which the constraints will be applied
    '''
    disturb_list = []
    # Each disturbance for the mode.
    for disturb in disturbance_specs:
        assert 'disturbance_func' in disturb.keys(), '[ERROR]: Every distrubance must specify a disturbance_func.'
        disturb_func = disturb['disturbance_func']
        assert disturb_func in DISTURBANCE_TYPES, '[ERROR] in BenchmarkEnv._setup_disturbances(), disturbance type not available.'
        disturb_cls = DISTURBANCE_TYPES[disturb_func]
        cfg = {key: disturb[key] for key in disturb if key != 'disturbance_func'}
        disturb = disturb_cls(env, **shared_args, **cfg)
        disturb_list.append(disturb)
    return DisturbanceList(disturb_list)
