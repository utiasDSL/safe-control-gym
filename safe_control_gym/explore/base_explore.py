'''Base explore policy. '''

import numpy as np
from gym.utils import seeding


class BaseExplore:
    '''Template for controller/agent, implement the following methods as needed. '''

    def __init__(self,
                 env_func,
                 seed=0,
                 **kwargs
                 ):
        self.action_space = env_func().action_space
        self.act_dim = self.action_space.shape[0]
        self.seed(seed)
        
    def seed(self, seed):
        '''Makes random generator for all exploration noises.'''
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        
    def select_action(self, obs, info=None, action=None):
        '''Determine the action to take at the current timestep.'''
        raise NotImplementedError
    
    def reset(self):
        '''Do initializations for training or evaluation. '''
        pass

    def reset_before_run(self, obs=None, info=None, env=None):
        '''Reinitialize just the controller before a new run.'''
        pass
        
        
class EpsilonGreedyExplore(BaseExplore):
    '''Implements the epsilon-greedy exploration strategy.
    '''
    def __init__(self, 
                 env_func,
                 seed=0,
                 epsilon=0.,
                 action_scale=1.,
                 **kwargs
                 ):
        super().__init__(env_func, seed=seed, **kwargs)
        self.epsilon = epsilon
        self.action_scale = np.array(action_scale)
        
    def select_action(self, obs, info=None, action=None):
        '''Determine the action to take at the current timestep.'''
        prob = self.np_random.rand()
        if prob < self.epsilon:
            return self.action_space.sample() * self.action_scale
        else:
            return action 
        

class GaussianNoiseExplore(BaseExplore):
    '''Adds (bounded) independent gaussian noise to actions.
    '''
    def __init__(self, 
                 env_func,
                 seed=0,
                 noise_mean=0.,
                 noise_std=1.,
                 noise_bound_min=-float('inf'),
                 noise_bound_max=float('inf'),
                 **kwargs
                 ):
        super().__init__(env_func, seed=seed, **kwargs)
        self.noise_mean = np.array(noise_mean)
        self.noise_std = np.array(noise_std)
        self.noise_bound_min = np.array(noise_bound_min)
        self.noise_bound_max = np.array(noise_bound_max)
        
    def select_action(self, obs, info=None, action=None):
        '''Determine the action to take at the current timestep.'''
        noise = np.clip(
            self.np_random.normal(loc=self.noise_mean, scale=self.noise_std),
            self.noise_bound_min,
            self.noise_bound_max
        )
        return action + noise 
    

class OrnsteinUhlenbeckNoiseExplore(BaseExplore):
    '''Adds Ornstein Uhlenbeck action noise.
    Reference: https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    '''
    def __init__(self, 
                 env_func,
                 seed=0,
                 noise_mu=0.,
                 noise_sigma=1.,
                 noise_theta=0.15,
                 noise_dt=0.01,
                 noise_initial=None,
                 **kwargs
                 ):
        super().__init__(env_func, seed=seed, **kwargs)
        self.noise_mu = np.array(noise_mu)
        self.noise_sigma = np.array(noise_sigma)
        self.noise_theta = np.array(noise_theta)
        self.noise_dt = np.array(noise_dt)
        self.noise_initial = np.array(noise_initial)
                
    def select_action(self, obs, info=None, action=None):
        '''Determine the action to take at the current timestep.'''
        noise = self.noise_prev + self.noise_theta * (self.noise_mu - self.noise_prev) * self.noise_dt
        noise += self.noise_sigma * np.sqrt(self.noise_dt) * self.np_random.normal(size=self.act_dim)
        self.noise_prev = noise 
        return action + noise 
    
    def reset_before_run(self, obs=None, info=None, env=None):
        '''Reinitialize just the controller before a new run.'''
        self.noise_prev = self.noise_initial if self.noise_initial is not None else np.zeros_like(self.act_dim)