'''Utility function for the PPO-based safe explorer.'''

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box

from safe_control_gym.math_and_models.neural_networks import MLP
from safe_control_gym.math_and_models.distributions import Normal, Categorical
import safe_control_gym.controllers.ppo.ppo_utils as ppo_utils


class SafePPOAgent(ppo_utils.PPOAgent):
    '''A PPO class that encapsulates models, optimizers and update functions.'''

    def __init__(self,
                 obs_space,
                 act_space,
                 hidden_dim=64,
                 use_clipped_value=False,
                 clip_param=0.2,
                 target_kl=0.01,
                 entropy_coef=0.01,
                 actor_lr=0.0003,
                 critic_lr=0.001,
                 opt_epochs=10,
                 mini_batch_size=64,
                 action_modifier=None,
                 **kwargs
                 ):
        # Parameters.
        self.obs_space = obs_space
        self.act_space = act_space
        self.use_clipped_value = use_clipped_value
        self.clip_param = clip_param
        self.target_kl = target_kl
        self.entropy_coef = entropy_coef
        self.opt_epochs = opt_epochs
        self.mini_batch_size = mini_batch_size
        # Model.
        self.ac = MLPActorCritic(obs_space,
                                 act_space,
                                 hidden_dims=[hidden_dim] * 2,
                                 activation='tanh',
                                 action_modifier=action_modifier)
        # Optimizers.
        self.actor_opt = torch.optim.Adam(self.ac.actor.parameters(), actor_lr)
        self.critic_opt = torch.optim.Adam(self.ac.critic.parameters(), critic_lr)

    def compute_policy_loss(self, batch):
        '''Returns policy loss(es) given batch of data.'''
        obs, act, logp_old, adv, c = batch['obs'], batch['act'], batch['logp'], batch['adv'], batch['c']
        dist, logp = self.ac.actor(obs, act, c=c)
        # Policy.
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = -torch.min(ratio * adv, clip_adv).mean()
        # Entropy.
        entropy_loss = -dist.entropy().mean()
        # KL/trust region.
        approx_kl = (logp_old - logp).mean()
        return policy_loss, entropy_loss, approx_kl


class MLPActor(nn.Module):
    '''Actor model.'''

    def __init__(self,
                 obs_dim,
                 act_dim,
                 hidden_dims,
                 activation,
                 discrete=False,
                 action_modifier=None
                 ):
        super().__init__()
        self.pi_net = MLP(obs_dim, act_dim, hidden_dims, activation)
        # Construct output action distribution.
        self.discrete = discrete
        if discrete:
            self.dist_fn = lambda x: Categorical(logits=x)
        else:
            self.logstd = nn.Parameter(-0.5 * torch.ones(act_dim))
            self.dist_fn = lambda x: Normal(x, self.logstd.exp())
        # Safety filter.
        self.action_modifier = action_modifier

    def forward(self,
                obs,
                act=None,
                c=None
                ):
        mu = self.pi_net(obs)
        # Filter action if needed.
        if self.action_modifier:
            if len(mu.shape) == 1:
                # During evalution or single env runs.
                mu_safe = self.action_modifier(obs.unsqueeze(0),
                                               mu.unsqueeze(0),
                                               c.unsqueeze(0)).view(-1)
            else:
                # During training or vectorized runs.
                mu_safe = self.action_modifier(obs, mu, c)
        else:
            mu_safe = mu
        dist = self.dist_fn(mu_safe)
        logp_a = None
        if act is not None:
            logp_a = dist.log_prob(act)
        return dist, logp_a


class MLPActorCritic(ppo_utils.MLPActorCritic):
    '''Model for the actor-critic agent.

    Attributes:
        actor (MLPActor): policy network.
        critic (MLPCritic): value network.
    '''

    def __init__(self,
                 obs_space,
                 act_space,
                 hidden_dims=(64, 64),
                 activation='tanh',
                 action_modifier=None
                 ):
        nn.Module.__init__(self)
        obs_dim = obs_space.shape[0]
        if isinstance(act_space, Box):
            act_dim = act_space.shape[0]
            discrete = False
        else:
            act_dim = act_space.n
            discrete = True
        # Policy.
        self.actor = MLPActor(obs_dim, act_dim, hidden_dims, activation, discrete, action_modifier)
        # Value function.
        self.critic = ppo_utils.MLPCritic(obs_dim, hidden_dims, activation)

    def step(self,
             obs,
             c=None
             ):
        dist, _ = self.actor(obs, c=c)
        a = dist.sample()
        logp_a = dist.log_prob(a)
        v = self.critic(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self,
            obs,
            c=None
            ):
        dist, _ = self.actor(obs, c=c)
        a = dist.mode()
        return a.numpy()


class SafePPOBuffer(ppo_utils.PPOBuffer):
    '''Storage for a batch of episodes during training.

    Attributes:
        max_length (int): maximum length of episode.
        batch_size (int): number of episodes per batch.
        scheme (dict): describs shape & other info of data to be stored.
        keys (list): names of all data from scheme.
    '''

    def __init__(self,
                 obs_space,
                 act_space,
                 num_constraints,
                 max_length,
                 batch_size
                 ):
        self.max_length = max_length
        self.batch_size = batch_size
        T, N = max_length, batch_size
        obs_dim = obs_space.shape
        if isinstance(act_space, Box):
            act_dim = act_space.shape[0]
        else:
            act_dim = act_space.n
        self.scheme = {
            'obs': {
                'vshape': (T, N, *obs_dim)
            },
            'act': {
                'vshape': (T, N, act_dim)
            },
            'rew': {
                'vshape': (T, N, 1)
            },
            'mask': {
                'vshape': (T, N, 1),
                'init': np.ones
            },
            'v': {
                'vshape': (T, N, 1)
            },
            'logp': {
                'vshape': (T, N, 1)
            },
            'ret': {
                'vshape': (T, N, 1)
            },
            'adv': {
                'vshape': (T, N, 1)
            },
            'terminal_v': {
                'vshape': (T, N, 1)
            },
            'c': {
                'vshape': (T, N, num_constraints)
            },
        }
        self.keys = list(self.scheme.keys())
        self.reset()
