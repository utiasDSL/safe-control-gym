"""TD3 Utils."""

from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Box

from safe_control_gym.math_and_models.distributions import Categorical, Normal
from safe_control_gym.math_and_models.neural_networks import MLP

# -----------------------------------------------------------------------------------
#                   Agent
# -----------------------------------------------------------------------------------


class TD3Agent:
    """A TD3 class that encapsulates model, optimizer and update functions."""

    def __init__(self,
                 obs_space,
                 act_space,
                 hidden_dim=256,
                 gamma=0.99,
                 eps=0.01,
                 tau=0.005,
                 actor_lr=0.001,
                 critic_lr=0.001,
                 activation='relu',
                 device=None,
                 **kwargs):
        # params
        self.obs_space = obs_space
        self.act_space = act_space
        low, high = act_space.low, act_space.high
        self.action_space_low = torch.FloatTensor(low).to(device)
        self.action_space_high = torch.FloatTensor(high).to(device)

        self.gamma = gamma
        self.eps = eps
        self.tau = tau
        self.activation = activation
        self.device = device

        # model
        self.ac = MLPActorCritic(obs_space, act_space, eps=self.eps,
                                 hidden_dims=[hidden_dim] * 2, activation=self.activation)

        # target networks
        self.ac_targ = deepcopy(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # optimizers
        self.actor_opt = torch.optim.Adam(self.ac.actor.parameters(), actor_lr)
        self.critic_opt = torch.optim.Adam(list(self.ac.q1.parameters()) + list(self.ac.q2.parameters()), critic_lr)

    def to(self, device):
        """Puts agent to device."""
        self.ac.to(device)
        self.ac_targ.to(device)

    def train(self):
        """Sets training mode."""
        self.ac.train()

    def eval(self):
        """Sets evaluation mode."""
        self.ac.eval()

    def state_dict(self):
        """Snapshots agent state."""
        return {
            'ac': self.ac.state_dict(),
            'ac_targ': self.ac_targ.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict()
        }

    def load_state_dict(self, state_dict):
        """Restores agent state."""
        self.ac.load_state_dict(state_dict['ac'])
        self.ac_targ.load_state_dict(state_dict['ac_targ'])
        self.actor_opt.load_state_dict(state_dict['actor_opt'])
        self.critic_opt.load_state_dict(state_dict['critic_opt'])

    def compute_policy_loss(self, batch):
        """Returns policy loss(es) given batch of data."""
        obs = batch['obs']
        act = self.ac.actor(obs)
        q1 = self.ac.q1(obs, act)
        q2 = self.ac.q2(obs, act)
        q = torch.min(q1, q2)
        policy_loss = -q.mean()
        return policy_loss

    def compute_q_loss(self, batch):
        """Returns q-value loss(es) given batch of data."""
        obs, act, rew, next_obs, mask = batch['obs'], batch['act'], batch['rew'], batch['next_obs'], batch['mask']
        q1 = self.ac.q1(obs, act)
        q2 = self.ac.q2(obs, act)

        with torch.no_grad():
            next_act = self.ac.actor(next_obs)
            noise = (0.5*torch.randn_like(next_act)).clamp(-0.2, 0.2)
            next_act = (next_act+noise).clamp(self.action_space_low, self.action_space_high)
            next_q1_targ = self.ac_targ.q1(next_obs, next_act)
            next_q2_targ = self.ac_targ.q2(next_obs, next_act)
            next_q_targ = torch.min(next_q1_targ, next_q2_targ)
            # q value regression target
            q_targ = rew + self.gamma * mask * next_q_targ

        q1_loss = (q1 - q_targ).pow(2).mean()
        q2_loss = (q2 - q_targ).pow(2).mean()
        critic_loss = q1_loss + q2_loss
        return critic_loss

    def update(self, batch, device=None):
        """Updates model parameters based on current training batch."""
        results = defaultdict(list)

        # actor update
        policy_loss = self.compute_policy_loss(batch)
        self.actor_opt.zero_grad()
        policy_loss.backward()
        self.actor_opt.step()

        # critic update
        critic_loss = self.compute_q_loss(batch)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # update target networks
        soft_update(self.ac, self.ac_targ, self.tau)

        results['policy_loss'] = policy_loss.item()
        results['critic_loss'] = critic_loss.item()
        return results


# -----------------------------------------------------------------------------------
#                   Models
# -----------------------------------------------------------------------------------


class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_dims, activation, postprocess_fn=lambda x: x):
        super().__init__()
        self.net = MLP(obs_dim, act_dim, hidden_dims, activation)
        self.postprocess_fn = postprocess_fn

    def forward(self, obs):
        action = self.net(obs)
        action = torch.tanh(action)
        action = self.postprocess_fn(action)
        return action


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_dims, activation):
        super().__init__()
        self.q_net = MLP(obs_dim + act_dim, 1, hidden_dims, activation)

    def forward(self, obs, act):
        return self.q_net(torch.cat([obs, act], dim=-1))


class MLPActorCritic(nn.Module):
    """Model for the actor-critic agent.

    Attributes:
        actor (MLPActor): policy network.
        q1, q2 (MLPQFunction): q-value networks.
    """

    def __init__(self, obs_space, act_space, eps=0.01, hidden_dims=(64, 64), activation='relu'):
        super().__init__()

        obs_dim = obs_space.shape[0]
        if isinstance(act_space, Box):
            act_dim = act_space.shape[0]
            discrete = False
        else:
            raise NotImplementedError

        # policy
        low, high = act_space.low, act_space.high
        low = torch.FloatTensor(low)
        high = torch.FloatTensor(high)

        def unscale_fn(x):  # Rescale action from [-1, 1] to [low, high]
            return low.to(x.device) + (0.5 * (x + 1.0) * (high.to(x.device) - low.to(x.device)))
        self.actor = MLPActor(obs_dim, act_dim, hidden_dims, activation, postprocess_fn=unscale_fn)

        # Q functions
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_dims, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_dims, activation)

    def act(self, obs):
        a = self.actor(obs)
        return a.cpu().numpy()


# -----------------------------------------------------------------------------------
#                   Storage
# -----------------------------------------------------------------------------------


class TD3Buffer(object):
    """Storage for replay buffer during training.

    Attributes:
        max_size (int): maximum size of the replay buffer.
        batch_size (int): number of samples (steps) per batch.
        scheme (dict): describes shape & other info of data to be stored.
        keys (list): names of all data from scheme.
    """

    def __init__(self, obs_space, act_space, max_size, batch_size=None):
        super().__init__()
        self.max_size = max_size
        self.batch_size = batch_size

        obs_dim = obs_space.shape
        if isinstance(act_space, Box):
            act_dim = act_space.shape[0]
        else:
            act_dim = act_space.n

        N = max_size
        self.scheme = {
            'obs': {
                'vshape': (N, *obs_dim)
            },
            'next_obs': {
                'vshape': (N, *obs_dim)
            },
            'act': {
                'vshape': (N, act_dim)
            },
            'rew': {
                'vshape': (N, 1)
            },
            'mask': {
                'vshape': (N, 1),
                'init': np.ones
            }
        }
        self.keys = list(self.scheme.keys())
        self.reset()

    def reset(self):
        """Allocate space for containers."""
        for k, info in self.scheme.items():
            assert 'vshape' in info, f'Scheme must define vshape for {k}'
            vshape = info['vshape']
            dtype = info.get('dtype', np.float32)
            init = info.get('init', np.zeros)
            self.__dict__[k] = init(vshape, dtype=dtype)

        self.pos = 0
        self.buffer_size = 0

    def __len__(self):
        """Returns current size of the buffer."""
        return self.buffer_size

    def state_dict(self):
        """Returns a snapshot of current buffer."""
        state = dict(
            pos=self.pos,
            buffer_size=self.buffer_size,
        )
        for k in self.scheme:
            v = self.__dict__[k]
            state[k] = v
        return state

    def load_state_dict(self, state):
        """Restores buffer from previous state."""
        for k, v in state.items():
            self.__dict__[k] = v

    def push(self, batch):
        """Inserts transition step data (as dict) to storage."""
        # batch size
        k = list(batch.keys())[0]
        n = batch[k].shape[0]

        for k, v in batch.items():
            shape = self.scheme[k]['vshape'][1:]
            dtype = self.scheme[k].get('dtype', np.float32)
            v_ = np.asarray(v, dtype=dtype).reshape((n,) + shape)

            if self.pos + n <= self.max_size:
                self.__dict__[k][self.pos:self.pos + n] = v_
            else:
                # wrap around
                remain_n = self.pos + n - self.max_size
                self.__dict__[k][self.pos:self.max_size] = v_[:-remain_n]
                self.__dict__[k][:remain_n] = v_[-remain_n:]

        if self.buffer_size < self.max_size:
            self.buffer_size = min(self.max_size, self.pos + n)
        self.pos = (self.pos + n) % self.max_size

    def sample(self, batch_size=None, device=None):
        """Returns data batch."""
        if not batch_size:
            batch_size = self.batch_size

        indices = np.random.randint(0, len(self), size=batch_size)
        batch = {}
        for k, info in self.scheme.items():
            shape = info['vshape'][1:]
            v = self.__dict__[k].reshape(-1, *shape)[indices]
            if device is None:
                batch[k] = torch.as_tensor(v)
            else:
                batch[k] = torch.as_tensor(v, device=device)
        return batch


# -----------------------------------------------------------------------------------
#                   Misc
# -----------------------------------------------------------------------------------


def soft_update(source, target, tau):
    """Synchronizes target networks with exponential moving average."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(source, target):
    """Synchronizes target networks by copying over parameters directly."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
