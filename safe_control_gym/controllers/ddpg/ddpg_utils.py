from collections import defaultdict
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces import Box, Discrete

from safe_control_gym.math_and_models.distributions import Normal, Categorical
from safe_control_gym.math_and_models.neural_networks import MLP, CNN, RNN, init_
from safe_control_gym.math_and_models.normalization import BaseNormalizer, MeanStdNormalizer, RewardStdNormalizer
from safe_control_gym.math_and_models.schedule import *
from safe_control_gym.math_and_models.random_process import * 
from safe_control_gym.controllers.sac.sac_utils import SACBuffer, soft_update

# -----------------------------------------------------------------------------------
#                   Agent
# -----------------------------------------------------------------------------------


class DDPGAgent:
    """A DDPG class that encapsulates model, optimizer and update functions."""

    def __init__(self,
                 obs_space,
                 act_space,
                 hidden_dim=256,
                 gamma=0.99,
                 tau=0.005,
                 actor_lr=0.001,
                 critic_lr=0.001,
                 **kwargs):
        # params
        self.obs_space = obs_space
        self.act_space = act_space

        self.gamma = gamma
        self.tau = tau

        # model
        self.ac = MLPActorCritic(obs_space, act_space, hidden_dims=[hidden_dim] * 2, activation="relu")

        # target networks
        self.ac_targ = deepcopy(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # optimizers
        self.actor_opt = torch.optim.Adam(self.ac.actor.parameters(), actor_lr)
        self.critic_opt = torch.optim.Adam(self.ac.q.parameters(), critic_lr)

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
            "ac": self.ac.state_dict(),
            "ac_targ": self.ac_targ.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Restores agent state."""
        self.ac.load_state_dict(state_dict["ac"])
        self.ac_targ.load_state_dict(state_dict["ac_targ"])
        self.actor_opt.load_state_dict(state_dict["actor_opt"])
        self.critic_opt.load_state_dict(state_dict["critic_opt"])

    def compute_policy_loss(self, batch):
        """Returns policy loss(es) given batch of data."""
        obs = batch["obs"]
        act = self.ac.actor(obs)
        q = self.ac.q(obs, act)
        policy_loss = -q.mean()
        return policy_loss

    def compute_q_loss(self, batch):
        """Returns q-value loss(es) given batch of data."""
        obs, act, rew, next_obs, mask = batch["obs"], batch["act"], batch["rew"], batch["next_obs"], batch["mask"]
        q = self.ac.q(obs, act)

        with torch.no_grad():
            next_act = self.ac.actor(next_obs)
            next_q_targ = self.ac_targ.q(next_obs, next_act)
            # q value regression target
            q_targ = rew + self.gamma * mask * next_q_targ

        critic_loss = (q - q_targ).pow(2).mean()
        return critic_loss

    def update(self, batch):
        """Updates model parameters based on current training batch."""
        resutls = defaultdict(list)

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

        resutls["policy_loss"] = policy_loss.item()
        resutls["critic_loss"] = critic_loss.item()
        return resutls


# -----------------------------------------------------------------------------------
#                   Models
# -----------------------------------------------------------------------------------


class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_dims, activation, postprocess_fn=lambda x: x):
        super().__init__()
        self.net = MLP(obs_dim, act_dim, hidden_dims, activation)
        # in case need to pre-apply scaling or clipping on actions
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
    """Model for the actor-critic agent."""

    def __init__(self, obs_space, act_space, hidden_dims=(64, 64), activation="relu"):
        super().__init__()
        obs_dim = obs_space.shape[0]
        act_dim = act_space.shape[0]

        # policy
        low, high = act_space.low, act_space.high
        low = torch.FloatTensor(low)
        high = torch.FloatTensor(high)
        # Rescale action from [-1, 1] to [low, high]
        unscale_fn = lambda x: low.to(x.device) + (0.5 * (x + 1.0) * (high.to(x.device) - low.to(x.device)))
        self.actor = MLPActor(obs_dim, act_dim, hidden_dims, activation, postprocess_fn=unscale_fn)

        # Q functions
        self.q = MLPQFunction(obs_dim, act_dim, hidden_dims, activation)

    def act(self, obs, **kwargs):
        a = self.actor(obs)
        return a.cpu().numpy()


# -----------------------------------------------------------------------------------
#                   Storage
# -----------------------------------------------------------------------------------


class DDPGBuffer(SACBuffer):
    """Storage for replay buffer during training.

    Attributes:
        max_size (int): maximum size of the replay buffer.
        batch_size (int): number of samples (steps) per batch.
        scheme (dict): describs shape & other info of data to be stored.
        keys (list): names of all data from scheme.
    """

    def __init__(self, obs_space, act_space, max_size, batch_size=None):
        self.max_size = max_size
        self.batch_size = batch_size

        obs_dim = obs_space.shape
        act_dim = act_space.shape[0]

        N = max_size
        self.scheme = {
            "obs": {
                "vshape": (N, *obs_dim)
            },
            "next_obs": {
                "vshape": (N, *obs_dim)
            },
            "act": {
                "vshape": (N, act_dim)
            },
            "rew": {
                "vshape": (N, 1)
            },
            "mask": {
                "vshape": (N, 1),
                "init": np.ones
            }
        }
        self.keys = list(self.scheme.keys())
        self.reset()


# -----------------------------------------------------------------------------------
#                   Misc
# -----------------------------------------------------------------------------------

def make_action_noise_process(noise_config, act_space):
    """Construct a process for generating action noise during agent training.
    """
    process_func = noise_config.pop("func")
    std_config = noise_config.pop("std")

    std_func = std_config.pop("func")
    std_args = std_config.pop("args")
    std = eval(std_func)(*std_args, **std_config)

    process = eval(process_func)(size=(act_space.shape[0],), std=std)     
    return process