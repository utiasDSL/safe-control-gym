"""Utility function for a generic safe explorer.

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym.spaces import Box
from collections import defaultdict
from copy import deepcopy

from safe_control_gym.envs.env_wrappers.vectorized_env.vec_env_utils import _flatten_obs
from safe_control_gym.math_and_models.neural_networks import MLP, CNN, RNN, init_
from safe_control_gym.math_and_models.distributions import Normal, Categorical
from safe_control_gym.controllers.ppo.ppo_utils import random_sample


class SafetyLayer:
    """Layer to learn constraint models and to impose action projection.

    """

    def __init__(self,
                 obs_space,
                 act_space,
                 hidden_dim=64,
                 num_constraints=1,
                 lr=0.001,
                 slack=None,
                 device='cpu',
                 **kwargs):
        # Parameters.
        self.num_constraints = num_constraints
        self.device = device
        # Seperate model per constraint.
        input_dim = obs_space.shape[0]
        output_dim = act_space.shape[0]
        
        # default 1 layer 
        if isinstance(hidden_dim, int):
            hidden_dims=[hidden_dim]
        elif isinstance(hidden_dim, list):
            hidden_dims = hidden_dim
        else:
            raise ValueError("hidden_dim can only be int or list.")
        self.constraint_models = nn.ModuleList([
            # MLP(input_dim, output_dim, hidden_dims=[hidden_dim])
            MLP(input_dim, output_dim, hidden_dims=hidden_dims)
            for _ in range(self.num_constraints)
        ])
        # Constraint slack variables/values.
        assert slack is not None and isinstance(slack, (int, float, list))
        if isinstance(slack, (int, float)):
            slack = [slack]
        self.slack = np.array(slack)
        # Optimizers.
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=lr)
            for model in self.constraint_models
        ]

    def to(self,
           device
           ):
        """Puts agent to device.

        """
        self.constraint_models.to(device)

    def train(self):
        """Sets training mode.

        """
        self.constraint_models.train()

    def eval(self):
        """Sets evaluation mode.

        """
        self.constraint_models.eval()

    def state_dict(self):
        """Snapshots agent state.

        """
        return {
            "constraint_models": self.constraint_models.state_dict(),
            "optimizers": [opt.state_dict() for opt in self.optimizers]
        }

    def load_state_dict(self,
                        state_dict
                        ):
        """Restores agent state.

        """
        self.constraint_models.load_state_dict(state_dict["constraint_models"])
        for i, opt_state_dict in enumerate(state_dict["optimizers"]):
            self.optimizers[i].load_state_dict(opt_state_dict)

    def compute_loss(self,
                     batch
                     ):
        """Gets constraint value L2 loss for each constraint.

        """
        obs, act = batch["obs"].to(self.device), batch["act"].to(self.device)
        c, c_next = batch["c"].to(self.device), batch["c_next"].to(self.device)

        gs = [model(obs) for model in self.constraint_models]

        # Each is (N,1,A) x (N,A,1) -> (N,), so [(N,)]_{n_constriants}
        c_next_pred = [
            c[:, i] + torch.bmm(g.view(g.shape[0], 1, -1),
                                act.view(act.shape[0], -1, 1)).view(-1)
            for i, g in enumerate(gs)
        ]
        losses = [
            torch.mean((c_next[:, i] - c_next_pred[i])**2).cpu()
            for i in range(self.num_constraints)
        ]
        return losses

    def update(self, batch):
        """Updates the constraint models from data batch.

        """
        losses = self.compute_loss(batch)
        for loss, opt in zip(losses, self.optimizers):
            opt.zero_grad()
            loss.backward()
            opt.step()
        results = {
            "constraint_{}_loss".format(i): loss.item()
            for i, loss in enumerate(losses)
        }
        return results

    def get_safe_action(self,
                        obs,
                        act,
                        c
                        ):
        """Does action projection with the trained safety layer.

        According to Dalal 2018, this simple projection works when only 1 constraint at a time
        is active; for multiple active constriants, either resort to in-graph QP solver such as
        OptLayer or see cvxpylayers (https://github.com/cvxgrp/cvxpylayers).

        Args:
            obs (torch.FloatTensor): observations, shape (B,O).
            act (torch.FloatTensor): actions, shape (B,A).
            c (torch.FloatTensor): constraints, shape (B,C).
            
        Returns:
            torch.FloatTensor: transformed/projected actions, shape (B,A).

        """
        self.eval()
        # [(B,A)]_C
        g = [model(obs) for model in self.constraint_models]
        # Find the lagrange multipliers [(B,)]_C
        multipliers = []
        for i in range(len(g)):
            g_i = g[i]  # (B,A)
            c_i = c[:, i]  # (B,)
            # (B,1,A)x(B,A,1) -> (B,1,1) -> (B,)
            numer = torch.bmm(g_i.unsqueeze(1),
                              act.unsqueeze(2)).view(-1) + c_i + self.slack[i]
            denomin = torch.bmm(g_i.unsqueeze(1),
                                g_i.unsqueeze(2)).view(-1) + 1e-8
            # Equation (5) from Dalal 2018.
            mult = F.relu(numer / denomin)  # (B,)
            multipliers.append(mult)
        multipliers = torch.stack(multipliers, -1)  # (B,C)
        # Calculate correction, equation (6) from Dalal 2018.
        max_mult, max_idx = torch.topk(multipliers, 1, dim=-1)  # (B,1)
        max_idx = max_idx.view(-1).tolist()  # []_B
        # [(A,)]_B -> (B,A)
        max_g = torch.stack([g[max_i][i] for i, max_i in enumerate(max_idx)])
        # (B,1) x (B,A) -> (B,A)
        correction = max_mult * max_g
        action_new = act - correction
        return action_new


class ConstraintBuffer(object):
    """Storage for replay buffer during training.

    Attributes:
        max_size (int): maximum size of the replay buffer.
        batch_size (int): number of samples (steps) per batch.
        scheme (dict): describs shape & other info of data to be stored.
        keys (list): names of all data from scheme.

    """

    def __init__(self,
                 obs_space,
                 act_space,
                 num_constraints,
                 max_size,
                 batch_size=None
                 ):
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
            "obs": {
                "vshape": (N, *obs_dim)
            },
            "act": {
                "vshape": (N, act_dim)
            },
            "c": {
                "vshape": (N, num_constraints)
            },
            "c_next": {
                "vshape": (N, num_constraints)
            }
        }
        self.keys = list(self.scheme.keys())
        self.reset()

    def reset(self):
        """Allocate space for containers.

        """
        for k, info in self.scheme.items():
            assert "vshape" in info, "Scheme must define vshape for {}".format(
                k)
            vshape = info["vshape"]
            dtype = info.get("dtype", np.float32)
            init = info.get("init", np.zeros)
            self.__dict__[k] = init(vshape, dtype=dtype)
        self.pos = 0
        self.buffer_size = 0

    def __len__(self):
        """Returns current size of the buffer.

        """
        return self.buffer_size

    def state_dict(self):
        """Returns a snapshot of current buffer.

        """
        state = dict(
            pos=self.pos,
            buffer_size=self.buffer_size,
        )
        for k in self.scheme:
            v = self.__dict__[k]
            state[k] = v
        return state

    def load_state_dict(self,
                        state
                        ):
        """Restores buffer from previous state.

        """
        for k, v in state.items():
            self.__dict__[k] = v

    def push(self,
             batch
             ):
        """Inserts transition step data (as dict) to storage.

        """
        # Batch size.
        k = list(batch.keys())[0]
        n = batch[k].shape[0]
        for k, v in batch.items():
            shape = self.scheme[k]["vshape"][1:]
            dtype = self.scheme[k].get("dtype", np.float32)
            v_ = np.asarray(v, dtype=dtype).reshape((n,) + shape)
            if self.pos + n <= self.max_size:
                self.__dict__[k][self.pos:self.pos + n] = v_
            else:
                # Wrap.
                remain_n = self.pos + n - self.max_size
                self.__dict__[k][self.pos:self.max_size] = v_[:-remain_n]
                self.__dict__[k][:remain_n] = v_[-remain_n:]
        if self.buffer_size < self.max_size:
            self.buffer_size = min(self.max_size, self.pos + n)
        self.pos = (self.pos + n) % self.max_size

    def sample(self,
               indices
               ):
        """Returns partial data.

        """
        batch = {}
        for k, info in self.scheme.items():
            shape = info["vshape"][1:]
            batch[k] = self.__dict__[k].reshape(-1, *shape)[indices]
        return batch

    def sampler(self,
                batch_size,
                device="cpu",
                drop_last=True
                ):
        """Makes sampler to loop through all data.

        """
        total_steps = len(self)
        sampler = random_sample(np.arange(total_steps), batch_size, drop_last)
        for indices in sampler:
            batch = self.sample(indices)
            batch = {
                k: torch.as_tensor(v, device=device) for k, v in batch.items()
            }
            yield batch
