"""Probability distributions for PyTorch. 

Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/distributions.py

"""
import torch


class Normal(torch.distributions.Normal):
    """Multivariate Gaussian distribution given mean and std tensors.

    """

    def log_prob(self, actions):
        """Log probability of actions given current distribution.

        Args:
            actions (torch.FloatTensor): shape (batch, *).

        Returns:
            (torch.FloatTensor): shape (batch, 1).

        """
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        """Entropy of current distribution.

        Returns:
            (torch.FloatTensor): shape (batch,).

        """
        return super().entropy().sum(-1)

    def mode(self):
        """Mode (max probability point) of current distribution.

        """
        return self.mean


class Categorical(torch.distributions.Categorical):
    """Categorical distribution given class probabilities or logits, not differentiable.

    """

    def sample(self):
        """Sample from the current distribution.

        Returns:
            (torch.FloatTensor): shape (batch, 1).

        """
        return super().sample().unsqueeze(-1)

    def log_prob(self, actions):
        """Log probability of actions given current distribution.

        Args:
            actions (torch.FloatTensor): shape (batch, *).

        Returns:
            (torch.FloatTensor): shape (batch, 1).

        """
        return (super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1))

    def mode(self):
        """Mode (max probability point) of current distribution.

        """
        return self.probs.argmax(dim=-1, keepdim=True)
