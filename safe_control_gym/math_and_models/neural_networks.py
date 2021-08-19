"""Neural networks.

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Args:
        name (int): The first parameter.

    Returns:
        bool: The return value. True for success, False otherwise.

    """
    return getattr(F, name) if name else lambda x: x


def init_(module):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Args:
        module (int): The first parameter.

    Returns:
        bool: The return value. True for success, False otherwise.

    """
    # could have different gains with different activations
    nn.init.orthogonal_(module.weight.data, gain=1)
    nn.init.constant_(module.bias.data, 0)
    return module


class MLP(nn.Module):
    """MLP network (can be used as value or policy).

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=[],
                 act="relu",
                 output_act=None,
                 init_weights=False,
                 **kwargs):
        """Mlti-layer perception / fully-connected network.

        Args:
            input_dim (int): input dimension.
            output_dim (int): output dimension.
            hidden_dims (list): hidden layer dimensions.
            act (str): hidden layer activation.
            output_act (str): output layer activation.

        """
        super(MLP, self).__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        init_func = init_ if init_weights else lambda x: x

        self.fcs = nn.ModuleList([
            init_func(nn.Linear(dims[i], dims[i + 1]))
            for i in range(len(dims) - 1)
        ])
        self.act = get_activation(act)
        self.output_act = get_activation(output_act)

    def forward(self, x):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        out = x
        for fc in self.fcs[:-1]:
            out = self.act(fc(out))
        out = self.output_act(self.fcs[-1](out))
        return out


class CNN(nn.Module):
    """CNN network for encoding images.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 act="relu",
                 output_act="relu",
                 **kwargs):
        """Example of docstring on the __init__ method.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1 (str): Description of `param1`.

        """
        super(CNN, self).__init__()
        self.convs = nn.ModuleList([
            init_(nn.Conv2d(input_dim, 32, 8, stride=4)),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            init_(nn.Conv2d(64, 32, 3, stride=1))
        ])
        # too lazy for auto-calculate, do it with a test run instead
        conv_out_dim = 32 * 7 * 7
        self.fc = init_(nn.Linear(conv_out_dim, output_dim))
        self.act = get_activation(act)
        self.output_act = get_activation(output_act)

    def forward(self, x, normalize=False):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        out = x / 255.0 if normalize else x
        N = x.shape[0]
        for conv in self.convs:
            out = self.act(conv(out))
        out = self.output_act(self.fc(out.view(N, -1)))
        return out


class RNN(nn.Module):
    """RNN network (can be used as value or policy).

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 **kwargs):
        """Example of docstring on the __init__ method.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1 (str): Description of `param1`.

        """
        super(RNN, self).__init__()
        self.gru = nn.GRU(input_dim, output_dim)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, hxs, masks):
        """Run the RNN taking account in masking.

        The first condition applies during exploraion or evaluation, the second 
        condition applies during training/action evaluation.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        if x.size(0) == hxs.size(0):
            # Forward one step, x, hxs, masks: (N, *).
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            # x, hxs: (N, *).
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # Forward a sequence, x, masks: (T, N, *), hxs: (N, *).
            T, N = x.shape[:2]
            # Let's figure out which steps in the sequence have a zero for any agent.
            # We will always assume t=0 has a zero in it as that makes the logic cleaner.
            has_zeros = (masks.squeeze(-1)[1:] == 0.0).any(dim=-1).nonzero()
            has_zeros = has_zeros.squeeze().cpu()
            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar.
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()
            # Add t=0 and t=T to the list.
            has_zeros = [0] + has_zeros + [T]
            # Run gru in masked segments, faster.
            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx], (hxs * masks[start_idx]).unsqueeze(0))
                outputs.append(rnn_scores)
            # x: (T, N, *), hxs: (N, *).
            x = torch.cat(outputs, dim=0)
            hxs = hxs.squeeze(0)
        return x, hxs
