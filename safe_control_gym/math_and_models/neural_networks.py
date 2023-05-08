'''Neural networks.'''

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name):
    return getattr(F, name) if name else lambda x: x


def init_(module):
    nn.init.orthogonal_(module.weight.data, gain=1)
    nn.init.constant_(module.bias.data, 0)
    return module


class MLP(nn.Module):
    '''MLP network (can be used as value or policy).'''

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=[],
                 act='relu',
                 output_act=None,
                 init_weights=False,
                 **kwargs
                 ):
        '''Multi-layer perception/fully-connected network.

        Args:
            input_dim (int): input dimension.
            output_dim (int): output dimension.
            hidden_dims (list): hidden layer dimensions.
            act (str): hidden layer activation.
            output_act (str): output layer activation.
        '''
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
        out = x
        for fc in self.fcs[:-1]:
            out = self.act(fc(out))
        out = self.output_act(self.fcs[-1](out))
        return out


class CNN(nn.Module):
    '''CNN network for encoding images.'''

    def __init__(self,
                 input_dim,
                 output_dim,
                 act='relu',
                 output_act='relu',
                 **kwargs
                 ):
        super(CNN, self).__init__()
        self.convs = nn.ModuleList([
            init_(nn.Conv2d(input_dim, 32, 8, stride=4)),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            init_(nn.Conv2d(64, 32, 3, stride=1))
        ])
        conv_out_dim = 32 * 7 * 7
        self.fc = init_(nn.Linear(conv_out_dim, output_dim))
        self.act = get_activation(act)
        self.output_act = get_activation(output_act)

    def forward(self, x, normalize=False):
        out = x / 255.0 if normalize else x
        N = x.shape[0]
        for conv in self.convs:
            out = self.act(conv(out))
        out = self.output_act(self.fc(out.view(N, -1)))
        return out


class RNN(nn.Module):
    '''RNN network (can be used as value or policy).'''

    def __init__(self,
                 input_dim,
                 output_dim,
                 **kwargs
                 ):
        super(RNN, self).__init__()
        self.gru = nn.GRU(input_dim, output_dim)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            # Forward one step, x, hxs, masks: (N, *).
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            # x, hxs: (N, *).
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # Forward a sequence, x, masks: (T, N, *), hxs: (N, *).
            T, _ = x.shape[:2]
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
