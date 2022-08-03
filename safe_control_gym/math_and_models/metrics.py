import warnings
import numpy as np 
import torch 



def compute_cvar(data, alpha, lower_range=True):
    """CVaR as mean of the lower-alpha-percentile of data.
        adapted from https://github.com/nuria95/O-RAAC/blob/57347bc682798ff9f5600131c606517832efe864/oraaclib/util/utilities.py
    """
    if not isinstance(data, torch.Tensor):
        data = torch.Tensor(data)
    if len(data.size()) < 2:
        data.unsqueeze_(0)
    batch_size, N = data.size()
    sorted_data, _ = torch.sort(data)

    # NOTE: what does it do?
    # if alpha == 1 or alpha <= 0.5:
    #     cvar = sorted_data[:, :int(alpha * N)].mean(1)
    # else:
    #     cvar = sorted_data[:, int(alpha * N)::].mean(1)
    if lower_range:
        cvar = sorted_data[:, :int(alpha * N)].mean(1)
    else:
        cvar = sorted_data[:, -int(alpha * N):].mean(1)
    if all(torch.isnan(cvar)):
        raise ValueError(f'Not enough samples to compute {alpha} '
                         f'CVaR from {data}')
    else:
        if len(data.size()) < 2:
            cvar = cvar.squeeze(0)
        return cvar.numpy()