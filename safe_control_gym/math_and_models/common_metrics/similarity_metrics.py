"""Implements a set of trajectory-based similarity metrics, either based on 
one-to-one trajectory comparisons or set-to-set trajectory comparisons (distriubtion metric).
"""
import numpy as np 
import torch 



def encode_data(data, tuple_length=1):
    """Processes trajectory data into step tuples, each is treated as a sample for MMD comparison.

    Args:
        data (dict): must contain keys `obs`, `act`, `n_trajs`. 
            the values for `obs` and `act` are list of list of np.arrays.
        tuple_length: length of each tuple as 1 sample.
        
    Returns:
        encoded_data (np.array): shape is (#tuples, obs_dim*(l+1)_act_dim*l) where l is tuple length.
    """
    # data = Munch(dict(n_steps=n_steps, obs=ep_obs_list, act=ep_act_list))
    sas_tuples = []
    for i in range(data["n_trajs"]):
        ep_obs, ep_act = data["obs"][i], data["act"][i]
        ep_obs_tuples = [ep_obs[j:j+tuple_length+1] for j in range(len(ep_obs) - tuple_length)]
        ep_act_tuples = [ep_act[j:j+tuple_length] for j in range(len(ep_act) - tuple_length + 1)]
        ep_tuples = [np.concatenate(o_tp + a_tp) 
                     for o_tp, a_tp in zip(ep_obs_tuples, ep_act_tuples)]
        sas_tuples.extend(ep_tuples)
    # shape (#tuples, O*(l+1)+A*l)
    encoded_data = np.asarray(sas_tuples)
    return encoded_data


def mmd_loss(samples1, samples2, mode="gaussian", sigma=0.2):
    """Computes the MMD loss as similarity metric between 2 sets of trajectories..

    adapted from https://github.com/aviralkumar2907/BEAR/blob/f2e31c1b5f81c4fb0e692a34949c7d8b48582d8f/algos.py#L326
    
    Args:
        samples1 (torch.FloatTensor): sample data, shape (B,N) where B is batch/sample size and N is sample dim.
        samples2 (torch.FloatTensor): sample data, shape (B,N) where B is batch/sample size and N is sample dim.
        mode (str): kernel name to use.
        sigma (float): std param in kernel.
        
    Returns:
        overall_loss (float): MMD value.
    """
    diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
    diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
    diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d

    if mode == "gaussian":
        diff_x_x = torch.mean((-(diff_x_x.pow(2)).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))
        diff_x_y = torch.mean((-(diff_x_y.pow(2)).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))
        diff_y_y = torch.mean((-(diff_y_y.pow(2)).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))
    elif mode == "laplacian":
        diff_x_x = torch.mean((-(diff_x_x.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))
        diff_x_y = torch.mean((-(diff_x_y.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))
        diff_y_y = torch.mean((-(diff_y_y.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))
    else:
        raise NotImplementedError("The given MMD mode is not available.")

    overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
    return overall_loss