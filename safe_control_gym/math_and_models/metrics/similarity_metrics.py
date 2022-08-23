"""Implements a set of trajectory-based similarity metrics, either based on
one-to-one trajectory comparisons or set-to-set trajectory comparisons (distriubtion metric).

Metrics:
* MMD
* DFD
* LCSS
* EDR
* DTW

References:
* [similaritymeasures](https://github.com/cjekel/similarity_measures): DTW, discrete Frechet
* [ts-dist](https://github.com/ymtoo/ts-dist): DTW, LCSS, EDR
* [Robust and Fast Similarity Search for Moving Object Trajectories](https://cs.uwaterloo.ca/~tozsu/publications/spatial/sigmod05-leichen.pdf): ED, DTW, LCSS, EDR

Todos:
* Normalization/masking on input data.
* Test effectiveness of the metrics and computation time.
* Use numba for some metric implementations to accelerate (refer to similaritymeasures).

"""
import numpy as np 
import torch 
from safe_control_gym.math_and_models.transformations import npRotXYZ



def encode_data(data, tuple_length=1, include_action=True):
    """Processes trajectory data into step tuples, each is treated as a sample for MMD comparison.

    Args:
        data (dict): Must contain keys `obs`, `act`, `n_trajs`.
            the values for `obs` and `act` are list of list of np.arrays.
        tuple_length (int): length of each tuple as 1 sample.
        include_action (bool): if to include action in each tuple sample.
        
    Returns:
        encoded_data (ndarray): Shape is (#tuples, obs_dim*(l+1)+act_dim*l) or 
            (#tuples, obs_dim*(l+1)) where l is tuple length.
    """
    # data = Munch(dict(n_steps=n_steps, obs=ep_obs_list, act=ep_act_list))
    sas_tuples = []
    for i in range(data["n_trajs"]):
        ep_obs = data["obs"][i]
        ep_obs_tuples = [ep_obs[j:j+tuple_length+1] for j in range(len(ep_obs) - tuple_length)]
        if include_action:
            ep_act = data["act"][i]
            ep_act_tuples = [ep_act[j:j+tuple_length] for j in range(len(ep_act) - tuple_length + 1)]
            ep_tuples = [np.concatenate(o_tp + a_tp) 
                        for o_tp, a_tp in zip(ep_obs_tuples, ep_act_tuples)]
        else:
            ep_tuples = [np.concatenate(o_tp) for o_tp in ep_obs_tuples]
        sas_tuples.extend(ep_tuples)
    # shape (#tuples, O*(l+1)+A*l) or (#tuples, O*(l+1))
    encoded_data = np.asarray(sas_tuples)
    return encoded_data


def mmd_loss(samples1, samples2, mode='gaussian', sigma=0.2):
    """Computes the MMD loss as similarity metric between 2 sets of trajectories..

    adapted from https://github.com/aviralkumar2907/BEAR/blob/f2e31c1b5f81c4fb0e692a34949c7d8b48582d8f/algos.py#L326

    Args:
        samples1 (torch.FloatTensor): Sample data, shape (B,N) where B is batch/sample size and N is sample dim.
        samples2 (torch.FloatTensor): Sample data, shape (B,N) where B is batch/sample size and N is sample dim.
        mode (str): Kernel name to use.
        sigma (float): Std param in kernel.

    Returns:
        overall_loss (float): MMD value.
    """
    diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
    diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
    diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d

    if mode == 'gaussian':
        diff_x_x = torch.mean((-(diff_x_x.pow(2)).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))
        diff_x_y = torch.mean((-(diff_x_y.pow(2)).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))
        diff_y_y = torch.mean((-(diff_y_y.pow(2)).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))
    elif mode == 'laplacian':
        diff_x_x = torch.mean((-(diff_x_x.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))
        diff_x_y = torch.mean((-(diff_x_y.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))
        diff_y_y = torch.mean((-(diff_y_y.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))
    else:
        raise NotImplementedError('The given MMD mode is not available.')

    overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
    return overall_loss


def mse(traj1, traj2):
    """Mean squared error/Euclidean distance.

    Args:
        traj1 (ndarray): shape (T1,D).
        traj1 (ndarray): shape (T2,D).
    
    Returns:
        cost (float): final MSE cost.
    """
    if len(traj1) != len(traj2):
        raise ValueError("Input trajectories must have the same length.")
    # normalize by the length of trajectory, then square root
    cost = np.sqrt(((traj1 - traj2)**2).sum()/len(traj1))
    return cost


def dtw(traj1, traj2, w=np.inf, distance_func=euclidean_distance, distance_func_kwargs={}):
    """Dynamic time wrapper. 
    Reference: https://github.com/ymtoo/ts-dist/blob/30de6eba0969611cda58754e212bddef2b28772e/ts_dist.py#L8
    Reference2: https://github.com/cjekel/similarity_measures/blob/bfcd744a052ea50c4a318f5b38b275b3f93b67d5/similaritymeasures/similaritymeasures.py#L671
    Reference3: https://www.cs.unm.edu/~mueen/DTW.pdf

    Args:
        traj1 (ndarray): shape (T1,D).
        traj1 (ndarray): shape (T2,D).
        w (int): window size (default inf, meaning to compare all pairs).
        distance_func (Callable): distance function for pair of elements (default is Eucleadian distance).

    Returns:
        cost (float): final DTW cost.
    """
    nx, ny = len(traj1), len(traj2)
    D = np.full((nx+1, ny+1), np.inf)
    D[0, 0]= 0
    w = max(w, abs(nx-ny))
    for i in range(1, nx+1):
        for j in range(max(1, i-w), min(ny+1, i+w+1)):
            # adapt a custom state-pair distance metric here
            subcost = distance_func(traj1[i-1], traj2[j-1], **distance_func_kwargs)
            D[i, j] = subcost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    # final DTW cost, not normalized since it depends on the distance_func used
    cost = D[nx, ny]
    return cost


def edr(traj1, traj2, epsilon=0.05, distance_func=euclidean_distance, distance_func_kwargs={}):
    """Edit distance on real sequences.
    Reference: https://github.com/ymtoo/ts-dist/blob/30de6eba0969611cda58754e212bddef2b28772e/ts_dist.py#L88
    
    Args:
        traj1 (ndarray): shape (T1,D).
        traj1 (ndarray): shape (T2,D).
        epsilon (float): threshold to decide if 2 elements are "equal".
        
    Returns:
        cost (float): final EDR cost.
    """
    nx, ny = len(traj1), len(traj2)
    D = np.full((nx+1, ny+1), np.inf)
    # base cases 
    D[:, 0] = np.arange(nx+1)
    D[0, :] = np.arange(ny+1)
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            # if np.all(np.abs(traj1[i-1]-traj2[j-1]) < epsilon):
            # adapt a custom state-pair distance metric here
            # TODO: should we also adapt subcost to be continuous instead of discrete
            if distance_func(traj1[i-1], traj2[j-1], **distance_func_kwargs) < epsilon:
                subcost = 0
            else:
                subcost = 1 
        D[i, j] = min(D[i-1, j-1]+subcost, D[i-1, j]+1, D[i, j-1]+1)
    # final EDR cost (normalized to [0,1])
    cost = D[nx, ny] / max(nx, ny)
    return cost 
    
    
def lcss(traj1, traj2, delta=np.inf, epsilon=0.05, distance_func=euclidean_distance, distance_func_kwargs={}):
    """Longest common subsequence. 
    Reference: https://github.com/ymtoo/ts-dist/blob/30de6eba0969611cda58754e212bddef2b28772e/ts_dist.py#L52
    
    Args:
        traj1 (ndarray): shape (T1,D).
        traj1 (ndarray): shape (T2,D).
        delta (float): time shreshold to decide if 2 timestamps are "equal".
        epsilon (float): threshold to decide if 2 elements are "equal".
    Returns:
        cost (float): final LCSS cost.
    """
    nx, ny = len(traj1), len(traj2)
    S = np.zeros((nx+1, ny+1))
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            # if np.all(np.abs(traj1[i-1]-traj2[j-1]) < epsilon) and (np.abs(i-j) < delta):
            # adapt a custom state-pair distance metric here
            # TODO: should we also adapt the +1 to be continuous state distance
            if distance_func(traj1[i-1], traj2[j-1], **distance_func_kwargs) < epsilon:
                S[i, j] = S[i-1, j-1]+1
            else:
                S[i, j] = max(S[i, j-1], S[i-1, j])
    # final LCSS cost (normalized to [0,1], lower the better/more similar)
    cost = 1 - S[nx, ny] / min(nx, ny)
    return cost 


def discrete_frechet(traj1, traj2, p=2, distance_func=euclidean_distance, distance_func_kwargs={}):
    """Discrete Frechet distance. 
    Reference: https://github.com/cjekel/similarity_measures/blob/bfcd744a052ea50c4a318f5b38b275b3f93b67d5/similaritymeasures/similaritymeasures.py#L430
    
    Args: 
        traj1 (ndarray): shape (T1,D).
        traj1 (ndarray): shape (T2,D).
        p (float): 1 <= p <= inf, which Minkowski p-norm to use, default 2 is Euclidean.
    
    Returns:
        cost (float): final DF cost.
    """
    nx, ny = len(traj1), len(traj2)
    D = np.full((nx, ny), -1)
    # base case
    # D[0, 0] = minkowski_distance(traj1[0], traj2[0], p=p)
    D[0, 0] = distance_func(traj1[0], traj2[0], **distance_func_kwargs)
    # build DP table 
    for i in range(1, nx):
        # D[i, 0] = max(D[i-1, 0], minkowski_distance(traj1[i], traj2[0], p=p))
        D[i, 0] = max(D[i-1, 0], distance_func(traj1[i], traj2[0], **distance_func_kwargs))
    for j in range(1, ny):
        # D[0, j] = max(D[0, j-1], minkowski_distance(traj1[0], traj2[j], p=p))
        D[0, j] = max(D[0, j-1], distance_func(traj1[0], traj2[j], **distance_func_kwargs))
    for i in range(1, nx):
        for j in range(1, ny):
            # D[i, j] = max(min(D[i-1, j], D[i, j-1], D[i-1, j-1]),
            #                minkowski_distance(traj1[i], traj2[j], p=p))
            D[i, j] = max(min(D[i-1, j], D[i, j-1], D[i-1, j-1]),
                           distance_func(traj1[i], traj2[j], **distance_func_kwargs))
    # final DF cost, not normalized
    cost = D[nx-1, ny-1]
    return cost 



def euclidean_distance(u, v, weights=None):
    """ Euclidean distance, or use https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html
    
    Args:
        u, v (ndarray): 1D array, shape (D,). 
        weights (float|ndarray): weights for each vector dimension.
    """
    weights = np.array(weights) if weights else 1
    return np.sqrt((weights * (u-v)**2).sum())


def minkowski_distance(u, v, p=2, weights=None):
    """Minkowski distance, or use https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.minkowski.html
    
    Args:
        u, v (ndarray): 1D array, shape (D,). 
        p (float): order of the norm.
        weights (float|ndarray): weights for each vector dimension.
    """
    weights = np.array(weights) if weights else 1
    return np.pow((weights * np.abs(u-v)**p).sum(), 1./p)


def rotation_distance(rot1, rot2, mode="geodesic"):
    """Metrics for 3D Rotations. 
    Reference: https://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf
    
    Args:
        rot1, rot2 (ndarray): rotation matrices, shape (3,3).
        mode (str): options from geodesic|dev_id|inner_unit_quad2|inner_unit_quad|norm_diff_quad.
    """
    # eqn 23 in reference, range [0,pi]
    # use clipping to avoid NaNs for rounding errors
    # reference: https://github.com/utiasSTARS/liegroups/blob/fe1d376b7d33809dec78724b456f01833507c305/liegroups/numpy/so3.py#L180
    rot_d = np.arccos(np.clip(0.5*np.trace(rot1 @ np.transpose(rot2))-0.5, -1., 1.))
    # can derive other metrics using their functional equivalence 
    if mode == "geodesic":
        pass
    elif mode == "dev_id":
        # # eqn 25 in reference 
        # rot_d = np.sqrt(2* (3 - np.trace(rot1 @ np.transpose(rot2))))
        # eqn 33, range [0, 2*sqrt(2)]
        rot_d = 2 * np.sqrt(2) * np.sin(rot_d/2)
    elif mode == "inner_unit_quad":
        # eqn 34, range [0, pi/2]
        rot_d = rot_d / 2
    elif mode == "inner_unit_quad2":
        # eqn 32, range [0, 1]
        rot_d = 1 - np.cos(rot_d/2)
    elif mode == "norm_diff_quad":
        # eqn 31, range [0, sqrt(2)]
        rot_d = np.sqrt(2 * (1 - np.cos(rot_d/2)))
    else:
        raise NotImplementedError("The given rotation metric is not available.")
    return rot_d 


def state_distance(state1, state2, weights=None, rot_mode="geodesic", task="cartpole", quad_type=3):
    """Metric distance between two env states.
    
    Args:
        state1, state2 (ndarray): trajectory states, shape (D,).
        weights (list|ndarray): weights to different elements/groups in state.
        rot_mode (str): rotation metric mode.

    Returns:
        state_d (float): distance metric between the 2 states.
    """    
    if task == "cartpole":
        # state is [x, x_dot, theta, theta_dot]
        state_d = euclidean_distance(state1, state2, weights)
    elif task == "quadrotor":
        if quad_type == 1:
            # state is [z, z_dot]
            if weights:
                assert len(weights) == 2, "weights must have shape (2,)"
            state_d = euclidean_distance(state1, state2, weights)
        elif quad_type == 2:
            # state is [x, x_dot, z, z_dot, theta, theta_dot]
            if weights:
                assert len(weights) == 4, "weights must have shape (4,)"
            else:
                weights = np.ones(4)
            # extract state
            pos1, pos2 = state1[[0,2]], state2[[0,2]] 
            vel1, vel2 = state1[[1,3]], state2[[1,3]] 
            ang1, ang2 = state1[4], state2[4] 
            angvel1, angvel2 = state1[5], state2[5]
            # convert euler angles to rotation matrices 
            rot1, rot2 = npRotXYZ(0, ang1, 0), npRotXYZ(0, ang2, 0)
            # compose total state distance
            state_d = weights[0] * euclidean_distance(pos1, pos2) + \
                    weights[1] * euclidean_distance(vel1, vel2) + \
                    weights[2] * rotation_distance(rot1, rot2, rot_mode) + \
                    weights[3] * euclidean_distance(angvel1, angvel2)
        else:
            # state is [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body]
            if weights:
                assert len(weights) == 4, "weights must have shape (4,)"
            else:
                weights = np.ones(4)
            # extract state
            pos1, pos2 = state1[[0,2,4]], state2[[0,2,4]] 
            vel1, vel2 = state1[[1,3,5]], state2[[1,3,5]] 
            ang1, ang2 = state1[6:9], state2[6:9] 
            angvel1_body, angvel2_body = state1[9:12], state2[9:12]
            # convert euler angles to rotation matrices 
            rot1, rot2 = npRotXYZ(*ang1), npRotXYZ(*ang2)
            # convert body rates to angular velocities w.r.t inertial frame
            angvel1, angvel2 = rot1 @ angvel1_body, rot2 @ angvel2_body
            # compose total state distance
            state_d = weights[0] * euclidean_distance(pos1, pos2) + \
                    weights[1] * euclidean_distance(vel1, vel2) + \
                    weights[2] * rotation_distance(rot1, rot2, rot_mode) + \
                    weights[3] * euclidean_distance(angvel1, angvel2)
    else:
        raise NotImplementedError("The given task state distance is not available.")
    return state_d