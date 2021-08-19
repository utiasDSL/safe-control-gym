import os
import contextlib
import numpy as np


class CloudpickleWrapper(object):
    """ Uses cloudpickle to serialize contents 
    (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


@contextlib.contextmanager
def clear_mpi_env_vars():
    """
    from mpi4py import MPI will call MPI_Init by default.  
    
    If the child process has MPI environment variables, 
    MPI will think that the child process is an MPI process 
    just like the parent and do bad things such as hang.
    
    This context manager is a hacky way to clear those environment variables 
    temporarily such as when we are starting multiprocessing Processes.
    """
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ['OMPI_', 'PMI_']:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)


def tile_images(img_nhwc):
    """Tile N images into one big PxQ image

    (P,Q) are chosen to be as close as possible, and if N is square, then P=Q.

    Args: 
        img_nhwc: list or array of images, ndim=4 once turned into array 
            n = batch index, h = height, w = width, c = channel

    Returns:
        img_Hh_Ww_c: ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(
        list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c


def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)


def _unflatten_obs(obs):
    assert isinstance(obs, (np.ndarray, dict))

    def split_batch(data):
        return [d[0] for d in np.split(data, len(data))]

    if isinstance(obs, dict):
        keys = list(obs.keys())
        unflat_obs = [split_batch(obs[k]) for k in keys]
        unflat_obs = list(zip(*unflat_obs))
        unflat_obs = [dict(zip(keys, v)) for v in unflat_obs]
        return unflat_obs
    else:
        return split_batch(obs)


def _flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])

    return [l__ for l_ in l for l__ in l_]
