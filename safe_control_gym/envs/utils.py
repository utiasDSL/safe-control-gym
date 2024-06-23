import numpy as np
import numpy.typing as npt


def map2pi(x: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    return (x + np.pi) % (2 * np.pi) - np.pi
