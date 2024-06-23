"""Disturbances."""

from __future__ import annotations

import sys

import numpy as np
import numpy.typing as npt


class Disturbance:
    """Base class for disturbance or noise applied to inputs or dyanmics."""

    def __init__(self, dim: int, mask: npt.NDArray[np.bool_] | None = None):
        self.dim = dim
        self.mask = mask
        self.np_random = np.random.default_rng()
        if mask is not None:
            self.mask = np.asarray(mask)
            assert self.dim == len(self.mask)
        self.step = 0

    def reset(self):
        self.step = 0

    def step(self):
        self.step += 1

    def apply(self, target: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        """Default is identity."""
        return target

    def seed(self, seed: int | None = None):
        """Reset seed from env."""
        self.np_random = np.random.default_rng(seed)


class UniformNoise(Disturbance):
    """i.i.d uniform noise ~ U(low, high) per time step."""

    def __init__(
        self,
        dim: int,
        mask: npt.NDArray[np.bool_] | None = None,
        low: float = 0.0,
        high: float = 1.0,
    ):
        super().__init__(dim, mask)
        assert isinstance(low, (float, list, np.ndarray)), "low must be float or list."
        assert isinstance(high, (float, list, np.ndarray)), "high must be float or list."
        self.low = np.array([low] * self.dim) if isinstance(low, float) else np.asarray(low)
        self.high = np.array([high] * self.dim) if isinstance(high, float) else np.asarray(high)

    def apply(self, target: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        noise = self.np_random.uniform(self.low, self.high, size=self.dim)
        if self.mask is not None:
            noise *= self.mask
        return target + noise


class WhiteNoise(Disturbance):
    """I.i.d Gaussian noise per time step."""

    def __init__(
        self,
        dim: int,
        mask: npt.NDArray[np.bool_] | None = None,
        std: float | npt.NDArray[np.float_] = 1.0,
    ):
        super().__init__(dim, mask)
        assert isinstance(std, (float, list, np.ndarray)), "std must be float or list."
        self.std = np.array([std] * self.dim) if isinstance(std, float) else np.asarray(std)
        assert self.dim == len(self.std), "std shape should be the same as dim."

    def apply(self, target: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        noise = self.np_random.normal(0, self.std, size=self.dim)
        if self.mask is not None:
            noise *= self.mask
        return target + noise


class DisturbanceList:
    """Combine list of disturbances as one."""

    def __init__(self, disturbances: list[Disturbance]):
        """Initialization of the list of disturbances."""
        self.disturbances = disturbances

    def reset(self):
        """Sequentially reset disturbances."""
        for disturb in self.disturbances:
            disturb.reset()

    def apply(self, target: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        """Sequentially apply disturbances."""
        disturbed = target
        for disturb in self.disturbances:
            disturbed = disturb.apply(disturbed)
        return disturbed

    def seed(self, seed: int | None = None):
        """Reset seed from env."""
        for disturb in self.disturbances:
            disturb.seed(seed)

    @staticmethod
    def from_specs(disturbance_specs: list[dict]) -> DisturbanceList:
        """Create a DisturbanceList from a list of disturbance specifications.

        Args:
            disturbance_specs: List of dicts defining the disturbances info.
        """
        disturb_list = []
        # Each disturbance for the mode.
        for disturb in disturbance_specs:
            assert isinstance(disturb, dict), "Each disturbance must be specified as dict."
            assert "type" in disturb.keys(), "Each distrubance must have a 'type' key."
            d_class = getattr(sys.modules[__name__], disturb["type"])
            disturb_list.append(d_class(**{k: v for k, v in disturb.items() if k != "type"}))
        return DisturbanceList(disturb_list)
