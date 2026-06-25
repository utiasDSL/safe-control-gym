'''Sensor scheduling helpers for bandwidth-limited channels.'''

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class ScheduleDecision:
    '''Scheduled coordinates for one transmission opportunity.'''

    indices: np.ndarray
    value: np.ndarray


class FullStateScheduler:
    '''Transmit every coordinate when the channel has enough capacity.'''

    def select(self, value, step: int, error=None) -> ScheduleDecision:
        vector = np.asarray(value, dtype=float).reshape(-1)
        return ScheduleDecision(indices=np.arange(vector.size), value=vector.copy())


class RoundRobinScheduler:
    '''Transmit fixed-size coordinate blocks in cyclic order.'''

    def __init__(self, block_size: int = 1):
        if block_size <= 0:
            raise ValueError('block_size must be positive.')
        self.block_size = int(block_size)
        self._cursor = 0

    def reset(self):
        '''Reset the scheduler cursor.'''
        self._cursor = 0

    def select(self, value, step: int, error=None) -> ScheduleDecision:
        vector = np.asarray(value, dtype=float).reshape(-1)
        start = self._cursor
        indices = [(start + offset) % vector.size for offset in range(self.block_size)]
        self._cursor = (self._cursor + self.block_size) % vector.size
        indices = np.asarray(indices, dtype=int)
        return ScheduleDecision(indices=indices, value=vector[indices].copy())


class TryOnceDiscardScheduler:
    '''Transmit coordinates with the largest current update error.'''

    def __init__(self, block_size: int = 1):
        if block_size <= 0:
            raise ValueError('block_size must be positive.')
        self.block_size = int(block_size)

    def select(self, value, step: int, error=None) -> ScheduleDecision:
        vector = np.asarray(value, dtype=float).reshape(-1)
        if error is None:
            scores = np.abs(vector)
        else:
            scores = np.abs(np.asarray(error, dtype=float).reshape(-1))
        indices = np.argsort(scores)[-self.block_size:][::-1]
        return ScheduleDecision(indices=indices, value=vector[indices].copy())
