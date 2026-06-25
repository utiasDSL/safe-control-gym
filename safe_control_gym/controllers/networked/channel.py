'''Sampled communication channels with bounded delivery delay.'''

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np


ArrayLike = np.ndarray
DelaySampler = Callable[[int, int], int]


def constant_delay(delay_steps: int) -> DelaySampler:
    '''Return a deterministic delay sampler.'''
    delay_steps = int(delay_steps)

    def _sample(_sample_count: int, _transmission_count: int) -> int:
        return delay_steps

    return _sample


def uniform_delay(max_delay_steps: int, rng: Optional[np.random.Generator] = None) -> DelaySampler:
    '''Return an integer delay sampler on [0, max_delay_steps].'''
    max_delay_steps = int(max_delay_steps)
    rng = np.random.default_rng() if rng is None else rng

    def _sample(_sample_count: int, _transmission_count: int) -> int:
        return int(rng.integers(0, max_delay_steps + 1))

    return _sample


@dataclass
class ChannelPacket:
    '''A transmitted sample waiting for delivery.'''

    value: ArrayLike
    sampled_step: int
    transmit_step: int
    arrival_step: int


@dataclass
class ChannelState:
    '''Observable state of one networked channel.'''

    sampled_signal: ArrayLike
    held_signal: ArrayLike
    update_error: ArrayLike
    queue_depth: int
    max_delay_steps: int
    sample_count: int = 0
    transmission_count: int = 0
    delivery_count: int = 0
    skipped_count: int = 0
    stale_steps: int = 0
    last_sample_step: int = 0
    last_delivery_step: int = 0


@dataclass
class DelayChannel:
    '''A FIFO sampled-data channel with a maximum delay measured in samples.

    The class models the paper's practical communication objects: sampled
    signal, held destination signal, update error, and the delay backlog l_i.
    '''

    name: str
    dim: int
    sample_period_steps: int = 1
    max_delay_steps: int = 0
    delay_sampler: Optional[DelaySampler] = None
    initial_value: Optional[ArrayLike] = None
    _queue: List[ChannelPacket] = field(default_factory=list, init=False)
    _sampled_signal: ArrayLike = field(default=None, init=False)
    _held_signal: ArrayLike = field(default=None, init=False)
    _last_sample_step: int = field(default=0, init=False)
    _last_delivery_step: int = field(default=0, init=False)
    _last_arrival_step: int = field(default=0, init=False)
    _sample_count: int = field(default=0, init=False)
    _transmission_count: int = field(default=0, init=False)
    _delivery_count: int = field(default=0, init=False)
    _skipped_count: int = field(default=0, init=False)

    def __post_init__(self):
        if self.sample_period_steps <= 0:
            raise ValueError('sample_period_steps must be positive.')
        if self.max_delay_steps < 0:
            raise ValueError('max_delay_steps must be non-negative.')
        if self.delay_sampler is None:
            self.delay_sampler = constant_delay(self.max_delay_steps)
        value = np.zeros(self.dim, dtype=float) if self.initial_value is None else self._as_vector(self.initial_value)
        self.reset(value)

    def reset(self, value: Optional[ArrayLike] = None):
        '''Reset channel memory and counters.'''
        value = self._held_signal if value is None and self._held_signal is not None else value
        value = np.zeros(self.dim, dtype=float) if value is None else self._as_vector(value)
        self._queue.clear()
        self._sampled_signal = value.copy()
        self._held_signal = value.copy()
        self._last_sample_step = 0
        self._last_delivery_step = 0
        self._last_arrival_step = 0
        self._sample_count = 0
        self._transmission_count = 0
        self._delivery_count = 0
        self._skipped_count = 0

    def should_sample(self, step: int) -> bool:
        '''Return whether this channel samples at the given control step.'''
        return int(step) % self.sample_period_steps == 0

    def sample(self, value: ArrayLike, step: int) -> ChannelState:
        '''Record the latest source-side sample.'''
        self._sampled_signal = self._as_vector(value)
        self._last_sample_step = int(step)
        self._sample_count += 1
        return self.state(step)

    def transmit(self, step: int) -> ChannelPacket:
        '''Send the latest sampled value through the delayed channel.'''
        step = int(step)
        delay_steps = int(self.delay_sampler(self._sample_count, self._transmission_count))
        delay_steps = max(0, min(delay_steps, self.max_delay_steps))
        arrival_step = max(step + delay_steps, self._last_arrival_step)
        self._last_arrival_step = arrival_step
        packet = ChannelPacket(
            value=self._sampled_signal.copy(),
            sampled_step=self._last_sample_step,
            transmit_step=step,
            arrival_step=arrival_step,
        )
        self._queue.append(packet)
        self._transmission_count += 1
        return packet

    def skip(self):
        '''Record that a sampled value was intentionally not transmitted.'''
        self._skipped_count += 1

    def deliver_due(self, step: int) -> int:
        '''Deliver all packets whose arrival step has elapsed.'''
        step = int(step)
        delivered = 0
        pending = []
        for packet in self._queue:
            if packet.arrival_step <= step:
                self._held_signal = packet.value.copy()
                self._last_delivery_step = step
                self._delivery_count += 1
                delivered += 1
            else:
                pending.append(packet)
        self._queue = pending
        return delivered

    def state(self, step: int) -> ChannelState:
        '''Return the current channel state.'''
        step = int(step)
        return ChannelState(
            sampled_signal=self._sampled_signal.copy(),
            held_signal=self._held_signal.copy(),
            update_error=self._held_signal - self._sampled_signal,
            queue_depth=len(self._queue),
            max_delay_steps=self.max_delay_steps,
            sample_count=self._sample_count,
            transmission_count=self._transmission_count,
            delivery_count=self._delivery_count,
            skipped_count=self._skipped_count,
            stale_steps=max(0, step - self._last_delivery_step),
            last_sample_step=self._last_sample_step,
            last_delivery_step=self._last_delivery_step,
        )

    @property
    def held_signal(self) -> ArrayLike:
        '''Latest signal available at the destination side.'''
        return self._held_signal.copy()

    def _as_vector(self, value: ArrayLike) -> ArrayLike:
        array = np.asarray(value, dtype=float).reshape(-1)
        if array.size != self.dim:
            raise ValueError(f'Channel {self.name} expected dimension {self.dim}, got {array.size}.')
        return array
