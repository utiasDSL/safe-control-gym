'''Periodic event-triggered transmission policies.'''

from dataclasses import dataclass
from typing import Dict

import numpy as np

from safe_control_gym.controllers.networked.channel import ChannelState


@dataclass
class TriggerDecision:
    '''Result of evaluating an event trigger.'''

    transmit: bool
    margin: float
    eta: float
    error_norm: float
    signal_norm: float


class StaticPETCTrigger:
    '''Static periodic event trigger evaluated only at sampling instants.'''

    def __init__(self, error_weight=1.0, signal_weight=0.05, min_error=0.0):
        self.error_weight = float(error_weight)
        self.signal_weight = float(signal_weight)
        self.min_error = float(min_error)

    def reset(self):
        '''Reset trigger state.'''
        return

    def update_flow(self, channel_state: ChannelState, dt_steps: int = 1):
        '''Advance trigger dynamics between samples.'''
        return

    def evaluate(self, channel_state: ChannelState) -> TriggerDecision:
        '''Return a transmit/skip decision for the current sampled value.'''
        error_norm = float(np.linalg.norm(channel_state.update_error))
        signal_norm = float(np.linalg.norm(channel_state.sampled_signal))
        threshold = self.signal_weight * signal_norm + self.min_error
        margin = self.error_weight * error_norm - threshold
        return TriggerDecision(
            transmit=margin >= 0.0,
            margin=margin,
            eta=0.0,
            error_norm=error_norm,
            signal_norm=signal_norm,
        )


class DynamicPETCTrigger(StaticPETCTrigger):
    '''Dynamic PETC trigger with a scalar memory variable eta.

    eta grows when the channel is quiet and decays at sampled decisions. A
    larger eta raises the transmission threshold, so historical low-error
    behavior allows the wrapper to skip more samples.
    '''

    def __init__(
        self,
        error_weight=1.0,
        signal_weight=0.05,
        eta_decay=0.05,
        eta_growth=0.01,
        eta_transmit_drop=0.5,
        eta_skip_gain=0.1,
        eta_max=1.0,
        min_error=0.0,
    ):
        super().__init__(error_weight=error_weight, signal_weight=signal_weight, min_error=min_error)
        self.eta_decay = float(eta_decay)
        self.eta_growth = float(eta_growth)
        self.eta_transmit_drop = float(eta_transmit_drop)
        self.eta_skip_gain = float(eta_skip_gain)
        self.eta_max = float(eta_max)
        self.eta = 0.0

    def reset(self):
        '''Reset the trigger memory variable.'''
        self.eta = 0.0

    def update_flow(self, channel_state: ChannelState, dt_steps: int = 1):
        '''Advance eta between sampled decisions.'''
        error_norm = float(np.linalg.norm(channel_state.update_error))
        signal_norm = float(np.linalg.norm(channel_state.sampled_signal))
        growth = self.eta_growth * (1.0 + signal_norm) / (1.0 + error_norm)
        self.eta = max(0.0, min(self.eta_max, self.eta + dt_steps * (growth - self.eta_decay * self.eta)))

    def evaluate(self, channel_state: ChannelState) -> TriggerDecision:
        '''Return a transmit/skip decision and update eta by a jump.'''
        error_norm = float(np.linalg.norm(channel_state.update_error))
        signal_norm = float(np.linalg.norm(channel_state.sampled_signal))
        threshold = self.signal_weight * signal_norm + self.min_error + self.eta
        margin = self.error_weight * error_norm - threshold
        transmit = margin >= 0.0 or channel_state.queue_depth >= channel_state.max_delay_steps + 1
        if transmit:
            self.eta = max(0.0, self.eta * self.eta_transmit_drop)
        else:
            self.eta = min(self.eta_max, self.eta + self.eta_skip_gain * max(0.0, -margin))
        return TriggerDecision(
            transmit=transmit,
            margin=margin,
            eta=self.eta,
            error_norm=error_norm,
            signal_norm=signal_norm,
        )

    def state_dict(self) -> Dict[str, float]:
        '''Return serializable trigger state.'''
        return {'eta': self.eta}
