'''Controller wrapper for sampled, event-triggered, delayed observations.'''

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from safe_control_gym.controllers.networked.channel import DelayChannel
from safe_control_gym.controllers.networked.metrics import NetworkedControlMetrics
from safe_control_gym.controllers.networked.trigger import DynamicPETCTrigger, TriggerDecision


@dataclass
class NetworkedStepInfo:
    '''Extra diagnostics returned by the networked wrapper.'''

    observation_used: np.ndarray
    transmitted: bool
    delivered: int
    stale_steps: int
    queue_depth: int
    trigger_margin: float
    eta: float


class NetworkedControllerWrapper:
    '''Wrap an existing safe-control-gym controller behind a delayed channel.

    The wrapper preserves the base controller API. It sends the held
    destination-side observation to the wrapped controller, while the source-side
    observation is sampled and transmitted only when the event trigger fires.
    '''

    def __init__(
        self,
        controller: Any,
        observation_dim: int,
        sample_period_steps: int = 1,
        max_delay_steps: int = 0,
        delay_sampler=None,
        trigger=None,
        name: str = 'observation',
    ):
        self.controller = controller
        self.channel = DelayChannel(
            name=name,
            dim=observation_dim,
            sample_period_steps=sample_period_steps,
            max_delay_steps=max_delay_steps,
            delay_sampler=delay_sampler,
        )
        self.trigger = trigger if trigger is not None else DynamicPETCTrigger()
        self.metrics = NetworkedControlMetrics()
        self._step = 0

    def reset(self):
        '''Reset wrapper and wrapped controller.'''
        self._step = 0
        self.channel.reset()
        self.trigger.reset()
        self.metrics.reset()
        if hasattr(self.controller, 'reset'):
            return self.controller.reset()
        return None

    def reset_before_run(self, obs=None, info=None, env=None):
        '''Reset the channel around a new evaluation run.'''
        initial_obs = np.asarray(obs, dtype=float).reshape(-1) if obs is not None else None
        self._step = 0
        self.channel.reset(initial_obs)
        self.trigger.reset()
        self.metrics.reset()
        if hasattr(self.controller, 'reset_before_run'):
            return self.controller.reset_before_run(obs=obs, info=info, env=env)
        return None

    def select_action(self, obs, info: Optional[Dict[str, Any]] = None):
        '''Select an action using the delayed held observation.'''
        step = self._extract_step(info)
        self.channel.deliver_due(step)
        channel_state = self.channel.state(step)
        self.trigger.update_flow(channel_state)

        transmitted = False
        decision = None
        if self.channel.should_sample(step):
            channel_state = self.channel.sample(obs, step)
            decision = self.trigger.evaluate(channel_state)
            if decision.transmit:
                self.channel.transmit(step)
                transmitted = True
            else:
                self.channel.skip()

        delivered = self.channel.deliver_due(step)
        channel_state = self.channel.state(step)
        obs_used = channel_state.held_signal
        action = self._select_action_from_wrapped(obs_used, info)

        if decision is None:
            error_norm = float(np.linalg.norm(channel_state.update_error))
            signal_norm = float(np.linalg.norm(channel_state.sampled_signal))
            eta = float(getattr(self.trigger, 'eta', 0.0))
            margin = float(self.trigger.error_weight * error_norm - self.trigger.signal_weight * signal_norm - eta)
            decision = TriggerDecision(
                transmit=False,
                margin=margin,
                eta=eta,
                error_norm=error_norm,
                signal_norm=signal_norm,
            )

        self.metrics.record({
            'step': step,
            'transmitted': float(transmitted),
            'skipped': float(not transmitted and self.channel.should_sample(step)),
            'delivered': float(delivered),
            'stale_steps': float(channel_state.stale_steps),
            'queue_depth': float(channel_state.queue_depth),
            'trigger_margin': float(decision.margin),
            'eta': float(decision.eta),
        })
        self._step = step + 1
        return action

    def network_info(self) -> NetworkedStepInfo:
        '''Return diagnostics for the latest wrapper state.'''
        state = self.channel.state(max(0, self._step - 1))
        last = self.metrics.events[-1] if self.metrics.events else {}
        return NetworkedStepInfo(
            observation_used=state.held_signal,
            transmitted=bool(last.get('transmitted', 0.0)),
            delivered=int(last.get('delivered', 0.0)),
            stale_steps=state.stale_steps,
            queue_depth=state.queue_depth,
            trigger_margin=float(last.get('trigger_margin', 0.0)),
            eta=float(last.get('eta', 0.0)),
        )

    def close(self):
        '''Close the wrapped controller if it owns resources.'''
        if hasattr(self.controller, 'close'):
            return self.controller.close()
        return None

    def save(self, path):
        '''Delegate checkpointing to the wrapped controller.'''
        if hasattr(self.controller, 'save'):
            return self.controller.save(path)
        return None

    def load(self, path):
        '''Delegate loading to the wrapped controller.'''
        if hasattr(self.controller, 'load'):
            return self.controller.load(path)
        return None

    def _extract_step(self, info: Optional[Dict[str, Any]]) -> int:
        if info is not None and 'current_step' in info:
            return int(info['current_step'])
        return self._step

    def _select_action_from_wrapped(self, obs, info):
        try:
            return self.controller.select_action(obs, info=info)
        except TypeError:
            return self.controller.select_action(obs)
