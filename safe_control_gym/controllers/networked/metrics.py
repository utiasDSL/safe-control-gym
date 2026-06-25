'''Metrics for networked-control evaluation.'''

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class NetworkedControlMetrics:
    '''Collect per-step communication metrics.'''

    events: List[Dict[str, float]] = field(default_factory=list)

    def reset(self):
        '''Clear all recorded metrics.'''
        self.events.clear()

    def record(self, event: Dict[str, float]):
        '''Append one per-step metric record.'''
        self.events.append(dict(event))

    def summary(self) -> Dict[str, float]:
        '''Return aggregate metrics used for before/after comparisons.'''
        if not self.events:
            return {
                'samples': 0,
                'transmissions': 0,
                'skips': 0,
                'event_rate': 0.0,
                'mean_stale_steps': 0.0,
                'max_queue_depth': 0.0,
                'mean_trigger_margin': 0.0,
                'mean_eta': 0.0,
            }
        transmissions = sum(event.get('transmitted', 0.0) for event in self.events)
        skips = sum(event.get('skipped', 0.0) for event in self.events)
        samples = transmissions + skips
        return {
            'samples': int(samples),
            'transmissions': int(transmissions),
            'skips': int(skips),
            'event_rate': float(transmissions / samples) if samples else 0.0,
            'mean_stale_steps': float(np.mean([event.get('stale_steps', 0.0) for event in self.events])),
            'max_queue_depth': float(np.max([event.get('queue_depth', 0.0) for event in self.events])),
            'mean_trigger_margin': float(np.mean([event.get('trigger_margin', 0.0) for event in self.events])),
            'mean_eta': float(np.mean([event.get('eta', 0.0) for event in self.events])),
        }
