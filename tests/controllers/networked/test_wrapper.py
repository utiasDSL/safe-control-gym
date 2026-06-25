import numpy as np

from safe_control_gym.controllers.networked import (
    DynamicPETCTrigger,
    NetworkedControllerWrapper,
    constant_delay,
)


class ProportionalController:
    def __init__(self):
        self.observations = []

    def select_action(self, obs, info=None):
        self.observations.append(np.asarray(obs, dtype=float).copy())
        return -np.asarray(obs, dtype=float)

    def reset(self):
        self.observations.clear()


def test_wrapper_uses_held_observation_until_delivery():
    controller = ProportionalController()
    wrapper = NetworkedControllerWrapper(
        controller,
        observation_dim=1,
        sample_period_steps=1,
        max_delay_steps=1,
        delay_sampler=constant_delay(1),
        trigger=DynamicPETCTrigger(error_weight=10.0, signal_weight=0.0),
    )
    wrapper.reset_before_run(obs=np.array([0.0]))

    action_0 = wrapper.select_action(np.array([2.0]), info={'current_step': 0})
    action_1 = wrapper.select_action(np.array([3.0]), info={'current_step': 1})
    summary = wrapper.metrics.summary()

    assert np.allclose(action_0, np.array([-0.0]))
    assert np.allclose(action_1, np.array([-2.0]))
    assert summary['transmissions'] >= 1
    assert summary['max_queue_depth'] >= 0.0
