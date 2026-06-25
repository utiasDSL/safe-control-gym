import numpy as np

from safe_control_gym.controllers.networked import ChannelState, DynamicPETCTrigger, StaticPETCTrigger


def make_state(update_error, sampled_signal=(1.0,), queue_depth=0, max_delay_steps=2):
    return ChannelState(
        sampled_signal=np.asarray(sampled_signal, dtype=float),
        held_signal=np.asarray(sampled_signal, dtype=float) + np.asarray(update_error, dtype=float),
        update_error=np.asarray(update_error, dtype=float),
        queue_depth=queue_depth,
        max_delay_steps=max_delay_steps,
    )


def test_static_trigger_transmits_when_error_crosses_threshold():
    trigger = StaticPETCTrigger(error_weight=1.0, signal_weight=0.1)

    assert not trigger.evaluate(make_state([0.01], sampled_signal=[1.0])).transmit
    assert trigger.evaluate(make_state([0.5], sampled_signal=[1.0])).transmit


def test_dynamic_trigger_eta_grows_after_skips_and_backlog_forces_transmit():
    trigger = DynamicPETCTrigger(
        error_weight=1.0,
        signal_weight=0.1,
        eta_growth=0.0,
        eta_skip_gain=0.5,
        eta_max=10.0,
    )
    skipped = trigger.evaluate(make_state([0.01], sampled_signal=[1.0]))
    forced = trigger.evaluate(make_state([0.01], sampled_signal=[1.0], queue_depth=3, max_delay_steps=2))

    assert not skipped.transmit
    assert skipped.eta > 0.0
    assert forced.transmit
