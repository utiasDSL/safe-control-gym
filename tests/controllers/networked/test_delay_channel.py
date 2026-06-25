import numpy as np

from safe_control_gym.controllers.networked import DelayChannel


def test_delay_channel_delivers_in_transmission_order_when_sampler_reorders():
    delays = iter([2, 0])

    def sampler(_sample_count, _transmission_count):
        return next(delays)

    channel = DelayChannel('obs', dim=1, max_delay_steps=2, delay_sampler=sampler)
    channel.sample(np.array([1.0]), step=0)
    packet_0 = channel.transmit(step=0)
    channel.sample(np.array([2.0]), step=1)
    packet_1 = channel.transmit(step=1)

    assert packet_0.arrival_step == 2
    assert packet_1.arrival_step == 2
    assert channel.deliver_due(step=1) == 0
    assert channel.deliver_due(step=2) == 2
    assert np.allclose(channel.held_signal, np.array([2.0]))


def test_delay_channel_tracks_update_error_and_staleness():
    channel = DelayChannel('obs', dim=2, max_delay_steps=1)
    channel.reset(np.array([1.0, -1.0]))
    channel.sample(np.array([2.0, -3.0]), step=4)
    state = channel.state(step=4)

    assert np.allclose(state.update_error, np.array([-1.0, 2.0]))
    assert state.queue_depth == 0
    assert state.stale_steps == 4
