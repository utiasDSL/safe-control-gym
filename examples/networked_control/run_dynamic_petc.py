'''Minimal dynamic PETC wrapper demonstration.

This example intentionally uses a small local plant instead of recreating a
paper example. In safe-control-gym, replace ProportionalController with an LQR,
MPC, PPO, SAC, or PID controller instance and keep the wrapper unchanged.
'''

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safe_control_gym.controllers.networked import DynamicPETCTrigger, NetworkedControllerWrapper, uniform_delay


class ProportionalController:
    '''Small controller with the same select_action shape used by safe-control-gym.'''

    def select_action(self, obs, info=None):
        obs = np.asarray(obs, dtype=float)
        return np.array([-0.8 * obs[0] - 0.25 * obs[1]])

    def reset(self):
        return None

    def close(self):
        return None


def simulate(controller, n_steps=200, dt=0.05):
    '''Run a double-integrator stabilization task.'''
    x = np.array([2.0, 0.0])
    trajectory = []
    for step in range(n_steps):
        action = np.asarray(controller.select_action(x, info={'current_step': step}), dtype=float)
        u = float(np.clip(action[0], -3.0, 3.0))
        x = np.array([x[0] + dt * x[1], x[1] + dt * u])
        trajectory.append((step, x[0], x[1], u))
    return np.asarray(trajectory)


def rms_position(trajectory):
    return float(np.sqrt(np.mean(trajectory[:, 1] ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--max-delay-steps', type=int, default=3)
    parser.add_argument('--sample-period-steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=4)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    base = ProportionalController()
    direct = simulate(base, n_steps=args.steps)

    networked = NetworkedControllerWrapper(
        ProportionalController(),
        observation_dim=2,
        sample_period_steps=args.sample_period_steps,
        max_delay_steps=args.max_delay_steps,
        delay_sampler=uniform_delay(args.max_delay_steps, rng=rng),
        trigger=DynamicPETCTrigger(
            error_weight=1.0,
            signal_weight=0.03,
            eta_decay=0.02,
            eta_growth=0.02,
            eta_skip_gain=0.15,
            eta_transmit_drop=0.3,
            eta_max=0.5,
        ),
    )
    networked.reset_before_run(obs=np.array([2.0, 0.0]))
    delayed = simulate(networked, n_steps=args.steps)

    print('direct_rms_position:', f'{rms_position(direct):.4f}')
    print('networked_rms_position:', f'{rms_position(delayed):.4f}')
    print('networked_metrics:', networked.metrics.summary())


if __name__ == '__main__':
    main()
