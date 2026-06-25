'''Mobile robot MPC path-tracking demo with a dynamic PETC wrapper.'''

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safe_control_gym.controllers.networked import DynamicPETCTrigger, NetworkedControllerWrapper, uniform_delay


class ShootingMPCController:
    '''Derivative-free finite-horizon MPC for a unicycle robot.

    The implementation is intentionally dependency-light: it samples bounded
    control sequences, rolls out the unicycle dynamics, and applies the first
    action from the lowest-cost sequence.
    '''

    def __init__(
        self,
        path,
        dt=0.05,
        horizon=12,
        candidates=384,
        speed_bounds=(0.0, 1.0),
        omega_bounds=(-2.2, 2.2),
        seed=3,
    ):
        self.path = np.asarray(path, dtype=float)
        self.dt = float(dt)
        self.horizon = int(horizon)
        self.candidates = int(candidates)
        self.speed_bounds = tuple(speed_bounds)
        self.omega_bounds = tuple(omega_bounds)
        self.rng = np.random.default_rng(seed)
        self._warm_start = np.column_stack([
            np.full(self.horizon, 0.65),
            np.zeros(self.horizon),
        ])

    def reset(self):
        self._warm_start[:, 0] = 0.65
        self._warm_start[:, 1] = 0.0

    def select_action(self, obs, info=None):
        state = np.asarray(obs, dtype=float).reshape(-1)
        sequences = self._sample_sequences()
        costs = np.array([self._rollout_cost(state, seq) for seq in sequences])
        best = sequences[int(np.argmin(costs))]
        self._warm_start[:-1] = best[1:]
        self._warm_start[-1] = best[-1]
        return best[0].copy()

    def close(self):
        return None

    def _sample_sequences(self):
        sequences = np.empty((self.candidates, self.horizon, 2), dtype=float)
        sequences[0] = self._warm_start
        sequences[1] = np.column_stack([np.full(self.horizon, 0.65), np.zeros(self.horizon)])

        v_min, v_max = self.speed_bounds
        w_min, w_max = self.omega_bounds
        base_v = self._warm_start[:, 0]
        base_w = self._warm_start[:, 1]
        for idx in range(2, self.candidates):
            noise_scale = 0.16 if idx < self.candidates // 2 else 0.35
            v = np.clip(base_v + self.rng.normal(0.0, noise_scale, self.horizon), v_min, v_max)
            w = np.clip(base_w + self.rng.normal(0.0, noise_scale * 3.0, self.horizon), w_min, w_max)
            sequences[idx, :, 0] = v
            sequences[idx, :, 1] = w
        return sequences

    def _rollout_cost(self, state, sequence):
        x = state.copy()
        total = 0.0
        previous = sequence[0]
        for k, action in enumerate(sequence):
            x = step_unicycle(x, action, self.dt)
            err, heading_err = path_error_and_heading(x, self.path)
            v, omega = action
            smooth = np.sum((action - previous) ** 2)
            total += (1.0 + 0.04 * k) * (10.0 * err ** 2 + 0.55 * heading_err ** 2)
            total += 0.02 * omega ** 2 + 0.03 * smooth
            total += 0.08 * max(0.0, 0.35 - v) ** 2
            previous = action
        goal_distance = np.linalg.norm(x[:2] - self.path[-1])
        total += 0.4 * goal_distance
        return float(total)


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def step_unicycle(state, action, dt):
    x, y, theta = state
    v, omega = action
    return np.array([
        x + dt * v * np.cos(theta),
        y + dt * v * np.sin(theta),
        wrap_angle(theta + dt * omega),
    ])


def make_path(n=500):
    xs = np.linspace(0.0, 10.0, n)
    ys = 1.1 * np.sin(0.55 * xs)
    return np.column_stack([xs, ys])


def path_error_and_heading(state, path):
    point = state[:2]
    deltas = path - point
    distances = np.linalg.norm(deltas, axis=1)
    idx = int(np.argmin(distances))
    nearest_error = float(distances[idx])
    next_idx = min(idx + 5, len(path) - 1)
    tangent = path[next_idx] - path[max(0, idx - 1)]
    path_heading = np.arctan2(tangent[1], tangent[0])
    heading_error = abs(wrap_angle(path_heading - state[2]))
    return nearest_error, float(heading_error)


def simulate(controller, path, n_steps=260, dt=0.05):
    state = np.array([0.0, -0.25, 0.15])
    rows = []
    for step in range(n_steps):
        action = np.asarray(controller.select_action(state, info={'current_step': step}), dtype=float)
        action = np.array([np.clip(action[0], 0.0, 1.0), np.clip(action[1], -2.2, 2.2)])
        state = step_unicycle(state, action, dt)
        err, heading_err = path_error_and_heading(state, path)
        rows.append({
            'step': step,
            'time': step * dt,
            'x': state[0],
            'y': state[1],
            'theta': state[2],
            'v': action[0],
            'omega': action[1],
            'tracking_error': err,
            'heading_error': heading_err,
        })
    return rows


def summarize(rows):
    errors = np.array([row['tracking_error'] for row in rows], dtype=float)
    omega = np.array([row['omega'] for row in rows], dtype=float)
    return {
        'rms_error_m': float(np.sqrt(np.mean(errors ** 2))),
        'max_error_m': float(np.max(errors)),
        'final_error_m': float(errors[-1]),
        'mean_abs_omega_rad_s': float(np.mean(np.abs(omega))),
    }


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_result(out_path, path, direct_rows, networked_rows, network_events):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    direct_xy = np.array([[row['x'], row['y']] for row in direct_rows])
    network_xy = np.array([[row['x'], row['y']] for row in networked_rows])
    direct_error = np.array([row['tracking_error'] for row in direct_rows])
    network_error = np.array([row['tracking_error'] for row in networked_rows])
    time = np.array([row['time'] for row in direct_rows])

    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    axs[0, 0].plot(path[:, 0], path[:, 1], 'k--', label='reference')
    axs[0, 0].plot(direct_xy[:, 0], direct_xy[:, 1], label='fresh MPC')
    axs[0, 0].plot(network_xy[:, 0], network_xy[:, 1], label='PETC MPC')
    axs[0, 0].axis('equal')
    axs[0, 0].set_title('Mobile robot MPC path tracking')
    axs[0, 0].set_xlabel('x [m]')
    axs[0, 0].set_ylabel('y [m]')
    axs[0, 0].legend()

    axs[0, 1].plot(time, direct_error, label='fresh MPC')
    axs[0, 1].plot(time, network_error, label='PETC MPC')
    axs[0, 1].set_title('Tracking error')
    axs[0, 1].set_xlabel('time [s]')
    axs[0, 1].set_ylabel('nearest-path error [m]')
    axs[0, 1].legend()

    event_steps = np.array([event['step'] for event in network_events])
    transmitted = np.array([event['transmitted'] for event in network_events])
    eta = np.array([event['eta'] for event in network_events])
    stale = np.array([event['stale_steps'] for event in network_events])
    axs[1, 0].step(event_steps, transmitted, where='post')
    axs[1, 0].set_ylim(-0.1, 1.1)
    axs[1, 0].set_title('Transmission decisions')
    axs[1, 0].set_xlabel('step')
    axs[1, 0].set_ylabel('transmit')

    axs[1, 1].plot(event_steps, eta, label='eta')
    axs[1, 1].plot(event_steps, stale, label='stale steps')
    axs[1, 1].set_title('Trigger memory and observation age')
    axs[1, 1].set_xlabel('step')
    axs[1, 1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=260)
    parser.add_argument('--max-delay-steps', type=int, default=4)
    parser.add_argument('--sample-period-steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--output-dir', default=r'C:\My_project_usage\mobile_robot_mpc_petc_output')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = make_path()
    dt = 0.05

    direct = ShootingMPCController(path, dt=dt, seed=args.seed)
    direct_rows = simulate(direct, path, n_steps=args.steps, dt=dt)

    networked = NetworkedControllerWrapper(
        ShootingMPCController(path, dt=dt, seed=args.seed),
        observation_dim=3,
        sample_period_steps=args.sample_period_steps,
        max_delay_steps=args.max_delay_steps,
        delay_sampler=uniform_delay(args.max_delay_steps, rng=np.random.default_rng(args.seed + 1)),
        trigger=DynamicPETCTrigger(
            error_weight=1.0,
            signal_weight=0.018,
            eta_decay=0.012,
            eta_growth=0.023,
            eta_skip_gain=0.075,
            eta_transmit_drop=0.35,
            eta_max=0.32,
        ),
    )
    networked.reset_before_run(obs=np.array([0.0, -0.25, 0.15]))
    networked_rows = simulate(networked, path, n_steps=args.steps, dt=dt)

    write_csv(output_dir / 'fresh_mpc.csv', direct_rows)
    write_csv(output_dir / 'petc_mpc.csv', networked_rows)
    write_csv(output_dir / 'network_events.csv', networked.metrics.events)
    plot_result(output_dir / 'mobile_robot_mpc_petc_result.png', path, direct_rows, networked_rows, networked.metrics.events)

    print('fresh_mpc:', summarize(direct_rows))
    print('petc_mpc:', summarize(networked_rows))
    print('network_metrics:', networked.metrics.summary())
    print('plot:', output_dir / 'mobile_robot_mpc_petc_result.png')


if __name__ == '__main__':
    main()
