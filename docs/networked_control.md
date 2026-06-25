# Networked Controller Wrapper

This module adds a controller wrapper for sampled, delayed, bandwidth-limited
communication. It does not replace LQR, MPC, PID, or learning controllers. It
wraps them and changes which observation reaches the controller at each step.

## Theory Mapping

- `max_delay_steps` maps to the maximum allowable delay number in sampling.
- `DelayChannel.queue_depth` maps to the number of transmitted samples waiting
  for delivery.
- `sampled_signal` is the source-side sampled signal.
- `held_signal` is the destination-side signal used by the wrapped controller.
- `update_error = held_signal - sampled_signal`.
- `DynamicPETCTrigger.eta` is the dynamic event-trigger memory.

## Quick Start

```python
from safe_control_gym.controllers.networked import (
    DynamicPETCTrigger,
    NetworkedControllerWrapper,
    uniform_delay,
)

ctrl = make(config.algo, env_func, **config.algo_config)
networked_ctrl = NetworkedControllerWrapper(
    ctrl,
    observation_dim=env.observation_space.shape[0],
    sample_period_steps=1,
    max_delay_steps=3,
    delay_sampler=uniform_delay(3),
    trigger=DynamicPETCTrigger(error_weight=1.0, signal_weight=0.03),
)
```

Use `networked_ctrl` anywhere a safe-control-gym controller instance is accepted.

## Before/After Evaluation

Run three conditions with the same task, seed, and base controller:

1. Direct controller with fresh observations.
2. Periodic controller with fixed delayed transmissions.
3. Dynamic PETC wrapper with the same delay sampler.

Report:

- task return or tracking error
- transmission count
- event rate
- mean stale steps
- max queue depth
- safety or constraint violations when the task exposes them

## Example

```bash
python examples/networked_control/run_dynamic_petc.py --max-delay-steps 3
```

The example uses a small local double-integrator task. It is a smoke test for
the wrapper, not a reproduction of a paper simulation.

To run a configured safe-control-gym controller behind the wrapper:

```bash
python examples/networked_control/networked_lqr_experiment.py \
  --algo lqr \
  --task cartpole \
  --overrides ./examples/lqr/config_overrides/cartpole/cartpole_stab.yaml \
              ./examples/lqr/config_overrides/cartpole/lqr_cartpole_stab.yaml
```

The integration point is intentionally small: create the normal controller, then
wrap it with `NetworkedControllerWrapper` before constructing `BaseExperiment`.

## Mobile Robot MPC Demo

The mobile robot MPC example demonstrates a concrete robotics use case: a
unicycle/differential-drive robot tracks a smooth path while localization
observations arrive through the dynamic PETC wrapper.

```bash
python examples/networked_control/run_mobile_robot_mpc_petc.py \
  --steps 260 \
  --max-delay-steps 4
```

The controller is a dependency-light finite-horizon MPC implemented by sampling
bounded control sequences and rolling out the kinematic model. It uses:

```text
state: [x, y, theta]
control: [linear velocity, angular velocity]
speed bounds: [0.0, 1.0] m/s
angular velocity bounds: [-2.2, 2.2] rad/s
horizon: 12 steps
candidate sequences: 384
```

One representative run produced:

```text
Fresh MPC RMS tracking error: 0.0513 m
PETC MPC RMS tracking error: 0.0565 m
Transmissions: 75 / 260 samples
Event rate: 28.85%
Mean stale steps: 4.22
Max queue depth: 4
```

This illustrates the intended engineering tradeoff: the wrapped MPC used about
29% of the observation transmissions while keeping the RMS path-tracking error
within 5.3 mm of the fresh-observation baseline in this scenario.

## Limits

The wrapper exposes certificate-like quantities such as update error, queue
depth, and trigger margin. It does not claim plant-level input-to-state
stability unless the user supplies compatible Lyapunov/error weights for the
task.
