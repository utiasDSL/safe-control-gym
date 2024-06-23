from safe_control_gym.envs.physics import PhysicsMode
import gymnasium
import pytest
from pathlib import Path
import toml
from functools import partial
from munch import munchify
from lsy_drone_racing.constants import FIRMWARE_FREQ


@pytest.mark.parametrize("physics", PhysicsMode)
def test_sim(physics: PhysicsMode):
    with open(Path(__file__).parent / "config/test_sim.toml") as f:
        config = munchify(toml.load(f))
    config.sim.physics = physics  # override physics mode
    env = gymnasium.make("drone_racing-v0", config=config)
    env.reset()
    while True:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break
