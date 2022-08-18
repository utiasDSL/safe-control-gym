import sys
import pytest

from experiments.cbf.cbf_experiment import run


@pytest.mark.parametrize('SYS', ['cartpole'])
@pytest.mark.parametrize('ALGO',['ppo', 'sac'])
@pytest.mark.parametrize('SAFETY_FILTER',['cbf', 'cbf_nn'])
def test_cbf(SYS, ALGO, SAFETY_FILTER):
    sys.argv[1:] = ['--algo', ALGO,
                    '--task', SYS,
                    '--safety filter', SAFETY_FILTER,
                    '--overrides',
                        f'./config_overrides/{SYS}_config.yaml',
                        f'./config_overrides/{ALGO}_config.yaml',
                        f'./config_overrides/{SAFETY_FILTER}_config.yaml']
    run(plot=False, training=True, n_episodes=None, n_steps=10)
