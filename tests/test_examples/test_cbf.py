import sys

import pytest

from examples.cbf.cbf_experiment import run


@pytest.mark.parametrize('SYS',           ['cartpole'])
@pytest.mark.parametrize('ALGO',          ['ppo', 'sac'])
@pytest.mark.parametrize('SAFETY_FILTER', ['cbf', 'cbf_nn'])
def test_cbf(SYS, ALGO, SAFETY_FILTER):
    sys.argv[1:] = ['--algo', ALGO,
                    '--task', SYS,
                    '--safety filter', SAFETY_FILTER,
                    '--overrides',
                        f'./examples/cbf/config_overrides/{SYS}_config.yaml',
                        f'./examples/cbf/config_overrides/{ALGO}_config.yaml',
                        f'./examples/cbf/config_overrides/{SAFETY_FILTER}_config.yaml',
                    '--kv_overrides',
                        'sf_config.max_num_steps=10',
                        'sf_config.num_episodes=2',
                        'sf_config.train_iterations=10']

    run(plot=False, training=True, n_episodes=None, n_steps=10, curr_path='./examples/cbf', save_data=False)
