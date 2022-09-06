import sys

from experiments.annual_reviews.figure8.mpsc_experiment import run


def test_figure8():
    sys.argv[1:] = ['--algo', 'ppo',
                    '--task', 'cartpole',
                    '--safety_filter', 'linear_mpsc',
                    '--overrides',
                        './experiments/annual_reviews/figure8/config_overrides/mpsc_config.yaml',
                        './experiments/annual_reviews/figure8/config_overrides/ppo_config.yaml',
                        './experiments/annual_reviews/figure8/config_overrides/cartpole_config.yaml']
    run(plot=False, training=False, n_steps=5, curr_path='./experiments/annual_reviews/figure8')
