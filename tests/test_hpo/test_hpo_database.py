import munch
import pytest

from safe_control_gym.hyperparameters.database import create, drop


@pytest.mark.parametrize('ALGO', ['ppo', 'sac', 'gp_mpc'])
def test_hpo_database(ALGO):
    pytest.skip('Requires MySQL Database to be running.')

    # create database
    create(munch.Munch({'tag': f'{ALGO}_hpo'}))

    # drop database
    drop(munch.Munch({'tag': f'{ALGO}_hpo'}))
