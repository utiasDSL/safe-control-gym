import sys, os
import pytest
import munch


from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.hyperparameters.database import create, drop

@pytest.mark.parametrize('ALGO',['ppo', 'sac'])
def test_hpo_database(ALGO):

    # create database
    create(munch.Munch({'tag': f'{ALGO}_hpo'}))

    # drop database
    drop(munch.Munch({'tag': f'{ALGO}_hpo'}))