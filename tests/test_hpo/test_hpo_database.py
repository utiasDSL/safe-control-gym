import os
import sys

import munch
import pytest

from safe_control_gym.hyperparameters.database import create, drop
from safe_control_gym.utils.configuration import ConfigFactory


@pytest.mark.parametrize('ALGO', ['ppo', 'sac'])
def test_hpo_database(ALGO):

    # create database
    create(munch.Munch({'tag': f'{ALGO}_hpo'}))

    # drop database
    drop(munch.Munch({'tag': f'{ALGO}_hpo'}))
