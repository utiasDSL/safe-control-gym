import sys, os
import pytest
import munch


from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.hyperparameters.database import create, drop


def test_hpo_gpmpc_database():

    # create database
    create(munch.Munch({'tag': 'gp_mpc_hpo'}))

    # drop database
    drop(munch.Munch({'tag': 'gp_mpc_hpo'}))