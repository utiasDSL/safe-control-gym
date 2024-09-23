'''This script already assumes that mysql server is up and hard
coded user 'optuna' without password was added.
'''

import mysql.connector

from safe_control_gym.utils.configuration import ConfigFactory


def create(config):
    '''This function is used to create database named after --Tag.'''

    db = mysql.connector.connect(
        host='localhost',
        user='optuna',
    )

    mycursor = db.cursor()

    mycursor.execute(f'CREATE DATABASE IF NOT EXISTS {config.tag}')


def drop(config):
    '''This function is used to drop database named after --Tag.
       Be sure to backup before dropping.
       * Backup: mysqldump --no-tablespaces -u optuna DATABASE_NAME > DATABASE_NAME.sql
       * Restore:
                  1. mysql -u optuna -e 'create database DATABASE_NAME'.
                  2. mysql -u optuna DATABASE_NAME < DATABASE_NAME.sql
    '''

    db = mysql.connector.connect(
        host='localhost',
        user='optuna',
    )

    mycursor = db.cursor()

    mycursor.execute(f'drop database if exists {config.tag}')


MAIN_FUNCS = {'drop': drop, 'create': create}


if __name__ == '__main__':
    fac = ConfigFactory()
    fac.add_argument('--func', type=str, default='create', help='main function to run.')
    config = fac.merge()

    func = MAIN_FUNCS.get(config.func, None)
    if func is None:
        raise Exception(f'Main function {config.func} not supported.')
    func(config)
