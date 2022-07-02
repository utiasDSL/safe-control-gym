"""A simple script to demonstrate safe-control-gym API.

Example:

    $ python3 verbose_api.py --overrides ./verbose_api.yaml

"""
import time
import inspect
import pybullet as p

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


def run():
    """The main function creating, running, and closing an environment.

    """
    # Set iterations and episode counter.
    num_episodes = 1
    ITERATIONS = int(2)
    # Start a timer.
    START = time.time()
    # Create an environment
    CONFIG_FACTORY = ConfigFactory()
    CONFIG_FACTORY.add_argument('--system', type=str, default='cartpole', choices=['cartpole', 'quadrotor'])
    config = CONFIG_FACTORY.merge()
    if config.system == 'cartpole':
        env = make(config.system, **config.cartpole_config)
    elif config.system == 'quadrotor':
        env = make(config.system, **config.quadrotor_config)
    # Reset the environment, obtain and print the initial observations.
    initial_obs, initial_info = env.reset()
    print('\n\n')
    # Dynamics info
    print_str_with_style('PyBullet dynamics info:', 7)
    if config.system == 'cartpole':
        print('\t' + str(p.getDynamicsInfo(bodyUniqueId=env.CARTPOLE_ID, linkIndex=-1, physicsClientId=env.PYB_CLIENT)))
    elif config.system == 'quadrotor':
        print('\t' + str(p.getDynamicsInfo(bodyUniqueId=env.DRONE_IDS[0], linkIndex=-1, physicsClientId=env.PYB_CLIENT)))
    print('\n\n')
    print_str_with_style('Initial reset.\n', 7)
    print_str_with_style('Open AI gym API:', 2)
    print_str_with_style('\tInitial observation: ' + str(initial_obs), 2)
    # 
    print_str_with_style('safe-control-gym API:', 0)
    print_str_with_style('\tA priori symbolic model:', 0)
    out = '\t\tState: ' + str(initial_info['symbolic_model'].x_sym).strip('vertcat')
    print_str_with_style(out, 0)
    out = '\t\tInput: ' + str(initial_info['symbolic_model'].u_sym).strip('vertcat')
    print_str_with_style(out, 0)
    out = '\t\tDynamics: ' + str(initial_info['symbolic_model'].x_dot).strip('vertcat')
    print_str_with_style(out, 0)
    out = '\t\tCost: ' + str(initial_info['symbolic_model'].cost_func).replace('vertcat', '').replace(', (',',\n\t\t\t(').replace(', @',',\n\t\t\t@')
    print_str_with_style(out, 0)
    print_str_with_style('\tConstraints:', 0)
    for fun in initial_info['symbolic_constraints']:
        out = '\t' + str(inspect.getsource(fun)).strip('\n')
        print_str_with_style(out, 0)
    print_str_with_style('\tA priori parameters:', 0)
    out = '\t\t' + str(initial_info['physical_parameters'])
    print_str_with_style(out, 0)
    print_str_with_style('\tX reference:', 0)
    out = '\t\t' + str(initial_info['x_reference'])
    print_str_with_style(out, 0)
    print_str_with_style('\tU reference:', 0)
    out = '\t\t' + str(initial_info['u_reference'])
    print_str_with_style(out, 0)
    print('\n\n')
    # Run an experiment.
    for i in range(ITERATIONS):
        # Step by keyboard input
        # _ = input('Press any key to continue.')
        # Sample a random action.
        action = env.action_space.sample()
        # Step the environment and print all returned information.
        obs, reward, done, info = env.step(action)
        #
        print_str_with_style(str(i)+'-th step.', 7)
        out = '\tApplied action: ' + str(action) + '\n'
        print(out)

        print_str_with_style('Open AI gym API:', 2)
        out = '\tObservation: ' + str(obs)
        print_str_with_style(out, 2)
        out = '\tReward: ' + str(reward)
        print_str_with_style(out, 2)
        out = '\tDone: ' + str(done)
        print_str_with_style(out, 2)

        print_str_with_style('safe-control-gym API:', 0)
        out = '\tConstraints evaluations: ' + str(info['constraint_values'])
        print_str_with_style(out, 0)
        out = '\tConstraints violation: ' + str(bool(info['constraint_violation']))
        print_str_with_style(out, 0)

        print('\n\n')
        # If an episode is complete, reset the environment.
        if done:
            num_episodes += 1
            new_initial_obs, new_initial_info = env.reset()
            print_str_with_style(str(num_episodes)+'-th reset.', 7)
            print_str_with_style('Reset obs' + str(new_initial_obs), 2)
            print_str_with_style('Reset info' + str(new_initial_info), 0)
            print('\n\n------------------------------------------------------------------------------')
            print('------------------------------------------------------------------------------\n\n')

    # Close the environment and print timing statistics.
    env.close()
    elapsed_sec = time.time() - START
    out = str("\n{:d} iterations (@{:d}Hz) and {:d} episodes in {:.2f} seconds, i.e. {:.2f} steps/sec for a {:.2f}x speedup.\n\n"
          .format(ITERATIONS, env.CTRL_FREQ, num_episodes, elapsed_sec, ITERATIONS/elapsed_sec, (ITERATIONS*env.CTRL_TIMESTEP)/elapsed_sec))
    print_str_with_style(out,7)

class bcolors:
    """Support for color output.

    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_str_with_style(string: str='', style: int=0):
    """Function to convert to string and print in color.

    """
    string = str(string)
    if style == 0:
        print(bcolors.HEADER + string + bcolors.ENDC)
    elif style == 1:
        print(bcolors.OKBLUE + string + bcolors.ENDC)
    elif style == 2:
        print(bcolors.OKCYAN + string + bcolors.ENDC)
    elif style == 3:
        print(bcolors.OKGREEN + string + bcolors.ENDC)
    elif style == 4:
        print(bcolors.WARNING + string + bcolors.ENDC)
    elif style == 5:
        print(bcolors.FAIL + string + bcolors.ENDC)
    elif style == 6:
        print(bcolors.BOLD + string + bcolors.ENDC)
    elif style == 7:
        print(bcolors.UNDERLINE + string + bcolors.ENDC)
    else:
        raise ValueError('[ERROR] unknown style!')


if __name__ == "__main__":
    run()
