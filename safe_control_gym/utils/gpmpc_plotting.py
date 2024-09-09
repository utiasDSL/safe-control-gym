'''GP-MPC lotting utilities.'''

import os

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from safe_control_gym.utils.utils import mkdirs
from safe_control_gym.controllers.mpc.mpc_utils import compute_state_rmse


def get_runtime(test_runs, train_runs):
    ''' get the mean, std, and max runtime '''
    # NOTE: only implemented for single episode
    # NOTE: the first step is popped out because of the ipopt initial guess

    num_epochs = len(train_runs.keys())
    # num_episode = len(train_runs[0].keys())
    num_train_samples_by_epoch = [] # number of training data
    # steps_per_episode = len(test_runs[0][0]['runtime'])
    num_steps_max = np.max([len(test_runs[epoch][0]['runtime']) for epoch in range(num_epochs)])

    mean_runtime = np.zeros(num_epochs)
    std_runtime = np.zeros(num_epochs)
    max_runtime = np.zeros(num_epochs)

    runtime = [] 
    for epoch in range(num_epochs):
        num_samples =  len(train_runs[epoch].keys())
        num_train_samples_by_epoch.append(num_samples)
        # num_steps = len(test_runs[epoch][0]['runtime'])
        runtime = test_runs[epoch][0]['runtime'][1:] # remove the first step
        
        mean_runtime[epoch] = np.mean(runtime)
        std_runtime[epoch] = np.std(runtime)
        max_runtime[epoch] = np.max(runtime)

    runtime_result = {'mean': mean_runtime, 'std': std_runtime, 'max': max_runtime, 'num_train_samples': num_train_samples_by_epoch}
    
    return runtime_result

def plot_runtime(runtime, num_points_per_epoch, dir):
    mean_runtime = runtime['mean']
    std_runtime = runtime['std']
    max_runtime = runtime['max']
    # num_train_samples = runtime['num_train_samples']
    plt.plot(num_points_per_epoch, mean_runtime, label='mean')
    plt.fill_between(num_points_per_epoch, mean_runtime - std_runtime, mean_runtime + std_runtime, alpha=0.3, label='1-std')
    plt.plot(num_points_per_epoch, max_runtime, label='max', color='r')
    plt.legend()
    plt.xlabel('Train Steps')
    plt.ylabel('Runtime (s) ')
    stem = 'runtime'
    fname = os.path.join(dir, stem + '.png')
    plt.savefig(fname)
    plt.cla()
    plt.clf()
    data = np.vstack((num_points_per_epoch, mean_runtime, std_runtime, max_runtime)).T
    fname = os.path.join(dir, stem + '.csv')
    np.savetxt( fname,
                data,
                delimiter=',',
                header='Train Steps, Mean, Std, Max')

def plot_runs(all_runs, num_epochs, episode=0, ind=0, ylabel='x position', dir=None, traj = None):
    # plot the reference trajectory
    if traj is not None:
        plt.plot(traj[:, ind], label='Reference', color='gray', linestyle='--')
    # plot the prior controller
    plt.plot(all_runs[0][episode]['state'][:, ind], label='Linear MPC')
    # plot each learning epoch
    for epoch in range(1, num_epochs):
        # plot the first episode of each epoch
        plt.plot(all_runs[epoch][episode]['state'][:, ind], label='GP-MPC %s' % epoch)
    plt.title(ylabel)
    plt.xlabel('Step')
    plt.ylabel(ylabel)
    plt.legend()
    save_str = 'ep%s_ind%s_state.png' % (episode, ind)
    if dir is not None:
        save_str = os.path.join(dir, save_str)
        plt.savefig(save_str)
    else:
        plt.show()
    plt.cla()
    plt.clf()

def plot_runs_input(all_runs, num_epochs, episode=0, ind=0, ylabel='x position', dir=None,):
    # plot the prior controller
    plt.plot(all_runs[0][episode]['action'][:, ind], label='Linear MPC')
    # plot each learning epoch
    for epoch in range(1, num_epochs):
        # plot the first episode of each epoch
        plt.plot(all_runs[epoch][episode]['action'][:, ind], label='GP-MPC %s' % epoch)
    plt.title(ylabel)
    plt.xlabel('Step')
    plt.ylabel(ylabel)
    plt.legend()
    save_str = 'ep%s_ind%s_action.png' % (episode, ind)
    if dir is not None:
        save_str = os.path.join(dir, save_str)
        plt.savefig(save_str)
    else:
        plt.show()
    plt.cla()
    plt.clf()

def get_constraint_violations(test_runs,
                              train_runs):
    num_train_samples_by_epoch = []
    violations_per_epoch = []
    max_violations_per_epoch = []
    mean_violations_per_epoch = []
    num_samples = 0
    num_epochs = len(train_runs.keys())
    n_train_samples_per_epoch = 0
    for epoch in range(num_epochs):
        violations_per_episode = []
        max_violations_per_episode = []
        num_train_samples_per_episode = len(train_runs[epoch].keys())
        # get number of training samples
        for train_episode in range(num_train_samples_per_episode):
            n_train_samples_per_epoch += len(train_runs[epoch][train_episode]['info'])
        num_test_episodes_per_epoch = len(test_runs[epoch].keys())
        for test_episode in range(num_test_episodes_per_epoch):
            # violations in a single episode
            violations = 0
            max_violations = np.zeros(test_runs[epoch][test_episode]['info'][0]['constraint_values'].shape)
            n = len(test_runs[epoch][test_episode]['info']) # number of samples in episode
            for i in range(n):
                # print('test_runs[epoch][test_episode][info][i][constraint_values]', test_runs[epoch][test_episode]['info'][i]['constraint_values'])
                # input('press enter to continue')
                #violations += test_runs[epoch][test_episode]['info'][i]['constraint_violation'] # Due to bug.
                violations += int(np.any(test_runs[epoch][test_episode]['info'][i]['constraint_values'] > 0))
                max_violations = np.maximum(max_violations,
                                            test_runs[epoch][test_episode]['info'][i]['constraint_values'])

            violations_per_episode.append(violations)
            max_violations_per_episode.append(max_violations)
        num_train_samples_by_epoch.append(n_train_samples_per_epoch)
        violations_per_epoch.append(violations_per_episode)
        max_violations_per_epoch.append(np.vstack(max_violations_per_episode))
        mean_violations_per_epoch.append(np.mean(violations_per_episode))
        num_samples += n
    return num_train_samples_by_epoch, violations_per_epoch, mean_violations_per_epoch, max_violations_per_epoch

def plot_constraint_violation(viol_samp, viols, dir):
    violations = np.array(viols)
    train_time = np.array(viol_samp)
    mean_viol = np.mean(violations, axis=1)
    max = np.max(violations, axis=1)
    min = np.min(violations, axis=1)

    plt.plot(train_time, mean_viol, label='mean')
    plt.plot(train_time, max, label='max')
    plt.plot(train_time, min, label='min')
    plt.legend()
    plt.xlabel('Train Steps')
    plt.ylabel('Number of violations')
    stem = 'number_viol'
    fname = os.path.join(dir, stem + '.png')
    plt.savefig(fname)
    plt.cla()
    plt.clf()

    data = np.vstack((train_time, mean_viol, max, min)).T
    fname = os.path.join(dir, stem + '.csv')
    np.savetxt( fname,
                data,
                delimiter=',',
                header='Train Steps (s),Mean,Max,Min')
    
def get_average_rmse_error(runs):
    num_epochs = len(runs)
    num_episodes = len(runs[0])
    costs = np.zeros((num_epochs, num_episodes))
    for epoch in range(num_epochs):
        for episode in range(num_episodes):
            mse, rmse = runs[epoch][episode]['total_rmse_state_error']
            costs[epoch, episode] = rmse

    mean_cost = np.mean(costs, axis=1)
    return mean_cost

def plot_learning_curve(avg_rewards, num_points_per_epoch, stem, dir):
    samples = num_points_per_epoch # data number
    rewards = np.array(avg_rewards)
    plt.plot(samples, rewards)
    plt.title('Avg Episode' + stem)
    plt.xlabel('Training Steps')
    plt.ylabel(stem)
    save_str = os.path.join(dir, stem + '.png')
    plt.savefig(save_str)
    plt.cla()
    plt.clf()
    data = np.vstack((samples, rewards)).T
    fname = os.path.join(dir, stem + '.csv')
    header = 'Train steps,Cost'
    np.savetxt(fname,
               data,
               delimiter=',',
               header=header)

def get_cost(test_runs):
    num_epochs = len(test_runs)
    num_episodes = len(test_runs[0])
    costs = np.zeros((num_epochs, num_episodes))
    for epoch in range(num_epochs):
        for episode in range(num_episodes):
            cost = np.sum(test_runs[epoch][episode]['obs'] ** 2)
            costs[epoch, episode] = cost
    mean_cost = np.mean(costs, axis=1)
    return mean_cost

def get_quad_average_rmse_error(runs, ref):
    num_epochs = len(runs)
    num_episodes = len(runs[0])
    costs = np.zeros((num_epochs, num_episodes))
    for epoch in range(num_epochs):
        for episode in range(num_episodes):
            mse, rmse = runs[epoch][episode]['total_rmse_state_error']
            costs[epoch, episode] = rmse

    mean_cost = np.mean(costs, axis=1)
    return mean_cost

def get_quad_cost(test_runs, ref):
    num_epochs = len(test_runs)
    num_episodes = len(test_runs[0])
    test_runs = match_runs_by_append_last(test_runs, ref)
    costs = np.zeros((num_epochs, num_episodes))
    for epoch in range(num_epochs):
        for episode in range(num_episodes):
            cost = np.sum((test_runs[epoch][episode]['obs'][:,[0,2]] - ref[:,[0,2]]) ** 2)
            costs[epoch, episode] = cost
    mean_cost = np.mean(costs, axis=1)
    return mean_cost

def get_quad_average_rmse_error_xz_only(runs, ref):
    num_epochs = len(runs)
    num_episodes = len(runs[0])
    runs = match_runs_by_append_last(runs, ref)
    costs = np.zeros((num_epochs, num_episodes))
    for epoch in range(num_epochs):
        for episode in range(num_episodes):
            mse, rmse = compute_state_rmse(runs[epoch][episode]['state'][:,[0,2]] - ref[:,[0,2]])
            costs[epoch, episode] = rmse

    mean_cost = np.mean(costs, axis=1)
    return mean_cost

def plot_xz_trajectory(runs, ref, dir):
    num_epochs = len(runs)
    plt.figure()
    plt.plot(ref[:, 0], ref[:, 2], label='Reference', color='gray', linestyle='--')
    plt.plot(runs[0][0]['obs'][:, 0], runs[0][0]['obs'][:, 2], label='Linear MPC')
    for epoch in range(1, num_epochs):
        plt.plot(runs[epoch][0]['obs'][:, 0], runs[epoch][0]['obs'][:, 2], label='GP-MPC %s' % epoch)
    plt.title('X-Z plane path')
    plt.xlabel('X [m]')
    plt.ylabel('Z [m]')
    plt.legend()
    save_str = os.path.join(dir, 'xz_path.png')
    plt.savefig(save_str)
    plt.cla()
    plt.clf()

def make_plots(test_runs, train_runs, dir):
    nx = test_runs[0][0]['state'].shape[1]
    nu = test_runs[0][0]['action'].shape[1]
    num_epochs = len(test_runs)
    num_episodes = len(test_runs[0])
    fig_dir = os.path.join(dir,'figs')
    mkdirs(fig_dir)

    # Make plot of all trajectories.
    num_points_per_epoch = []
    for episode_i in range(num_episodes):
        for ind in range(nx):
            ylabel = 'x%s' % ind
            plot_runs(test_runs, num_epochs, episode=episode_i, ind=ind, ylabel=ylabel, dir=fig_dir)
        for ind in range(nu):
            ylabel = 'u%s' % ind
            plot_runs_input(test_runs, num_epochs, episode=episode_i, ind=ind, ylabel=ylabel, dir=fig_dir)

    # Compute the number of training points (x-axis for most figures).
    num_points = 0
    num_points_per_epoch.append(num_points)
    for epoch in range(1, num_epochs):
        num_train_episodes = len(train_runs[epoch])
        for episode in range(num_train_episodes):
            num_points += train_runs[epoch][episode]['obs'].shape[0]
        num_points_per_epoch.append(num_points)

    # Plot violation data
    # TODO: check why this breaks the evaluation
    nsamp, viol, mean_viol, maximums = get_constraint_violations(test_runs,
                                                                 train_runs)
    plot_constraint_violation(num_points_per_epoch, viol, fig_dir)

    # Plot Learning Curves
    avg_rmse_error = get_average_rmse_error(test_runs)
    plot_learning_curve(avg_rmse_error, num_points_per_epoch, 'avg_rmse_cost_learning_curve', fig_dir)
    common_costs = get_cost(test_runs)
    plot_learning_curve(common_costs, num_points_per_epoch, 'common_cost_learning_curve', fig_dir)


def make_quad_plots(test_runs, train_runs, trajectory, dir):
    nx = test_runs[0][0]['state'].shape[1]
    nu = test_runs[0][0]['action'].shape[1]
    # trim the traj steps to mach the evaluation steps
    num_steps = test_runs[0][0]['obs'].shape[0]
    trajectory = trajectory[0:num_steps, :]
    num_epochs = len(test_runs)
    num_episodes = len(test_runs[0])
    fig_dir = os.path.join(dir,'figs')
    mkdirs(fig_dir)
    num_points_per_epoch = []
    for episode_i in range(num_episodes):
        plot_xz_trajectory(test_runs, trajectory, fig_dir)
        for ind in range(nx):
            ylabel = 'x%s' % ind
            plot_runs(test_runs, num_epochs, episode=episode_i, ind=ind, ylabel=ylabel, dir=fig_dir, traj=trajectory)
        for ind in range(nu):
            ylabel = 'u%s' % ind
            plot_runs_input(test_runs, num_epochs, episode=episode_i, ind=ind, ylabel=ylabel, dir=fig_dir)
    num_points = 0
    num_points_per_epoch.append(num_points)
    for epoch in range(1, num_epochs):
        num_train_episodes = len(train_runs[epoch])
        for episode in range(num_train_episodes):
            num_points += train_runs[epoch][episode]['obs'].shape[0]
        num_points_per_epoch.append(num_points)

    # plot learning curves
    common_costs = get_quad_cost(test_runs, trajectory)
    plot_learning_curve(common_costs, num_points_per_epoch, 'common_xz_cost_learning_curve', fig_dir)
    rmse_error = get_quad_average_rmse_error(test_runs, trajectory)
    plot_learning_curve(rmse_error, num_points_per_epoch, 'rmse_error_learning_curve', fig_dir)
    rmse_error_xz = get_quad_average_rmse_error_xz_only(test_runs, trajectory)
    plot_learning_curve(rmse_error_xz, num_points_per_epoch, 'rmse_xz_error_learning_curve', fig_dir)
    runtime_result = get_runtime(test_runs, train_runs)
    plot_runtime(runtime_result, num_points_per_epoch, fig_dir)
    
def match_runs_by_append_last(test_runs, ref):
    num_epochs = len(test_runs)
    num_episodes = len(test_runs[0])
    ref_length = ref.shape[0]
    # if any of the runs has a different length, append the last state to match the length
    for epoch in range(num_epochs):
        for episode in range(num_episodes):
            if test_runs[epoch][episode]['obs'].shape[0] != ref_length:
                last_state = test_runs[epoch][episode]['obs'][-1, :]
                for i in range(ref_length - test_runs[epoch][episode]['obs'].shape[0]):
                    test_runs[epoch][episode]['obs'] = np.vstack((test_runs[epoch][episode]['obs'], last_state))
    return test_runs