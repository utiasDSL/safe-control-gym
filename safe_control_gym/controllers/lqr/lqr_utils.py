"""Linear Quadratic Regulator (LQR) utilities

"""
import numpy as np
import scipy.linalg
from termcolor import colored
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def get_arr_str(vector, str_format_arg=".2f"):
    str_format = "{:%s}" % str_format_arg
    vector_str = [str_format.format(i) for i in vector]
    vector_str_joined = " ".join(vector_str)
    return vector_str_joined


def compute_lqr_gain(model, x_0, u_0, Q, R, discrete_dynamics=True):
    # Linearization.
    df = model.df_func(x_0, u_0)
    A, B = df[0].toarray(), df[1].toarray()

    # Compute controller gain.
    if discrete_dynamics:
        # x[k+1] = A x[k] + B u[k]
        A, B = discretize_linear_system(A, B, model.dt)
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        btp = np.dot(B.T, P)
        gain = np.dot(np.linalg.inv(R + np.dot(btp, B)),
                      np.dot(btp, A))
    else:
        # dx/dt = A x + B u
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        gain = np.dot(np.linalg.inv(R), np.dot(B.T, P))

    return gain


def discretize_linear_system(A, B, dt, exact=False):
    """ discretization of a linear system 
    
    dx/dt = A x + B u 
    --> xd[k+1] = Ad xd[k] + Bd ud[k] where xd[k] = x(k*dt)

    Args:
        A: np.array, system transition matrix  
        B: np.array, input matrix 
        dt: scalar, step time interval 
        exact: bool, if to use exact discretization 

    Returns:
        discretized matrices Ad, Bd 

    """
    state_dim, input_dim = A.shape[1], B.shape[1]

    if exact:
        M = np.zeros((state_dim + input_dim, state_dim + input_dim))
        M[:state_dim, :state_dim] = A
        M[:state_dim, state_dim:] = B

        Md = scipy.linalg.expm(M * dt)
        Ad = Md[:state_dim, :state_dim]
        Bd = Md[:state_dim, state_dim:]
    else:
        I = np.eye(state_dim)
        Ad = I + A * dt
        Bd = B * dt

    return Ad, Bd


def get_cost_weight_matrix(weights, dim):
    """Gets weight matrix from input args.

    """
    if len(weights) == dim:
        W = np.diag(weights)
    elif len(weights) == 1:
        W = np.diag(weights * dim)
    else:
        raise Exception("Wrong dimension for cost weights.")
    return W

def post_analysis(goal_stack, state_stack, input_stack, env,
                  ite_counter, ep_counter, plot_traj, save_plot, save_data,
                  plot_dir, data_dir):
    # Get model
    model = env.symbolic
    stepsize = model.dt

    # Get times
    plot_length = np.min([np.shape(goal_stack)[0], np.shape(state_stack)[0]])
    times = np.linspace(0, stepsize * plot_length, plot_length)

    # Plot states
    fig, axs = plt.subplots(model.nx)
    if model.nx == 1:
        axs = [axs]
    for k in range(model.nx):
        axs[k].plot(times, state_stack.transpose()[k, 0:plot_length], label='actual')
        axs[k].plot(times, goal_stack.transpose()[k, 0:plot_length], color='r', label='desired')
        axs[k].set(ylabel=env.STATE_LABELS[k] + '\n[%s]' % env.STATE_UNITS[k])
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if not (k == model.nx - 1):
            axs[k].set_xticks([])
    axs[0].set_title('State Trajectories')
    axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc="lower right")
    axs[-1].set(xlabel='time (sec)')
    if save_plot:
        plt.savefig(plot_dir + "state_ite%d" % ite_counter)

    # Plot inputs
    _, axs = plt.subplots(model.nu)
    if model.nu == 1:
        axs = [axs]
    for k in range(model.nu):
        axs[k].plot(times, input_stack.transpose()[k, 0:plot_length])
        axs[k].set(ylabel='input %d' % k)
        axs[k].set(ylabel=env.ACTION_LABELS[k] + '\n[%s]' % env.ACTION_UNITS[k])
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs[0].set_title('Input Trajectories')
    axs[-1].set(xlabel='time (sec)')

    # Compute RMSE for each state
    state_error = state_stack.transpose()[:, 0:plot_length].transpose() -\
                  goal_stack.transpose()[:, 0:plot_length].transpose()

    # Check if state is an angle and wrap angle error to [-pi, pi]
    angle_state_index = [i for i, x in enumerate(env.STATE_UNITS) if x == "rad"]
    for k in angle_state_index:
        state_error[:, k] = wrap2pi_vec(state_error[:, k])

    state_rmse, state_rmse_scalar = compute_state_rmse(state_error)

    if plot_traj:
        plt.show()

    if save_plot:
        plt.savefig(plot_dir + "input_ite%d" % ite_counter)
        plt.close()

    if save_data:
        np.savetxt(data_dir + "test%d_times.csv" % ep_counter, times.transpose(), delimiter=',', fmt='%.8f')
        np.savetxt(data_dir + "test%d_states.csv" % ep_counter, state_stack.transpose()[:, 0:plot_length].transpose(), delimiter=',', fmt='%.8f')
        np.savetxt(data_dir + "test%d_states_des.csv" % ep_counter, goal_stack.transpose()[:, 0:plot_length].transpose(), delimiter=',', fmt='%.8f')
        np.savetxt(data_dir + "test%d_inputs.csv" % ep_counter, input_stack.transpose()[:, 0:plot_length].transpose(), delimiter=',', fmt='%.8f')
        np.savetxt(data_dir + "test%d_state_rmse.csv" % ep_counter, state_rmse, delimiter=',', fmt='%.8f')
        np.savetxt(data_dir + "test%d_state_rmse_scalar.csv" % ep_counter, np.array([state_rmse_scalar]), delimiter=',', fmt='%.8f')

    # Return analysis data
    analysis_data = {}
    analysis_data["state_rmse"] = state_rmse
    analysis_data["state_rmse_scalar"] = state_rmse_scalar

    return analysis_data


def wrap2pi_vec(angle_vec):
    for k in range(len(angle_vec)):
        angle_vec[k] = wrap2pi(angle_vec[k])
    return angle_vec


def wrap2pi(angle):
    while angle > np.pi:
        angle -= np.pi
    while angle <= -np.pi:
        angle += np.pi
    return angle


def compute_state_rmse(state_error):
    # Compute root-mean-square error
    mse = np.mean(state_error ** 2, axis=0)
    state_rmse = np.sqrt(mse)
    state_rmse_scalar = np.sqrt(np.sum(mse))
    # Print root-mean-square error
    print(colored("rmse by state: " + get_arr_str(state_rmse), "blue"))
    print(colored("scalarized rmse: %.2f" % state_rmse_scalar, "blue"))
    return state_rmse, state_rmse_scalar

