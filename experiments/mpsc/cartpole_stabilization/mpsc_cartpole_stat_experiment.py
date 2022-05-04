"""This script tests the MPSC safety filter implementation

"""
import os
import time
import shutil
import pickle
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from safe_control_gym.utils.registration import make
from safe_control_gym.utils.configuration import ConfigFactory


def main():
    fac = ConfigFactory()
    config = fac.merge()
    env_func = partial(make,
                       config.task,
                       **config.task_config)

    # Setup controller.
    ctrl = make(config.algo,
                    env_func,
                    **config.algo_config)
    
    # Load state_dict from trained.
    model_dir = os.path.dirname(os.path.abspath(__file__))+'/models'
    ctrl.load(os.path.join(model_dir,f'{config.algo}_model_cartpole.pt'))  # Show violation.
    
    # Remove temporary files and directories
    shutil.rmtree(os.path.dirname(os.path.abspath(__file__))+'/temp', ignore_errors=True)

    # Setup MPSC.
    mpsc = make(config.safety_filter,
                env_func,
                **config.sf_config)
    mpsc.reset()
    
    train_env = env_func(init_state=None, disturbances=None) # training without disturbances
    mpsc.learn(env=train_env)

    state_constraints_sym = mpsc.state_constraints_sym
    input_constraints_sym = mpsc.input_constraints_sym
    
    iterations = 150
    limits = np.array([[10, 0.5, 0.05, 0.5]])

    all_results = {}
    all_results['uncert'] = {'init_state': [], 'final_state': [], 'iters': [], 'violations': [], 'success': [], 'time': []}
    all_results['cert'] = {'init_state': [], 'final_state': [], 'iters': [], 'violations': [], 'success': [], 'time': [], 'outside_ss': [], 'outside_max_ss': [], 'corrections': []}
    
    with open('./temp-data/init_points_freq_15.pkl', "rb") as f:
        init_points = pickle.load(f)
    
    init_points = init_points['uncert']['init_state']

    for iter in range(len(init_points)):
        init_state = init_points[iter]
        env = env_func(init_state=init_state)

        # Run without safety filter
        START = time.time()
        ctrl.safety_filter = None
        _, results = ctrl.run(env=env, num_iterations=iterations)
        elapsed_time_uncert = time.time() - START

        num_violations = sum([results.info[i][0]['constraint_violation'] for i in range(len(results.info))])

        ctrl.reset()

        # Run with safety filter
        START = time.time()
        ctrl.safety_filter = mpsc
        _, certified_results = ctrl.run(env=env, num_iterations=iterations)
        elapsed_time_cert = time.time() - START

        # Reset
        ctrl.reset()
        mpsc.reset()
        env.close()

        # Record results
        success = np.all(np.abs(certified_results.obs[-1, :]) < limits)
        all_results['uncert']['init_state'].append(init_state)
        all_results['uncert']['time'].append(elapsed_time_uncert)
        all_results['uncert']['iters'].append(results.obs.shape[0])
        all_results['uncert']['success'].append(success)
        all_results['uncert']['violations'].append(num_violations)
        all_results['uncert']['final_state'].append(results.obs[-1, :])

        success = np.all(np.abs(certified_results.obs[-1, :]) < limits)
        num_violations = sum([certified_results.info[i][0]['constraint_violation'] for i in range(len(certified_results.info))])
        all_results['cert']['init_state'].append(init_state)
        all_results['cert']['time'].append(elapsed_time_cert)
        all_results['cert']['iters'].append(certified_results.obs.shape[0])
        all_results['cert']['success'].append(success)
        all_results['cert']['violations'].append(num_violations)
        all_results['cert']['corrections'].append(np.sum(certified_results.corrections>1e-6))
        all_results['cert']['final_state'].append(certified_results.obs[-1, :])

        print("ITERATION: ", iter)

        fig, ax = plt.subplots()
        ax.plot(certified_results.obs[:,2], certified_results.obs[:,3],'.-', label='Certified')
        corrections = certified_results.corrections>1e-6
        ax.plot(certified_results.obs[corrections, 2], certified_results.obs[corrections, 3], 'r.', label='Modified')
        ax.plot(results.obs[:, 2], results.obs[:, 3], 'r--', label='Uncertified')
        ax.axvline(x=-0.2, color='k', lw=2, label='Limit')
        ax.scatter(results.obs[0, 2], results.obs[0, 3], color='g', marker='o', s=100, label='Initial State')
        ax.axvline(x=0.2, color='k', lw=2)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$\dot{\theta}$")
        ax.set_box_aspect(0.5)
        ax.legend(loc='upper left')

        plt.tight_layout()
        # plt.savefig('./temp-data/mpsc.png', dpi=500)
        plt.show()

    print("NUM SUCCESSES CERT:", sum(all_results['cert']['success']))
    print("NUM SUCCESSES UNCERT:", sum(all_results['uncert']['success']))
    valid = np.array(all_results['cert']['success'])
    print("NUM VIOLATIONS:", sum(all_results['cert']['violations']), sum(np.array(all_results['cert']['violations'])[valid]))

    with open('results_15Hz_dist1_learn.pkl', "wb") as f:
        pickle.dump(all_results, f)
        
    ctrl.close()
    mpsc.close()


if __name__ == "__main__":
    main()
