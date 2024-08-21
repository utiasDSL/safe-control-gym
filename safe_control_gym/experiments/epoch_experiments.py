'''customized training/evaluation environment for the quadrotor task with GP-MPC controller'''

from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.envs.gym_control.cartpole import CartPole
from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor
from safe_control_gym.utils.gpmpc_plotting import make_plots, make_quad_plots

class EpochExperiment(BaseExperiment):
    '''The class for running experiments for model-based methods.'''
    def __init__(self, 
                 env,
                 ctrl,
                 train_env=None,
                 safety_filter=None,
                 verbose: bool = False
                 ):      
          
        super().__init__(env=env,
                         ctrl=ctrl,
                         train_env=train_env,
                         safety_filter=safety_filter,
                         verbose=verbose
                         )
    
    def launch_training(self, **kwargs):
        self.reset()
        train_runs, test_runs = self.ctrl.learn(env=self.train_env, **kwargs)

        if self.safety_filter:
            raise NotImplementedError('Safety filter not supported for GP-MPC')
        
        # plot training results if num_epoch > 1
        if self.ctrl.num_epochs > 1:
            if isinstance(self.env.env, Quadrotor):
                make_quad_plots(test_runs=test_runs, 
                                train_runs=train_runs, 
                                trajectory=self.ctrl.traj.T,
                                dir=self.ctrl.output_dir)
            elif isinstance(self.env.env, CartPole):
                make_plots(test_runs=test_runs, 
                           train_runs=train_runs, 
                            dir=self.ctrl.output_dir)
        print('Training done.')

        trajs_data = {}
        if self.train_env is not None:
            trajs_data = self.train_env.data

        return dict(trajs_data), train_runs, test_runs