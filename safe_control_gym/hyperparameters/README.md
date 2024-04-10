# Instruction for running hyperparameter optimization

## Install on Ubuntu

### Create a `conda` environment

Create the environment by following the steps presented in the main
[`README.md`](README.md)


To perform hyperparmeter optimization, you may need `MySQL` database:

```bash
sudo apt-get install mysql-server # MySQL server starts automatically after the installation
```

To set up, run the following commands sequencially:

```bash
sudo mysql
CREATE USER optuna@"%"; # this creates the user name 'optuna' accessible by any host
CREATE DATABASE {algo}_hpo; 
GRANT ALL ON {algo}_hpo.* TO optuna@"%";
exit
```

Addtionally, install the following packages:

```bash
pip install mysql-connector-python
pip install pymysql
pip install optuna
pip install optuna-dashboard
```


 You may replace `{algo}` with `gp_mpc`, `ppo`, `sac`, or `ddpg` in order to run the scripts.

### Toy Examples

The results for toy examples in the paper can be reproduced in [`toy_example.ipynb`](experiments/comparisons/rl/toy_example.ipynb)


### Reinforcement Learning

To run hyperparameter optimization (HPO) for DDPG, run:

```bash
bash experiments/comparisons/rl/main.sh hostx TPESampler ddpg cartpole stab False
```

To run hyperparameter optimization (HPO) for PPO, run:

```bash
bash experiments/comparisons/rl/main.sh hostx TPESampler ppo cartpole stab False
```

To run hyperparameter optimization (HPO) for SAC, run:

```bash
bash experiments/comparisons/rl/main.sh hostx TPESampler sac cartpole stab False
```

### Learning-Based Control

To run hyperparameter optimization (HPO) for GP-MPC, run:

```bash
bash experiments/comparisons/gpmpc/main.sh hostx TPESampler cartpole stab False
```

#### Note
You may need to adjust the `path` of `conda.sh` in the sub-scripts called by `main.sh` such as `rl_hpo_strategy_eval.sh`.