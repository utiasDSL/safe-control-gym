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


You may replace `{algo}` with `ppo` or `sac` in order to run the scripts.

### Usages

#### Perform hyperparameter optimization

Execute the following command for optimizing hyperparameters

```bash
python ./experiments/comparisons/rl/rl_experiment.py \
                    '--algo', {algo}, \
                    '--task', {sys}, \
                    '--overrides', \
                    ./experiments/comparisons/rl/config_overrides/{sys}/{sys}_{task}.yaml, \
                    ./experiments/comparisons/rl/{algo}/config_overrides/{sys}/{algo}_{sys}_.yaml, \
                    ./experiments/comparisons/rl/{algo}/config_overrides/{sys}/{algo}_{sys}_hpo_.yaml, \
                    --output_dir, {output_dir}, \
                    --seed, 7, \
                    --use_gpu, True
```

You may replace `{sys}` with `carpole` in order to run the script.

#### Train policy using optimized hyperparameters

Execute the following command if optimized hyperparameters are given

```bash
python ./experiments/comparisons/rl/rl_experiment.py \
                    --algo {algo} \
                    --overrides \
                    ./experiments/comparisons/rl/{algo}/config_overrides/cartpole/{algo}_{sys}_.yaml \
                    ./experiments/comparisons/rl/config_overrides/{sys}/{sys}_{task}.yaml \
                    --output_dir {output_dir} \
                    --tag {run_name} \
                    --opt_hps {best_hp_file} \
                    --task {sys} --seed 2 --use_gpu True
```

You may replace `{sys}` with `carpole` in order to run the script. As an example, `{best_hp_file}` can be replaced with `experiments/comparisons/rl/sac/config_overrides/cartpole/hyperparameters_134.4275.yaml` for `sac`.