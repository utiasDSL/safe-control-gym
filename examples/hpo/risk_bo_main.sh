#!/bin/bash

cd ~/safe-control-gym

localOrHost=$1
algo=$2
run=$3

# localOrHost bo_algo algo gamma n_budget repeat_eval seed n_initial min_repeat_eval metric run

# for rl
bash examples/hpo/risk_bo.sh ${localOrHost} gpucb ${algo} 1 70 6 24 10 2 average_rmse ${run}
bash examples/hpo/risk_bo.sh ${localOrHost} rahbo ${algo} 1 70 6 24 10 2 average_rmse ${run}
bash examples/hpo/risk_bo.sh ${localOrHost} erahbo ${algo} 1 70 6 24 10 2 average_rmse ${run}

# for gpmpc
bash examples/hpo/risk_bo.sh ${localOrHost} gpucb gp_mpc 1 50 6 24 10 2 average_rmse ${run}
bash examples/hpo/risk_bo.sh ${localOrHost} rahbo gp_mpc 1 50 6 24 10 2 average_rmse ${run}
bash examples/hpo/risk_bo.sh ${localOrHost} erahbo gp_mpc 1 50 6 24 10 2 average_rmse ${run}





