#!/bin/bash
#SBATCH --job-name=safe_control_gym_train_rl_mpsf_reset
#SBATCH --output=/h/pizarrob/safe-control-gym/experiments/mpsc/temp-data/mpsf_reset_%j.out
#SBATCH --error=/h/pizarrob/safe-control-gym/experiments/mpsc/temp-data/mpsf_reset_%j.err
#SBATCH --open-mode=append
#SBATCH --partition=cpu,gpu,wsgpu
#SBATCH --ntasks=1
#SBATCH -t 0-12:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --qos=normal

./train_rl_model.sh
./mpsc_experiment.sh
python plotting_results.py
