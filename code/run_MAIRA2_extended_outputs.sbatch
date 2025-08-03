#!/bin/bash
#SBATCH --job-name=maira2
#SBATCH --output=slurm_output/MAIRA2-%A.out    
##SBATCH --error=slurm_output/MAIRA2-%A.err
##SBATCH --partition=universe,asteroids 
#SBATCH --qos=master-queuesave
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
# Load python module ml python/anaconda3
# Activate corresponding environment

source /meta/opt/anaconda3/etc/profile.d/conda.sh #ADJUST AS NECESSARY
conda activate /vol/miltank/users/vac/VLM/maira2_env #ADJUST AS NECESSARY

exec 2>&1
python3 -u MAIRA2_local.py --chest_ext_unproc
python3 -u MAIRA2_local.py --chest_ext_proc_bars
python3 -u MAIRA2_local.py --chest_ext_proc_no_bars
python3 -u MAIRA2_local.py --brain_ext_simple
python3 -u MAIRA2_local.py --brain_ext_specific