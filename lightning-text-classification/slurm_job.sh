#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH -D .                    # Working Directory
#SBATCH -J ML     	# Job Name
#SBATCH --nodes=2
#SBATCH --gres=gpu:tesla:2	# request two GPUs
#SBATCH --ntasks-per-node=2
#SBATCH --time=0-00:30:00
#SBATCH --partition=gpu

#Job-Status per Mail:
#SBATCH --mail-type=NONE
#SBATCH --mail-user=tilo.himmelsbach@tu-berlin.de

source activate ml_gpu

python training.py     --gpus 2  --num_nodes 2    --batch_size 8     --loader_workers 4


