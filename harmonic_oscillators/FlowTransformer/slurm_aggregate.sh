#!/bin/bash
#SBATCH --job-name=TransformerAggregate
#SBATCH --output=transformer_aggregate_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00


source "$HOME/miniforge3/bin/activate"
conda activate harmosc-hpc

cd "$SLURM_SUBMIT_DIR"

srun python -u main.py --aggregate