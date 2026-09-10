#!/bin/bash
#SBATCH --job-name=TransformerFlow
#SBATCH --output=torch_gpu_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00


source "$HOME/miniforge3/bin/activate"
conda activate harmosc-hpc
cd "$SLURM_SUBMIT_DIR"
nvidia-smi

srun python main.py