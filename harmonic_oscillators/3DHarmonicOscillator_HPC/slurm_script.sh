#!/bin/bash
#SBATCH --job-name=3DHarmonicOscillator
#SBATCH --output=torch_gpu_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00


source ~/miniforge3/etc/profile.d/conda.sh
conda activate harmosc-hpc

nvidia-smi

srun python HarmOsc_HPC.py