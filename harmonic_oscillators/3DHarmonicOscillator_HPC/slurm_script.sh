#!/bin/bash

#SBATCH --job-name=harmosc3d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=harmosc_%j.out
#SBATCH --error=harmosc_%j.err

# Stop immediately if a command fails.
set -euo pipefail

# Run from the directory where sbatch was submitted.
cd "$SLURM_SUBMIT_DIR"

# Activate Miniforge and the project environment.
source "$HOME/miniforge3/bin/activate"
conda activate harmosc-hpc

# Print Python output immediately rather than buffering it.
export PYTHONUNBUFFERED=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"

# Display the allocated GPU.
nvidia-smi --list-gpus

# Check the Python environment and CUDA availability.
python --version
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Run the experiment.
srun python -u HarmOsc_HPC.py

echo "Finish time: $(date)"