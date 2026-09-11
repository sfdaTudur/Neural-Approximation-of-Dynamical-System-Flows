#!/bin/bash
#SBATCH --job-name=TransformerFlow
#SBATCH --output=torch_width_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --array=0-16%4


source "$HOME/miniforge3/bin/activate"
conda activate harmosc-hpc

cd "$SLURM_SUBMIT_DIR"

nvidia-smi


WIDTHS=(
    4 8 12 16 20 24 28 32 36
    40 44 48 52 56 60 64 68
)

WIDTH=${WIDTHS[$SLURM_ARRAY_TASK_ID]}

echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Training width: $WIDTH"

srun python -u main.py --width "$WIDTH"