#!/bin/bash
#SBATCH --job-name=TransformerFlow
#SBATCH --output=torch_width_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --array=0-16%4



source "$HOME/miniforge3/bin/activate"
conda activate harmosc-hpc || exit 1
module load cudatoolkit || exit 1

#Set these AFTER loading modules and activating Conda.
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CPATH="${CUDA_HOME:?CUDA_HOME is unset}/include${CPATH:+:${CPATH}}"
# Check the compiler selected for Triton.
"$CC" --version || exit 1
"$CC" -Wno-psabi -fsyntax-only -x c - <<'EOF' || exit 1
#include <cuda.h>

EOF

echo "Compiler and CUDA header check passed"

cd "$SLURM_SUBMIT_DIR"

nvidia-smi
 

WIDTHS=(
    4 8 12 16 20 24 28 32 36
    40 44 48 52 56 60 64 68
)

WIDTH=${WIDTHS[$SLURM_ARRAY_TASK_ID]}

echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Training width: $WIDTH"
set -x
srun --ntasks=1 --cpu-bind=none python -u main.py --width "$WIDTH"