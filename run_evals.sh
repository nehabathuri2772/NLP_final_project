#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -J NLP_PTSD_FineTune%j
#SBATCH -o console%j.out
#SBATCH -e console%j.err
#SBATCH -p academic
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00

set -e

echo "Starting job $SLURM_JOB_ID on $(hostname)"

# Load modules
module purge
module load cuda/12.8.0
module load python/3.12.10

# Install uv
if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "uv version: $(uv --version)"

# Load venv
uv sync
source .venv/bin/activate

# Run training
python -u -m main