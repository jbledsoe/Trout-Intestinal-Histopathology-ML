#!/bin/bash
#SBATCH --job-name=enteritis-r18
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm_enteritis_%j.out

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/enteritis_resnet18_public.yaml}"

cd "$PROJECT_ROOT"
mkdir -p logs

if [ -n "${VENV_PATH:-}" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

python scripts/train.py --config "$CONFIG_PATH"

