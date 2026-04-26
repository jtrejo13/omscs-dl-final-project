#!/bin/bash
#SBATCH -N 1     
#SBATCH -c 64
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:15:00                  
#SBATCH --gres=gpu:V100:1          
#SBATCH --mem-per-gpu=32G
#SBATCH -J nafnet_smoke    # jobs name
#SBATCH -o ./logs/smoke_%j.out   # file to write logs, prints, etc

mkdir -p logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

conda activate phoenix 2>/dev/null || true

python train.py --opt experiments/train_smoke.yml
python test.py  --opt experiments/test_smoke.yml
