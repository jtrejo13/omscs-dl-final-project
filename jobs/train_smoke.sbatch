#!/bin/bash
#SBATCH --job-name=nafnet_smoke
#SBATCH --account=cs7643
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

mkdir -p logs

cd /storage/ice-shared/cs7643/shared-group-project-data/phoenix/omscs-dl-final-project

conda activate phoenix

python train.py --opt experiments/train_smoke.yml
python test.py  --opt experiments/test_smoke.yml
