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

cd /storage/ice-shared/cs7643/shared-group-project-data/phoenix/omscs-dl-final-project

conda activate phoenix

python train.py --opt experiments/train_smoke.yml
python test.py  --opt experiments/test_smoke.yml
