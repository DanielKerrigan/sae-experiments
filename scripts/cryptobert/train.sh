#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --job-name=cbert
#SBATCH --mem=10GB
#SBATCH --ntasks=1
#SBATCH --output=/home/kerrigan.d/work/mi/sae-experiments/output/cbert_train_%j.out
#SBATCH --error=/home/kerrigan.d/work/mi/sae-experiments/output/cbert_train_%j.err

srun pixi run python train.py --path /home/kerrigan.d/work/mi/sae-experiments
