#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --job-name=ts-1m
#SBATCH --mem=10GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/kerrigan.d/mi/sae-experiments/output/TinyStories-1M_single_train_%j.out
#SBATCH --error=/scratch/kerrigan.d/mi/sae-experiments/output/TinyStories-1M_single_train_%j.err

source /home/kerrigan.d/miniconda3/bin/activate
conda activate sae-experiments

srun python train.py --path /scratch/kerrigan.d/mi/sae-experiments
