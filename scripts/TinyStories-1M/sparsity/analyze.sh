#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --job-name=ts-1m
#SBATCH --mem=10GB
#SBATCH --ntasks=1
#SBATCH --output=/scratch/kerrigan.d/mi/sae-experiments/output/TinyStories-1M_k_analyze_%A_%a.out
#SBATCH --error=/scratch/kerrigan.d/mi/sae-experiments/output/TinyStories-1M_k_analyze_%A_%a.err
#SBATCH --array=1,4,16,32

source /home/kerrigan.d/miniconda3/bin/activate
conda activate sae-experiments

srun python analyze.py -k $SLURM_ARRAY_TASK_ID --path /scratch/kerrigan.d/mi/sae-experiments
