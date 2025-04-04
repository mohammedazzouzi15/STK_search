#!/bin/bash
#SBATCH --job-name=run_benchmark
#SBATCH --output=run_benchmark_%A_%a.log
#SBATCH --error=run_benchmark_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --array=0-20  # Adjust the range according to the number of jobs you want to run

# Load necessary modules
conda activate /home/mazzouzi/miniconda3/envs/stk_search

# Run the Python script with the array task ID as an argument
python /media/mohammed/Work/STK_search/Example_notebooks/run_benchmark_slurm.py -s $SLURM_ARRAY_TASK_ID