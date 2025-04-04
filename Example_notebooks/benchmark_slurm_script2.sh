#!/bin/bash
#SBATCH --job-name=run_benchmark
#SBATCH --output=slurm_log/run_benchmark_%A_%a.log
#SBATCH --error=slurm_log/run_benchmark_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-35  # Adjust the range according to the number of jobs you want to run
#SBATCH --exclude=node40,node63,node25,node19,node35,node64,node62

# Load necessary modules
conda activate /home/mazzouzi/miniconda3/envs/stk_search

# Run the Python script with the array task ID as an argument
python /media/mohammed/Work/STK_search/Example_notebooks/run_benchmark_slurm.py -s $SLURM_ARRAY_TASK_ID