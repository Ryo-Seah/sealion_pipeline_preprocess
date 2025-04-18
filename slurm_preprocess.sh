#!/bin/bash
#SBATCH --job-name=torch-gpu-job        # Job name
#SBATCH --output=logs/output_%j.log     # Output log (%j = job ID)
#SBATCH --error=logs/error_%j.log       # Error log
#SBATCH --partition=gpu-long                # Request the GPU partition (name may vary)
#SBATCH --gpus=h100-96:1                    # Request 1 GPU
#SBATCH --cpus-per-task=4               # Number of CPU cores
#SBATCH --mem=64G                       # Memory
#SBATCH --time=1:00:00                 # Max wall time (hh:mm:ss)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ryo_seah@u.nus.edu

# Optional: Load modules (if you're using environment modules)
# module load python/3.9 cuda/11.8

# Optional: Activate your environment (conda or venv)
# source ~/miniconda3/bin/activate myenv

echo "Running on $(hostname)"
source ~/sealion_pipeline_preprocess/venv/bin/activate
nvidia-smi

# Run your script
python pipeline.py