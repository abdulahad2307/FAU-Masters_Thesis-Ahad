#!/bin/bash -l

#SBATCH --job-name=eaml_training          # Job name
#SBATCH --output=logs/%x_%j.out           # Standard output log
#SBATCH --error=logs/%x_%j.err            # Error log
#SBATCH --partition=v100                  # GPU partition name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=1                 # Number of CPU cores per task
#SBATCH --gres=gpu:v100:1                 # Number of GPUs
#SBATCH --time=23:30:00                   # Time limit hrs:min:sec
#SBATCH --export=NONE                     # Avoid inheriting unwanted environment variables

unset SLURM_EXPORT_ENV

# Load required modules
module load cuda/12.6
module load python/3.12-conda
conda activate mtil

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# Move to the repository folder
export PYTHONPATH=$PYTHONPATH:$(pwd)/FAU-Masters_Thesis-Ahad

echo "Starting EAML Model Training..."

# Run Model Training
export CUDA_LAUNCH_BLOCKING=1
python3 src/sota_eaml_model.py \
    --data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \
    --epochs 5 \
    --batch_size 64 \
    --lr 0.001 \
    --device cuda \
    --classes "letter" "form" "email" "handwritten" "advertisement" "scientific report" "invoice" "presentation" "resume" "memo"

echo "EAML Training Completed. Check logs/training_log.csv for results."

#sbatch run_eamlmodel.sh