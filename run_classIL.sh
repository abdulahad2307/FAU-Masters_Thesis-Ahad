#!/bin/bash -l

#SBATCH --job-name=cil_test_run          # Job name
#SBATCH --output=logs/cil_test_%j.out    # Standard output log
#SBATCH --error=logs/cil_test_%j.err     # Error log
#SBATCH --partition=v100                 # GPU partition name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=4                # CPU cores
#SBATCH --gres=gpu:v100:1                # Number of GPUs
#SBATCH --time=23:00:00                  # Shorter time for test run
#SBATCH --export=NONE                    # Avoid inheriting unwanted environment variables

unset SLURM_EXPORT_ENV

# Load required modules
module load cuda/12.6
module load python/3.12-conda
conda activate mtil

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# Move to the repository folder
export PYTHONPATH=$PYTHONPATH:$(pwd)/FAU-Masters_Thesis-Ahad

echo "Starting CIL Test Run..."

# Test with just 3 classes in incremental steps
export CUDA_LAUNCH_BLOCKING=1
python src/class_incremental.py \
    --data_dir /home/woody/iwi5/iwi5280h/dataset/small_dataset \
    --checkpoint_dir checkpoints/cil_test \
    --model_name "docformer" \          # or "eaml" for testing EAML
    --class_order "news article,specification,memo" \ # Only 3 classes for testing
    --start_step 0 \
    --batch_size 8 \
    --lr 2e-5 \
    --num_epochs 5 \
    --base_model_path /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/funsd/docformer_best_model.pt  # Path to pretrained model

echo "CIL Test Run Completed."

# To run:
# sbatch cil_test_run.sh

#--data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \