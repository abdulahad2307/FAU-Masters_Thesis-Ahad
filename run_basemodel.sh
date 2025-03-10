#!/bin/bash -l

#SBATCH --job-name=baseline_training      # Job name
#SBATCH --output=logs/%x_%j.out           # Standard output log
#SBATCH --error=logs/%x_%j.err            # Error log
#SBATCH --partition=v100                  # GPU partition name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=1                 # Number of CPU cores per task
#SBATCH --gres=gpu:v100:1                 # Number of GPUs
#SBATCH --time=10:00:00                    # Time limit hrs:min:sec
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

echo "Starting Baseline Model Training..."

# Run Model Training
echo "Training ResNet50..."
export CUDA_LAUNCH_BLOCKING=1
python3 src/baseline_model.py \
    --data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \
    --model_name resnet50 \
    --num_classes 16 \
    --batch_size 64 \
    --epochs 100 \
    --learning_rate 0.001 \
    --optimizer adamw \
    --device cuda

echo "Training DenseNet121..."
python3 src/baseline_model.py \
    --data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \
    --model_name densenet121 \
    --num_classes 16 \
    --batch_size 64 \
    --epochs 100 \
    --learning_rate 0.001 \
    --optimizer adamw \
    --device cuda

echo "Training Completed. Check logs/training_log.csv for results."

#sbatch.tinygpu run_basemodel.sh