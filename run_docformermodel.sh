#!/bin/bash -l

#SBATCH --job-name=docformer_training     # Job name
#SBATCH --output=logs/docformer_%j.out    # Standard output log
#SBATCH --error=logs/docformer_%j.err     # Error log
#SBATCH --partition=v100                  # GPU partition name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=4                 # Increased CPU cores for OCR processing
#SBATCH --gres=gpu:v100:1                 # Number of GPUs
#SBATCH --time=23:30:00                   # Time limit hrs:min:sec
#SBATCH --mem=16G                         # Memory allocation
#SBATCH --export=NONE                     # Avoid inheriting unwanted environment variables

unset SLURM_EXPORT_ENV

# Load required modules
module load cuda/12.6
module load python/3.12-conda
module load tesseract/5.3.3  # Required for OCR processing
conda activate mtil

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# Move to the repository folder
export PYTHONPATH=$PYTHONPATH:$(pwd)/FAU-Masters_Thesis-Ahad

# Set Tesseract data path if needed
export TESSDATA_PREFIX=/path/to/tessdata  # Update if required

echo "Starting DocFormer Training..."

# Training command
export CUDA_LAUNCH_BLOCKING=1
python src/run_docformer.py \
    --data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \
    --dataset funsd \
    --output_dir outputs/funsd \
    --batch_size 8 \
    --num_epochs 50 \
    --learning_rate 2.5e-5

echo "DocFormer Training Completed."

# Evaluation command (uncomment to run evaluation after training)
# echo "Starting DocFormer Evaluation..."
# python src/run_docformer.py \
#     --data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \
#     --dataset funsd \
#     --output_dir outputs/funsd \
#     --eval_only \
#     --resume outputs/funsd/best_model.pt
# echo "DocFormer Evaluation Completed."

# To submit: sbatch run_docformer.sh