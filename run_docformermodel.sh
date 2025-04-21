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

echo "Starting DocFormer Training..."

# Training command with class specification
export CUDA_LAUNCH_BLOCKING=1
python src/sota_docformer_model.py \
    --data_dir /home/woody/iwi5/iwi5280h/dataset/small_dataset \
    --output_dir outputs/funsd \
    --batch_size 8 \
    --num_epochs 100 \
    --learning_rate 1e-3 \
    --classes "letter,form,email,handwritten,advertisement,scientific report,invoice,resume"

echo "DocFormer Training Completed."

# sbatch run_docformermodel.sh
#--data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \