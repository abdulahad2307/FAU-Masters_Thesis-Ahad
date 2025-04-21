#!/bin/bash -l

#SBATCH --job-name=eaml_training          # Job name
#SBATCH --output=logs/%x_%j.out           # Standard output log
#SBATCH --error=logs/%x_%j.err            # Error log
#SBATCH --partition=rtx3080                   # GPU partition name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=1                 # Number of CPU cores per task
#SBATCH --gres=gpu:rtx3080:1                 # Number of GPUs
#SBATCH --time=23:30:00                   # Time limit hrs:min:sec
#SBATCH --export=NONE                     # Avoid inheriting unwanted environment variables

unset SLURM_EXPORT_ENV

# Load required modules
module load cuda/12.6
module load python/3.12-conda
conda activate mtil

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

export PYTHONPATH=$PYTHONPATH:$(pwd)/FAU-Masters_Thesis-Ahad

echo "Starting EAML Training..."

# Define classes as space-separated list like DocFormer
CLASSES="letter form email handwritten advertisement scientific_report invoice presentation questionnaire resume memo"

python src/sota_eaml_model.py \
    --data_dir /home/woody/iwi5/iwi5280h/dataset/small_dataset \
    --num_epochs 100 \
    --batch_size 8 \
    --learning_rate 1e-3 \
    --classes $CLASSES \
    --device cuda

echo "Training Completed. Output: $SLURM_OUTPUT"

#sbatch run_eamlmodel.sh
#--data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \