#!/bin/bash -l

#SBATCH --job-name=eaml_training          # Job name
#SBATCH --output=logs/%x_%j.out           # Standard output log
#SBATCH --error=logs/%x_%j.err            # Error log
#SBATCH --partition=v100                  # GPU partition name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=1                 # Number of CPU cores per task
#SBATCH --gres=gpu:v100:1                 # Number of GPUs
#SBATCH --time=23:55:00                   # Time limit hrs:min:sec
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

# Create output directory
OUTPUT_DIR="outputs/eaml_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR
mkdir -p logs

# Check if resuming from a checkpoint
if [ -n "$1" ] && [ -f "$1" ]; then
    RESUME_ARG="--resume $1"
    echo "Resuming training from checkpoint: $1"
else
    RESUME_ARG=""
fi

python src/sota_eaml_model.py \
  --data_dir /home/woody/iwi5/iwi5280h/dataset/small_dataset \
  --output_dir $OUTPUT_DIR \
  --num_epochs 100 \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --weight_decay 0.005 \
  --classes $CLASSES \
  --device cuda \
  --patience 10 \
  --keep_checkpoints 2 \
  --cls_weight 1.0 \
  --kld_weight 0.3 \
  --kld_threshold 0.1 \
  --embed_dim 512 \
  --dropout_rate 0.2 \
  --ocr_engine trocr \
  --ocr_model microsoft/trocr-base-handwritten \
  $RESUME_ARG

echo "Training Completed. Output: $SLURM_OUTPUT"

#sbatch run_eamlmodel.sh
#--data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \