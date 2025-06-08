#!/bin/bash -l

#SBATCH --job-name=alldocformer_small_training     # Job name
#SBATCH --output=logs/alldocformer_%j.out    # Standard output log
#SBATCH --error=logs/alldocformer_%j.err     # Error log
#SBATCH --partition=v100                  # GPU partition name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=4                 # Increased CPU cores for OCR processing
#SBATCH --gres=gpu:v100:1                 # Number of GPUs
#SBATCH --time=23:30:00                   # Time limit hrs:min:sec
#SBATCH --export=NONE                     # Avoid inheriting unwanted environment variables

unset SLURM_EXPORT_ENV

# Load required modules
# Load required modules
module load cuda/12.6
module load python/3.12-conda
conda activate mtil

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

echo "Starting DocFormer Training..."

# Define classes as comma-separated list matching paper implementation
CLASSES="advertisement,budget,email,file_folder,form,handwritten,invoice,letter,memo,news_article,presentation,questionnaire,resume,scientific_publication,scientific_report,specification"

# Create time-stamped output directory
OUTPUT_DIR="docformer_outputs/$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR
mkdir -p logs

# Training parameters (matches paper settings)
PHASE="finetune"  # or "pretrain" for pre-training phase
OCR_ENGINE="trocr"  # [tesseract|trocr|pero]
BATCH_SIZE=8
NUM_EPOCHS=50
LEARNING_RATE=2.5e-5
MAX_SEQ_LENGTH=512

# Check for resuming training
if [ -n "$1" ] && [ -f "$1" ]; then
    RESUME_ARG="--resume $1"
    echo "Resuming training from checkpoint: $1"
else
    RESUME_ARG=""
fi

# Start training with paper-compliant parameters
python src/main.py \
  --data_dir /home/woody/iwi5/iwi5280h/dataset/small_dataset \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --num_epochs $NUM_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --max_seq_length $MAX_SEQ_LENGTH \
  --classes "$CLASSES" \
  --ocr_engine $OCR_ENGINE \
  --phase $PHASE \
  $RESUME_ARG

echo "Training Completed. Output: $OUTPUT_DIR"

# sbatch run_docformermodel.sh
#--data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \