#!/bin/bash -l

#SBATCH --job-name=eaml_all_training_trocr          # Job name
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
OUTPUT_DIR="outputs/All_eaml_trocr_20250624_231314" #"outputs/All_eaml_trocr_$(date +%Y%m%d_%H%M%S)"
OCR_JSON_PATH="/home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_trocr.json"
mkdir -p $OUTPUT_DIR
mkdir -p logs

# Path to checkpoint
CHECKPOINT_PATH="/home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/All_eaml_trocr_20250624_231314/eaml_checkpoint_ep19.pt"

# Check if resuming from a checkpoint
if [ -n "$19" ] && [ -f "$CHECKPOINT_PATH" ]; then
    RESUME_ARG="--resume $CHECKPOINT_PATH"
    echo "Resuming training from checkpoint: $CHECKPOINT_PATH"
else
    RESUME_ARG=""
fi

python src/sota_eaml_model.py \
  --data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \
  --ocr_json_path $OCR_JSON_PATH \
  --output_dir $OUTPUT_DIR \
  --num_epochs 100 \
  --batch_size 16 \
  --learning_rate 5e-5 \
  --weight_decay 0.05 \
  --classes $CLASSES \
  --device cuda \
  --patience 13 \
  --keep_checkpoints 2 \
  --cls_weight 1.0 \
  --kld_weight 0.5 \
  --kld_threshold 0.1 \
  --embed_dim 512 \
  --dropout_rate 0.5 \
  $RESUME_ARG

echo "Training Completed. Output: $OUTPUT_DIR"

#sbatch run_eamlmodel.sh
#--data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \
#--data_dir /home/woody/iwi5/iwi5280h/dataset/small_dataset \

# json_small_trocr -- "/home/woody/iwi5/iwi5280h/dataset/small_dataset_ocr_texts_trocr.json"
# json_all_trocr -- "/home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_trocr.json"