#!/bin/bash -l

#SBATCH --job-name=eaml_all_training_tesseract          # Job name
#SBATCH --output=logs/%x_%j.out           # Standard output log
#SBATCH --error=logs/%x_%j.err            # Error log
#SBATCH --partition=v100                  # GPU partition name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=1                 # Number of CPU cores per task
#SBATCH --gres=gpu:v100:1                 # Number of GPUs
#SBATCH --time=23:59:00                   # Time limit hrs:min:sec
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

# Define classes
CLASS_MAPPING_PATH="/home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/class_mapping.json"
#CLASSES="letter form email handwritten advertisement scientific_report invoice presentation questionnaire resume memo"
#CLASSES="advertisement,budget,email,file_folder,form,handwritten,invoice,letter,memo,news_article,presentation,questionnaire,resume,scientific_publication,scientific_report,specification"
CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo"

# Create output directory
OUTPUT_DIR="outputs/all_eaml_SGD_tesseract_$(date +%Y%m%d_%H%M%S)"
OCR_DATA_PATH="/home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_tesseract.pt " #all_dataset_ocr_texts_trocr.pt
mkdir -p $OUTPUT_DIR
mkdir -p logs

# Path to checkpoint
CHECKPOINT_PATH=""  # Set path if resuming

# Resume from checkpoint
RESUME_ARG=""
if [ -n "$CHECKPOINT_PATH" ] && [ -f "$CHECKPOINT_PATH" ]; then
    RESUME_ARG="--resume $CHECKPOINT_PATH"
    echo "Resuming training from checkpoint: $CHECKPOINT_PATH"
fi

python src/sota_eaml_model.py \
  --data_dir /home/woody/iwi5/iwi5280h/dataset/all_prepdataset \
  --ocr_data_path $OCR_DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --num_epochs 50 \
  --batch_size 16 \
  --learning_rate 1e-3 \
  --weight_decay 0.01 \
  --class_mapping_path $CLASS_MAPPING_PATH \
  --classes $CLASSES \
  --device cuda \
  --patience 15 \
  --keep_checkpoints 2 \
  --cls_weight 1.0 \
  --kld_weight 0.5 \
  --kld_threshold 0.1 \
  --embed_dim 512 \
  --dropout_rate 0.5 \
  $RESUME_ARG

echo "Training Completed. Output: $OUTPUT_DIR"

#sbatch run_eamlmodel.sh
#--data_dir /home/woody/iwi5/iwi5280h/dataset/all_prepdataset \
#--data_dir /home/woody/iwi5/iwi5280h/dataset/small_dataset \

# json_small_trocr -- "/home/woody/iwi5/iwi5280h/dataset/small_dataset_ocr_texts_trocr.json"
# json_all_trocr -- "/home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_trocr.json"

# /home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_tesseract.pt