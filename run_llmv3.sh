#!/bin/bash -l

#SBATCH --job-name=FixedImage_12.5K_llmv3_all_class_training_tesseract_adamW          # Job name
#SBATCH --output=Fixed_llmv3_logs/%x-4_%j.out           # Standard output log
#SBATCH --error=Fixed_llmv3_logs/%x-4_%j.err            # Error log
#SBATCH --partition=v100                  # GPU partition name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task
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

CUDA_LAUNCH_BLOCKING=1

echo "Starting LLMV3 Training..."

# Define classes
CLASS_MAPPING_PATH="/home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/class_mapping.json"
ALL_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,scientific_publication,specification,file_folder,news_article,budget,invoice,presentation,questionnaire,resume,memo"

CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo"

# Create output directory
OUTPUT_DIR="/home/woody/iwi5/iwi5280h/FixedImage_llmv3/all_class_125k__LLMV3_AdamW_tesseract_$(date +%Y%m%d_%H%M%S)"  

## 11_class
#OUTPUT_DIR="/home/woody/iwi5/iwi5280h/emal_models/outputs/outputs/11_eaml_adamW_tesseract_20250905_004113"

OCR_DATA_PATH="/home/woody/iwi5/iwi5280h/dataset/all_predataset_combined_ocr_texts_rectbbox_tesseract.pt" #all_dataset_ocr_texts_trocr.pt

mkdir -p $OUTPUT_DIR
mkdir -p logs

# Path to checkpoint
#CHECKPOINT_PATH="/home/woody/iwi5/iwi5280h/emal_models/outputs/outputs/all_eaml_adamW_tesseract_20250914_221921/eaml_checkpoint_ep9.pt"  # Set path if resuming

#11_class
#CHECKPOINT_PATH="/home/woody/iwi5/iwi5280h/emal_models/outputs/outputs/11_eaml_adamW_tesseract_20250905_004113/eaml_checkpoint_ep25.pt"  # Set path if resuming

# Resume from checkpoint
RESUME_ARG=""
if [ -n "$CHECKPOINT_PATH" ] && [ -f "$CHECKPOINT_PATH" ]; then
    RESUME_ARG="--resume $CHECKPOINT_PATH"
    echo "Resuming training from checkpoint: $CHECKPOINT_PATH"
fi

DATASET=${1:-"rvl_cdip"}
OCR_TENSOR_DIR=${2:-"/home/woody/iwi5/iwi5280h/dataset/all_predataset_combined_ocr_texts_rectbbox_tesseract.pt"}
IMAGE_DIR=${3:-"/home/woody/iwi5/iwi5280h/dataset/all_prepdataset"}
BBOX_STYLE=${4:-"rect"}    # or "rect", "poly"; must match your OCR tensor style

python src/sota_llmv3_model.py \
    --dataset $DATASET \
    --ocr_tensor_file $OCR_TENSOR_DIR \
    --image_dir $IMAGE_DIR \
    --batch_size 8 \
    --epochs 50 \
    --lr 2e-5 \
    --save_dir $OUTPUT_DIR \
    --max_length 512 \
    --device cuda \
    --bbox_style $BBOX_STYLE \
    --patience 10 \
    --images_per_class 12500 \
    --seed 42 \
    --resume  /home/woody/iwi5/iwi5280h/FixedImage_llmv3/all_class_125k__LLMV3_AdamW_tesseract_20251110_161727/layoutlmv3_rvl_cdip_best.pt \
    --resume_epoch 30

echo "Training Completed. Output: $OUTPUT_DIR"


#sbatch run_llmv3.sh
#--data_dir /home/woody/iwi5/iwi5280h/dataset/all_prepdataset \
#--data_dir /home/woody/iwi5/iwi5280h/dataset/small_dataset \

# json_small_trocr -- "/home/woody/iwi5/iwi5280h/dataset/small_dataset_ocr_texts_trocr.json"
# json_all_trocr -- "/home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_trocr.json"

# /home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_tesseract.pt