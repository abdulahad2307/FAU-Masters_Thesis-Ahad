#!/bin/bash -l

#SBATCH --job-name=dil_llmv3_reg_evm_training          # Job name
#SBATCH --output=llmv3_dil_logs_final/%x_%j.out    # Standard output log
#SBATCH --error=llmv3_dil_logs_final/%x_%j.err     # Error log
#SBATCH --partition=v100                 # GPU partition name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=4                # CPU cores
#SBATCH --gres=gpu:v100:1                # Number of GPUs
#SBATCH --time=23:55:00                  # Shorter time for test run
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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# Domain Incremental Learning Configuration

DATA_DIR="/home/woody/iwi5/iwi5280h/dataset"
CHECKPOINT_DIR="/home/woody/iwi5/iwi5280h/llmv3_dil_reg_evm_training_final"
BASE_MODEL_PATH="/home/woody/iwi5/iwi5280h/FixedImage_llmv3/all_class_125k__LLMV3_AdamW_tesseract_20251111_234459/layoutlmv3_rvl_cdip_best.pt"

# Domain and class config
#DOMAINS="all_prepdataset,Tobacco3482-jpg"
GLOBAL_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,scientific_publication,specification,file_folder,news_article,budget,invoice,presentation,questionnaire,resume,memo,Note,Report"

# OCR tensor directories
OCR_TENSOR_PATH_BASE="/home/woody/iwi5/iwi5280h/dataset/all_prepdata_combined_ocr_texts_rectabbox_tesseract_single_image"
OCR_TENSOR_PATH_INC="/home/woody/iwi5/iwi5280h/dataset/tobacco_combined_ocr_texts_rectabbox_tesseract_single_image"
mkdir -p $CHECKPOINT_DIR

#IMAGES_PER_CLASS=12500

python src/llmv3_domain_incremental_reg_evm_training.py \
  --data_dir /home/woody/iwi5/iwi5280h/dataset \
  --ocr_tensor_path_base "$OCR_TENSOR_PATH_BASE" \
  --ocr_tensor_path_inc "$OCR_TENSOR_PATH_INC" \
  --all_classes "$GLOBAL_CLASSES" \
  --dataset_base rvl_cdip \
  --dataset_inc tobacco3482 \
  --base_model_path "$BASE_MODEL_PATH" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --batch_size 32 \
  --lr 1e-3 \
  --num_epochs 30 \
  --use_ewc \
  --lambda_ewc 5000.0 \
  --max_exemplars 16 \
  --exemplar_selection "herding" \
  --training_mode "last_layer" \
  --full_model_acc 0.9372 \
  #--images_per_class $IMAGES_PER_CLASS

# For resume functionality you can add:
#   --resume \
#   --resume_checkpoint "$RESUME_CKPT" \

echo "LayoutLMv3 Domain Incremental Learning Completed."


# To run:
# sbatch run_llmv3_domainIL_reg_evm.sh

#--data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \

# 1. --base_model_path /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250511_160135/eaml_best_model.pt\
# 2. --base_model_path /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250601_175404/eaml_best_model.pt\