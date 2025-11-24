#!/bin/bash -l

#SBATCH --job-name=llmv3_CIL_reg_evm_training       # Job name
#SBATCH --output=llmv3_cil_logs_final/%x_3_%j.out    # Standard output log
#SBATCH --error=llmv3_cil_logs_final/%x_3_%j.err     # Error log
#SBATCH --partition=v100                 # GPU partition name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=2                # CPU cores
#SBATCH --gres=gpu:v100:1                # Number of GPUs
#SBATCH --time=23:00:00                  # Shorter time for test run
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

OCR_TENSOR_PATH="/home/woody/iwi5/iwi5280h/dataset/all_prepdata_combined_ocr_texts_rectabbox_tesseract_single_image"

#CKPT_DIR="/home/woody/iwi5/iwi5280h/cil_models/llmv3_cil_with_bsamp_with_bcor$(date +%Y%m%d_%H%M%S)"
CKPT_DIR="/home/woody/iwi5/iwi5280h/llmv3_cil/llmv3_cil_models_reg_evm_final" #eaml_cil_with_bsamp_with_bcor_20250903_072048/"

ALL_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,scientific_publication,specification,file_folder,news_article,budget,invoice,presentation,questionnaire,resume,memo"

#1
#BASE_MODEL="/home/woody/iwi5/iwi5280h/FixedImage_llmv3/11_class_125k_LLMV3_AdamW_tesseract_20251109_153906/layoutlmv3_rvl_cdip_best.pt"
#BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo"
#UNSEEN_CLASSES="scientific_publication"

#2
BASE_MODEL="/home/woody/iwi5/iwi5280h/llmv3_cil/llmv3_cil_models_reg_evm_final/layoutlmv3_cil_incremental_reg_evm_scientific_publication_best.pt"
BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication"
UNSEEN_CLASSES="specification"

#3
BASE_MODEL="/home/woody/iwi5/iwi5280h/llmv3_cil/llmv3_cil_models_reg_evm_final/layoutlmv3_cil_incremental_reg_evm_specification_best.pt"
BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication,specification"
UNSEEN_CLASSES="file_folder"

#BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication,specification,file_folder"

# Incremental Training on
#UNSEEN_CLASSES="file_folder" #"scientific_publication,specification,file_folder,news_article,budget"


mkdir -p $CKPT_DIR

echo "Starting LayoutLMv3 Class Incremental Learning RegEVM..."

python src/llmv3_class_incremental_reg_evm_training.py \
  --data_dir /home/woody/iwi5/iwi5280h/dataset/all_prepdataset \
  --ocr_tensor_path "$OCR_TENSOR_PATH" \
  --all_classes "$ALL_CLASSES" \
  --base_classes "$BASE_CLASSES" \
  --unseen_classes "$UNSEEN_CLASSES" \
  --base_model_path "$BASE_MODEL" \
  --checkpoint_dir "$CKPT_DIR" \
  --batch_size 64 \
  --lr 1e-3 \
  --num_epochs 100 \
  --use_ewc \
  --lambda_ewc 5000.0 \
  --max_exemplars 320 \
  --exemplar_selection "herding" \
  --training_mode "last_layer" \
  --full_model_acc 0.6441 \
  #--weight_decay 0.01 \
  #--patience 10 \
  #--max_length 128 \
  # --bbox_style "rect"\

# For resume functionality you can add:
#   --resume \
#   --resume_checkpoint "$RESUME_CKPT" \
#   --global_best_acc <previous_global_best_acc>

echo "LayoutLMv3 Class Incremental Learning RegEVM Completed."


# To run:
# sbatch run_llmv3_classIL_reg_evm_training.sh

#--data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \

# 1. --base_model_path /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250511_160135/eaml_best_model.pt\
# 2. --base_model_path /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250601_175404/eaml_best_model.pt\