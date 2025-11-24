#!/bin/bash -l

#SBATCH --job-name=eaml_cil_evm_with_bsamp_with_bcor          # Job name
#SBATCH --output=logs/cil_eaml_evm_5-2_%j.out    # Standard output log
#SBATCH --error=logs/cil_eaml_evm_5-2_%j.err     # Error log
#SBATCH --partition=v100                 # GPU partition name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=2                # CPU cores
#SBATCH --gres=gpu:v100:1                # Number of GPUs
#SBATCH --time=23:58:00                  # Shorter time for test run
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

DATA_DIR="/home/woody/iwi5/iwi5280h/dataset/all_prepdataset"
OCR_TENSOR_PATH="/home/woody/iwi5/iwi5280h/dataset/all_predataset_combined_ocr_texts_rectbbox_tesseract.pt"
#BASE_MODEL="/home/woody/iwi5/iwi5280h/cil_models/all_eaml_SGD_tesseract_20250805_223607/eaml_best_model.pt"
#BASE_MODEL="/home/woody/iwi5/iwi5280h/cil_models/eaml_cil_evm_scientific_publication_20250825_041346/best_model_scientific_publication.pth"
#BASE_MODEL="/home/woody/iwi5/iwi5280h/cil_models/eaml_cil_evm_with_Specification_20250831_171049/best_model_specification.pth.pth"
#BASE_MODEL="/home/woody/iwi5/iwi5280h/cil_models/eaml_cil_evm_with_file_folder_20250905_021230/best_model_file_folder.pth"
BASE_MODEL="/home/woody/iwi5/iwi5280h/cil_models/eaml_cil_evm_with_20250907_043152/final_global_best_model.pth"

CKPT_DIR="/home/woody/iwi5/iwi5280h/cil_models/eaml_cil_evm_with_20250907_043152" #eaml_cil_evm_with_$(date +%Y%m%d_%H%M%S)"

ALL_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,scientific_publication,specification,file_folder,news_article,budget,invoice,presentation,questionnaire,resume,memo"
# Subset trained on
#BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo"
#BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication"
#BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication,specification"
#BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication,specification,file_folder"
BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication,specification,file_folder,news_article"
# Incremental Training on
UNSEEN_CLASSES="budget" #,scientific_publication,specification,file_folder,news_article,budget"

mkdir -p $CKPT_DIR

echo "Starting Class Incremental Learning WITH EVM..."

python src/class_incremental_evm.py \
  --data_dir "$DATA_DIR"  \
  --ocr_tensor_path "$OCR_TENSOR_PATH" \
  --checkpoint_dir "$CKPT_DIR" \
  --model_name "eaml" \
  --all_classes "$ALL_CLASSES" \
  --base_classes "$BASE_CLASSES" \
  --unseen_classes "$UNSEEN_CLASSES" \
  --batch_size 8 \
  --lr 1e-3 \
  --num_epochs 100 \
  --strategy "distillation" \
  --temperature 2.0 \
  --lambda_distill 1.0 \
  --lambda_ewc 5000.0 \
  --use_ewc \
  --use_exemplars \
  --max_exemplars 320 \
  --exemplar_selection "herding" \
  --training_mode "last_layer" \
  --base_model_path "$BASE_MODEL" \
  --evm_tailsize 0.3 \
  --evm_threshold 0.7 \
  --full_model_acc 0.6173 \
  --patience 10 \
  --resume \
  --resume_checkpoint "/home/woody/iwi5/iwi5280h/cil_models/eaml_cil_evm_with_20250907_043152/epoch11_budget.pth" \
  #--global_best_acc 0.920

echo "Class Incremental Learning WITH EVM Completed."


# To run:
# sbatch run_classIL_evm.sh

#--data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \

# 1. --base_model_path /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250511_160135/eaml_best_model.pt\
# 2. --base_model_path /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250601_175404/eaml_best_model.pt\