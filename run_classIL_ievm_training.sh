#!/bin/bash -l

#SBATCH --job-name=eaml_cil_ievm_training_SILS          # Job name
#SBATCH --output=CILlogs/%x-4_%j.out    # Standard output log
#SBATCH --error=CILlogs/%x-4_%j.err     # Error log
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
ALL_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,scientific_publication,specification,file_folder,news_article,budget,invoice,presentation,questionnaire,resume,memo"

#1
#BASE_MODEL="/home/woody/iwi5/iwi5280h/emal_models/outputs/outputs/11_eaml_adamW_tesseract_20250905_004113/eaml_best_model.pt"
#BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo"
# Incremental Training on
#UNSEEN_CLASSES="scientific_publication"

#2
#BASE_MODEL="/home/woody/iwi5/iwi5280h/eaml_cil/eaml_cil_ievm_training_SILS_Final/best_model_scientific_publication.pth"
#BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication"
# Incremental Training on
#UNSEEN_CLASSES="specification"

#3
#BASE_MODEL="/home/woody/iwi5/iwi5280h/eaml_cil/eaml_cil_ievm_training_SILS_Final/best_model_specification.pth "
#BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication,specification"
# Incremental Training on
#UNSEEN_CLASSES="file_folder"
#RESUME_CKPT="/home/woody/iwi5/iwi5280h/eaml_cil/eaml_cil_ievm_training_SILS_Final/epoch6_file_folder.pth"

#4
BASE_MODEL="/home/woody/iwi5/iwi5280h/eaml_cil/eaml_cil_ievm_training_SILS_Final/best_model_file_folder.pth"
BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication,specification,file_folder"
UNSEEN_CLASSES="news_article" 

#5
#BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication,specification,file_folder,news_article"
#BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication,specification,file_folder,news_article,budget"

# Incremental Training on
#UNSEEN_CLASSES="news_article" #"scientific_publication,specification,file_folder,news_article,budget"

CKPT_DIR="/home/woody/iwi5/iwi5280h/eaml_cil/eaml_cil_ievm_training_SILS_Final" #eaml_cil_evm_training_with_bsamp_with_bcor_$(date +%Y%m%d_%H%M%S)"

mkdir -p $CKPT_DIR

echo "Starting Class Incremental Learning WITH iEVM..."

python src/class_incremental_ievm_training.py \
  --data_dir "$DATA_DIR"  \
  --ocr_tensor_path "$OCR_TENSOR_PATH" \
  --checkpoint_dir "$CKPT_DIR" \
  --model_name "eaml" \
  --all_classes "$ALL_CLASSES" \
  --base_classes "$BASE_CLASSES" \
  --unseen_classes "$UNSEEN_CLASSES" \
  --batch_size 64 \
  --lr 1e-3 \
  --num_epochs 100 \
  --strategy "standard" \
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
  --full_model_acc 0.7235 \
  --patience 5 \
  #--resume \
  #--resume_checkpoint "$RESUME_CKPT" \
  #--global_best_acc <previous_global_best_acc> #0.9321 #0.8515 , 0.7960

echo "Class Incremental Learning WITH iEVM Completed."


# To run:
# sbatch run_classIL_ievm_training.sh

#--data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \

# 1. --base_model_path /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250511_160135/eaml_best_model.pt\
# 2. --base_model_path /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250601_175404/eaml_best_model.pt\