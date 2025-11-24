#!/bin/bash -l

#SBATCH --job-name=eaml_CIL_with_SILS        # Job name
#SBATCH --output=CILlogs/%x-4_%j.out    # Standard output log
#SBATCH --error=CILlogs/%x-4_%j.err     # Error log
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

OCR_TENSOR_PATH="/home/woody/iwi5/iwi5280h/dataset/all_predataset_combined_ocr_texts_rectbbox_tesseract.pt"

ALL_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,scientific_publication,specification,file_folder,news_article,budget,invoice,presentation,questionnaire,resume,memo"

#1
#BASE_MODEL="/home/woody/iwi5/iwi5280h/emal_models/outputs/outputs/11_eaml_adamW_tesseract_20250905_004113/eaml_best_model.pt"
#BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo"
# Incremental Training on
#UNSEEN_CLASSES="scientific_publication"

#2
#BASE_MODEL="/home/woody/iwi5/iwi5280h/eaml_cil/eaml_cil_with_SILS_Final/best_model_scientific_publication.pth"
#BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication"
# Incremental Training on
#UNSEEN_CLASSES="specification"

#3
#BASE_MODEL="/home/woody/iwi5/iwi5280h/eaml_cil/eaml_cil_with_SILS_Final/best_model_specification.pth"
#BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication,specification"
#UNSEEN_CLASSES="file_folder"
#RESUME_CKPT="/home/woody/iwi5/iwi5280h/eaml_cil/eaml_cil_with_SILS_Final/epoch9_file_folder.pth"

#4
BASE_MODEL="/home/woody/iwi5/iwi5280h/eaml_cil/eaml_cil_with_SILS_Final/best_model_file_folder.pth"
BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication,specification,file_folder"
UNSEEN_CLASSES="news_article" #"scientific_publication,specification,file_folder,news_article,budget"

# Subset trained on
#BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication,specification,file_folder,news_article"
#BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication,specification,file_folder,news_article,budget"

CKPT_DIR="/home/woody/iwi5/iwi5280h/eaml_cil/eaml_cil_with_SILS_Final"

mkdir -p $CKPT_DIR

echo "Starting Enhanced Class Incremental Learning..."

python src/class_incremental.py \
  --data_dir /home/woody/iwi5/iwi5280h/dataset/all_prepdataset \
  --ocr_tensor_path "$OCR_TENSOR_PATH" \
  --all_classes "$ALL_CLASSES" \
  --base_classes "$BASE_CLASSES" \
  --unseen_classes "$UNSEEN_CLASSES" \
  --base_model_path "$BASE_MODEL" \
  --model_name "eaml" \
  --checkpoint_dir "$CKPT_DIR" \
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
  --full_model_acc 0.6860 \
  --weight_decay 0.01 \
  --patience 5 \
  #--resume \
  #--resume_checkpoint "$RESUME_CKPT" \

  # For resume:
  # --resume \
  # --resume_checkpoint "$RESUME_CKPT" \
  # --global_best_acc <previous_global_best_acc>

echo "Class Incremental Learning Completed."


# To run:
# sbatch run_classIL.sh

#--data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \

# 1. --base_model_path /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250511_160135/eaml_best_model.pt\
# 2. --base_model_path /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250601_175404/eaml_best_model.pt\