#!/bin/bash -l

#SBATCH --job-name=eaml_cil_evm_ood_KDILS          # Job name
#SBATCH --output=CILlogs/%x-4_%j.out    # Standard output log
#SBATCH --error=CILlogs/%x-4_%j.err     # Error log
#SBATCH --partition=v100                 # GPU partition name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=2                # CPU cores
#SBATCH --gres=gpu:v100:1                # Number of GPUs
#SBATCH --time=23:59:00                  # Shorter time for test run
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
#UNSEEN_CLASSES="scientific_publication"

#2
BASE_MODEL="/home/woody/iwi5/iwi5280h/eaml_cil/eaml_cil_evm_ood_KDILS_Final/best_model_scientific_publication.pth"
BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication"
# Incremental Training on
UNSEEN_CLASSES="specification"

#3
BASE_MODEL="/home/woody/iwi5/iwi5280h/eaml_cil/eaml_cil_evm_ood_KDILS_Final/best_model_specification.pth"
BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication,specification"
UNSEEN_CLASSES="file_folder"

#4
BASE_MODEL="/home/woody/iwi5/iwi5280h/eaml_cil/eaml_cil_evm_ood_KDILS_Final/best_model_file_folder.pth"
BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication,specification,file_folder"
UNSEEN_CLASSES="news_article"

#BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication,specification,file_folder,news_article"
#BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication,specification,file_folder,news_article,budget"

# Incremental Training on
#UNSEEN_CLASSES="news_article" #"scientific_publication,specification,file_folder,news_article,budget"

CKPT_DIR="/home/woody/iwi5/iwi5280h/eaml_cil/eaml_cil_evm_ood_KDILS_Final" #eaml_cil_evm_ood_with_bsamp_with_bcor_$(date +%Y%m%d_%H%M%S)"

mkdir -p $CKPT_DIR

echo "Starting Class Incremental Learning WITH EVM+OOD..."

python src/class_incremental_evm_ood.py \
  --data_dir "$DATA_DIR" \
  --ocr_tensor_path "$OCR_TENSOR_PATH" \
  --all_classes "$ALL_CLASSES" \
  --base_classes "$BASE_CLASSES" \
  --unseen_classes "$UNSEEN_CLASSES" \
  --base_model_path "$BASE_MODEL" \
  --model_name eaml \
  --checkpoint_dir "$CKPT_DIR" \
  --ood_method vim \
  --ood_threshold 0.75 \
  --batch_size 64 \
  --lr 1e-3 \
  --num_epochs 100 \
  --strategy distillation \
  --temperature 2.0 \
  --lambda_distill 1.0 \
  --lambda_ewc 5000.0 \
  --use_ewc \
  --use_exemplars \
  --max_exemplars 320 \
  --exemplar_selection herding \
  --training_mode last_layer \
  --full_model_acc 0.8394 \
  --weight_decay 0.01 \
  --patience 5 \
  --use_balanced_sampler \
  --use_bias_correction \
  --lambda_evm 0.1 \
  --lambda_ood 0.1 \
  #--resume \
  #--resume_checkpoint "/home/woody/iwi5/iwi5280h/eaml_cil/eaml_cil_evm_ood_KDILS_Final/epoch6_scientific_publication.pth" \
  #--global_best_acc 0.953 # 0.9321, 0.8832, 0.8321

echo "Class Incremental Learning WITH EVM+OOD Completed."


# To run:
# sbatch run_classIL_evm_ood_KD.sh

#--data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \

# 1. --base_model_path /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250511_160135/eaml_best_model.pt\
# 2. --base_model_path /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250601_175404/eaml_best_model.pt\
