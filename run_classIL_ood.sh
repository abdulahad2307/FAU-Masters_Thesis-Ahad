#!/bin/bash -l

#SBATCH --job-name=eaml_cil_ood_with_bsamp_with_bcor        # Job name
#SBATCH --output=logs/cil_eaml_ood_4-2_%j.out    # Standard output log
#SBATCH --error=logs/cil_eaml_ood_4-2_%j.err     # Error log
#SBATCH --partition=v100                 # GPU partition name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=1                # CPU cores
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

# Paths
DATA_DIR="/home/woody/iwi5/iwi5280h/dataset/all_prepdataset"
OCR_TENSOR_PATH="/home/woody/iwi5/iwi5280h/dataset/all_predataset_combined_ocr_texts_rectbbox_tesseract.pt"
#BASE_MODEL="/home/woody/iwi5/iwi5280h/cil_models/all_eaml_SGD_tesseract_20250805_223607/eaml_best_model.pt"
#BASE_MODEL="/home/woody/iwi5/iwi5280h/cil_models/eaml_cil_ood_with_bsamp_with_bcor20250826_174249/best_model_scientific_publication.pth"
#BASE_MODEL="/home/woody/iwi5/iwi5280h/cil_models/eaml_cil_ood_with_bsamp_with_bcor20250826_174249/best_model_specification.pth"
BASE_MODEL="/home/woody/iwi5/iwi5280h/cil_models/eaml_cil_ood_with_bsamp_with_bcor20250826_174249/best_model_file_folder.pth"

CKPT_DIR="/home/woody/iwi5/iwi5280h/cil_models/eaml_cil_ood_with_bsamp_with_bcor20250826_174249" #eaml_cil_ood_with_bsamp_with_bcor$(date +%Y%m%d_%H%M%S)"


ALL_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,scientific_publication,specification,file_folder,news_article,budget,invoice,presentation,questionnaire,resume,memo"
# Subset trained on
#BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo"
#BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication"
#BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication,specification"
BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific_publication,specification,file_folder"

# Incremental Training on
UNSEEN_CLASSES="news_article" #"scientific_publication,specification,file_folder,news_article,budget"


# Model and training params
MODEL_NAME="eaml"  # docformer
OOD_METHOD="vim"   #  "msp", "vim", "gradnorm"
BATCH_SIZE=8
LR=1e-3
WEIGHT_DECAY=0.01
NUM_EPOCHS=100
STRATEGY="distillation"
TEMPERATURE=2.0
LAMBDA_DISTILL=1.0
USE_EWC=true 
LAMBDA_EWC=5000.0
USE_EXEMPLARS=true
MAX_EXEMPLARS=320
EXEMPLAR_SELECTION="herding"
TRAINING_MODE="last_layer"
PATIENCE=10
USE_BALANCED_SAMPLER=true
USE_BIAS_CORRECTION=true
OOD_THRESHOLD=0.7

# Resume options
RESUME=true
RESUME_CHECKPOINT="/home/woody/iwi5/iwi5280h/cil_models/eaml_cil_ood_with_bsamp_with_bcor20250826_174249/epoch10_news_article.pth"
mkdir -p $CKPT_DIR

echo "Starting Enhanced Class Incremental Learning with OOD..."

python src/class_incremental_ood.py \
  --data_dir "$DATA_DIR" \
  --ocr_tensor_path "$OCR_TENSOR_PATH" \
  --all_classes "$ALL_CLASSES" \
  --base_classes "$BASE_CLASSES" \
  --unseen_classes "$UNSEEN_CLASSES" \
  --base_model_path "$BASE_MODEL" \
  --model_name "$MODEL_NAME" \
  --checkpoint_dir "$CKPT_DIR" \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --full_model_acc 0.8762\
  --weight_decay $WEIGHT_DECAY \
  --num_epochs $NUM_EPOCHS \
  --strategy $STRATEGY \
  --temperature $TEMPERATURE \
  --lambda_distill $LAMBDA_DISTILL \
  $( [ "$USE_EWC" = true ] && echo "--use_ewc" ) \
  --lambda_ewc $LAMBDA_EWC \
  $( [ "$USE_EXEMPLARS" = true ] && echo "--use_exemplars" ) \
  --max_exemplars $MAX_EXEMPLARS \
  --exemplar_selection $EXEMPLAR_SELECTION \
  --training_mode $TRAINING_MODE \
  --patience $PATIENCE \
  $( [ "$USE_BALANCED_SAMPLER" = true ] && echo "--use_balanced_sampler" ) \
  $( [ "$USE_BIAS_CORRECTION" = true ] && echo "--use_bias_correction" ) \
  --ood_method $OOD_METHOD \
  --ood_threshold $OOD_THRESHOLD \
  $( [ "$RESUME" = true ] && echo "--resume" ) \
  $( [ -n "$RESUME_CHECKPOINT" ] && echo "--resume_checkpoint $RESUME_CHECKPOINT" )

echo "Class Incremental Learning with OOD Completed."


# To run:
# sbatch run_classIL_ood.sh

#--data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \

# 1. --base_model_path /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250511_160135/eaml_best_model.pt\
# 2. --base_model_path /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250601_175404/eaml_best_model.pt\