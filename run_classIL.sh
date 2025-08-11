#!/bin/bash -l

#SBATCH --job-name=eaml_CIL         # Job name
#SBATCH --output=logs/cil_test_%j.out    # Standard output log
#SBATCH --error=logs/cil_test_%j.err     # Error log
#SBATCH --partition=v100                 # GPU partition name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=4                # CPU cores
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

# Test with just 3 classes in incremental steps
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

OCR_TENSOR_PATH="/home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_tesseract.pt"
BASE_MODEL="/home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/all_eaml_SGD_tesseract_20250805_223607/eaml_best_model.pt"
CKPT_DIR="outputs/eaml_CIL/_cil_$(date +%Y%m%d_%H%M%S)"

ALL_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,scientific_publication,specification,file_folder,news_article,budget,invoice,presentation,questionnaire,resume,memo"
# Subset trained on
CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo"
mkdir -p $CKPT_DIR

echo "Starting Enhanced Class Incremental Learning..."

python src/class_incremental.py \
  --data_dir /home/woody/iwi5/iwi5280h/dataset/all_prepdataset \
  --ocr_tensor_path $OCR_TENSOR_PATH\
  --checkpoint_dir $CKPT_DIR \
  --model_name "eaml" \
  --class_order $ALL_CLASSES \
  --start_step 11 \
  --batch_size 16 \
  --lr 1e-3 \
  --num_epochs 50 \
  --strategy "distillation" \
  --temperature 2.0 \
  --lambda_distill 1.0 \
  --lambda_ewc 5000.0 \
  --use_ewc \
  --use_exemplars \
  --max_exemplars 200 \
  --exemplar_selection "herding" \
  --training_mode "last_layer" \
  --base_model_path $BASE_MODEL\
  --full_model_acc 0.953


echo "Class Incremental Learning Completed."

# To run:
# sbatch run_classIL.sh

#--data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \

# 1. --base_model_path /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250511_160135/eaml_best_model.pt\
# 2. --base_model_path /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250601_175404/eaml_best_model.pt\