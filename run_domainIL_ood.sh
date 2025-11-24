#!/bin/bash -l

#SBATCH --job-name=dil_with_ood          # Job name
#SBATCH --output=dil_logs/%x_%j.out    # Standard output log
#SBATCH --error=dil_logs/%x_%j.err     # Error log
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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# Domain Incremental Learning Configuration

DATA_DIR="/home/woody/iwi5/iwi5280h/dataset"
CHECKPOINT_DIR="/home/woody/iwi5/iwi5280h/dil_models/eaml_dil_normal$(date +%Y%m%d_%H%M%S)"
EAML_MODEL_PATH="/home/woody/iwi5/iwi5280h/cil_models/all_class_eaml_SGD_tesseract_20250807_215555/eaml_best_model.pt"

# Domain and class config
#DOMAINS="all_prepdataset,Tobacco3482-jpg"
GLOBAL_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,scientific_publication,specification,file_folder,news_article,budget,invoice,presentation,questionnaire,resume,memo,Note,Report"

LR=1e-4
EPOCHS=200
BATCH_SIZE=16
UNFREEZE_DEPTH=2
OOD_METHOD="vim"      # Available: msp, vim, gradnorm
OOD_THRESHOLD=0.75
PATIENCE=10

# OCR tensor directories
#OCR_TENSOR_DIRS="None" "/home/woody/iwi5/iwi5280h/dataset/Tobacco3482_ocr_texts_tesseract.pt"

mkdir -p $CHECKPOINT_DIR

# Validate data directories
echo "=== Validating Data Directories ==="
for dom in all_prepdataset Tobacco3482-jpg; do
    if [ ! -d "${DATA_DIR}/${dom}" ]; then
        echo "ERROR: ${dom} directory not found at ${DATA_DIR}/${dom}"
        exit 1
    fi
done
echo "all_prepdataset directory found"
echo "Tobacco3482 directory found"
echo ""

echo "========================================="
echo "Starting Domain Incremental Learning with OOD"
echo "Data Directory: ${DATA_DIR}"
echo "Device: $(nvidia-smi -L)"
echo "========================================="

python src/domain_incremental_ood.py \
  --data_dir "${DATA_DIR}" \
  --ocr_tensor_dirs "None" "/home/woody/iwi5/iwi5280h/dataset/Tobacco3482_ocr_texts_tesseract.pt" \
  --domains "all_prepdataset,Tobacco3482-jpg" \
  --global_classes "${GLOBAL_CLASSES}" \
  --eaml_ckpt_path "${EAML_MODEL_PATH}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --num_epochs ${EPOCHS} \
  --strategy distillation \
  --temperature 2.0 \
  --lambda_distill 1.0 \
  --lambda_ewc 5000.0 \
  --use_ewc \
  --unfreeze_depth ${UNFREEZE_DEPTH} \
  --patience ${PATIENCE} \
  --use_bias_correction \
  --ood_method "${OOD_METHOD}" \
  --ood_threshold ${OOD_THRESHOLD}

echo "========================================="
echo "Domain Incremental Learning with OOD Completed"
echo "Checkpoints saved in: ${CHECKPOINT_DIR}"



# To run:
# sbatch run_domainIL.sh

#--data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \

# 1. --base_model_path /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250511_160135/eaml_best_model.pt\
# 2. --base_model_path /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250601_175404/eaml_best_model.pt\