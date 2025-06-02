#!/bin/bash -l

#SBATCH --job-name=dil_evm_test_run          # Job name
#SBATCH --output=logs/dil_evm_test_%j.out    # Standard output log
#SBATCH --error=logs/dil_evm_test_%j.err     # Error log
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
DATA_DIR="/home/woody/iwi5/iwi5280h/dataset/"
CHECKPOINT_DIR="checkpoints/domain_il_evm"
EAML_MODEL_PATH="/home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250601_175404/eaml_best_model.pt"
DOCFORMER_MODEL_PATH=""

# Validate data directories exist
echo "=== Validating Data Directories ==="
if [ ! -d "${DATA_DIR}/small_dataset" ]; then
    echo "ERROR: small_dataset directory not found at ${DATA_DIR}/small_dataset"
    exit 1
fi

if [ ! -d "${DATA_DIR}/small_dataset2" ]; then
    echo "ERROR: small_dataset2 directory not found at ${DATA_DIR}/small_dataset2"
    exit 1
fi

echo "small_dataset directory found"
echo "small_dataset2 directory found"
echo ""

echo "========================================="
echo "Starting Domain Incremental Learning WITH EVM"
echo "Data Directory: ${DATA_DIR}"
echo "Device: $(nvidia-smi -L)"
echo "========================================="

python src/domain_incremental_evm.py \
  --data_dir "${DATA_DIR}" \
  --domain_list small_dataset small_dataset2 \
  --class_counts "small_dataset:16,small_dataset2:16" \
  --ckpt_dir "${CHECKPOINT_DIR}" \
  --eaml_path "${EAML_MODEL_PATH}" \
  --docformer_path "${DOCFORMER_MODEL_PATH}" \
  --model eaml \
  --batch_size 16 \
  --epochs 300 \
  --lr 1e-4 \
  --finetune_mode partial_finetune \
  --unfreeze_depth 2 \
  --num_classes 16 \
  --evm_tailsize 0.5 \
  --evm_threshold 0.7 \
  --evm_update_freq 5 \
  --full_model_acc 0.7775


echo "========================================="
echo "Domain Incremental Learning WITH EVM Completed"
echo "Checkpoints saved in: ${CHECKPOINT_DIR}"




# To run:
# sbatch run_domainIL_evm.sh

#--data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \

# 1. --base_model_path /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250511_160135/eaml_best_model.pt\
# 2. --base_model_path /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250601_175404/eaml_best_model.pt\