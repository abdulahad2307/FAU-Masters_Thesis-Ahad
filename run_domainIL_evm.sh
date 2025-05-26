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

#echo "Starting CIL Test Run..."

# Test with just 3 classes in incremental steps
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# Domain Incremental Learning Configuration
DATA_DIR="/home/woody/iwi5/iwi5280h/dataset/"
CHECKPOINT_DIR="checkpoints/domain_il_evm"
EAML_MODEL_PATH="/home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250511_160135/eaml_best_model.pt"
DOCFORMER_MODEL_PATH=""

# EVM Configuration
USE_EVM=true                              # Set to false to disable EVM
EVM_TAILSIZE=0.5
EVM_THRESHOLD=0.7
EVM_UPDATE_FREQ=3

# Create checkpoint directory
mkdir -p "${CHECKPOINT_DIR}"

# Print system information
echo "========================================="
echo "Domain Incremental Learning with EVM"
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start Time: $(date)"
echo "EVM Enabled: ${USE_EVM}"
echo ""

# GPU Information
echo "=== GPU Information ==="
nvidia-smi -L
echo ""

# Validate data directories
echo "=== Validating Data Directories ==="
for domain in small_dataset small_dataset2; do
    if [ ! -d "${DATA_DIR}/${domain}" ]; then
        echo "ERROR: ${domain} directory not found at ${DATA_DIR}/${domain}"
        exit 1
    fi
    echo "${domain} directory found"
done

echo "Data validation completed"
echo ""

echo "=== Starting Domain Incremental Learning with EVM ==="

# Build command with EVM options
CMD="python src/domain_incremental_evm.py \
  --data_dir \"${DATA_DIR}\" \
  --domain_list small_dataset small_dataset2 \
  --class_counts \"small_dataset:16,small_dataset2:16\" \
  --ckpt_dir \"${CHECKPOINT_DIR}\" \
  --eaml_path \"${EAML_MODEL_PATH}\" \
  --docformer_path \"${DOCFORMER_MODEL_PATH}\" \
  --model eaml \
  --batch_size 16 \
  --epochs 200 \
  --lr 1e-4 \
  --finetune_mode head_only \
  --num_classes 16"

# Add EVM parameters if enabled
if [ "$USE_EVM" = true ]; then
    CMD="$CMD \
  --use_evm \
  --evm_tailsize ${EVM_TAILSIZE} \
  --evm_threshold ${EVM_THRESHOLD} \
  --evm_update_freq ${EVM_UPDATE_FREQ}"
fi

# Execute the command
eval $CMD

# Check exit status
exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Domain Incremental Learning Completed Successfully"
    echo "Training Duration: $SECONDS seconds"
    echo "Checkpoints saved in: ${CHECKPOINT_DIR}"
    echo "End Time: $(date)"
    
    # List generated checkpoints
    echo ""
    echo "=== Generated Checkpoints ==="
    ls -la "${CHECKPOINT_DIR}/"
    
else
    echo ""
    echo "========================================="
    echo "Domain Incremental Learning FAILED"
    echo "Exit Code: $exit_code"
    echo "Check logs for details"
fi

echo "========================================="
exit $exit_code



# To run:
# sbatch run_domainIL_evm.sh
# sbatch dil_test_run.sh

#--data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \