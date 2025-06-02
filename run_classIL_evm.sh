#!/bin/bash -l

#SBATCH --job-name=cil_evm_test_run          # Job name
#SBATCH --output=logs/cil_test_evm_%j.out    # Standard output log
#SBATCH --error=logs/cil_test_evm_%j.err     # Error log
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

echo "Starting Class Incremental Learning WITH EVM..."

python src/class_incremental_evm.py \
  --data_dir /home/woody/iwi5/iwi5280h/dataset/small_dataset \
  --checkpoint_dir checkpoints/enhanced_cil_evm \
  --model_name "eaml" \
  --class_order "letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo,scientific publication,specification,file folder,news article,budget" \
  --start_step 11 \
  --batch_size 4 \
  --lr 1e-4 \
  --num_epochs 300 \
  --strategy "distillation" \
  --temperature 2.0 \
  --lambda_distill 1.0 \
  --lambda_ewc 5000.0 \
  --use_ewc \
  --use_exemplars \
  --max_exemplars 200 \
  --exemplar_selection "herding" \
  --training_mode "last_layer" \
  --base_model_path /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250601_175404/eaml_best_model.pt\
  --evm_tailsize 0.5 \
  --evm_threshold 0.7\
  --full_model_acc 0.7775

echo "Class Incremental Learning WITH EVM Completed."

# To run:
# sbatch run_classIL_evm.sh

#--data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \

# 1. --base_model_path /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250511_160135/eaml_best_model.pt\
# 2. --base_model_path /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/eaml_20250601_175404/eaml_best_model.pt\