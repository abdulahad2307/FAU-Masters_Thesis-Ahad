#!/bin/bash -l

#SBATCH --job-name=Combine_Tensors        # Job name
#SBATCH --output=logs/%x_%j.out           # Standard output log
#SBATCH --error=logs/%x_%j.err            # Error log
#SBATCH --partition=v100                  # GPU partition name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=1                 # Number of CPU cores per task
#SBATCH --gres=gpu:v100:1                 # Number of GPUs
#SBATCH --time=23:55:00                   # Time limit hrs:min:sec
#SBATCH --export=NONE                     # Avoid inheriting unwanted environment variables

unset SLURM_EXPORT_ENV

module load cuda/12.6
module load python/3.12-conda
conda activate ocr_env

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export PYTHONPATH=$PYTHONPATH:$(pwd)/FAU-Masters_Thesis-Ahad

OUTPUT_TENSORS ="/home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_trocr.pt"
echo "Combining Tensor Files"

python utils/combine_tensors.py \
  --inputs /home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_trocr1.pt /home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_trocr2.pt  /home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_trocr3.pt /home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_trocr4.pt \
  --output /home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_trocr.pt\
  --key input_ids

echo "Tensor combination Completed. Output: $OUTPUT_TENSORS"

#sbatch run_combinetensors.sh
#--data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \
#--data_dir /home/woody/iwi5/iwi5280h/dataset/small_dataset \