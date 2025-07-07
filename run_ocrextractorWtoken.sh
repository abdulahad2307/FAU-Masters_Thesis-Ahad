#!/bin/bash -l

#SBATCH --job-name=ocr_extraction_small2         # Job name
#SBATCH --output=logs/%x_%j.out           # Standard output log
#SBATCH --error=logs/%x_%j.err            # Error log
#SBATCH --partition=rtx3080                  # GPU partition name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=1                 # Number of CPU cores per task
#SBATCH --gres=gpu:rtx3080:1                 # Number of GPUs
#SBATCH --time=23:55:00                   # Time limit hrs:min:sec
#SBATCH --export=NONE                     # Avoid inheriting unwanted environment variables

unset SLURM_EXPORT_ENV

module load cuda/12.6
module load python/3.12-conda
conda activate ocr_env

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export PYTHONPATH=$PYTHONPATH:$(pwd)/FAU-Masters_Thesis-Ahad

echo "Starting OCR Extraction"

DATA_DIR="/home/woody/iwi5/iwi5280h/dataset/small_dataset2"
OUTPUT_TENSOR="/home/woody/iwi5/iwi5280h/dataset/small_dataset2_ocr_texts_docling.pt"
#OUTPUT_JSON="/home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_trocr1.json"
#OCR_ENGINE="trocr"
OCR_ENGINE="docling_ocr"
MAX_SIZE=1024

#utils/ocr_extraction_token.py \

python utils/ocr_extraction_docling.py \
  --data_dir $DATA_DIR \
  --output_pt $OUTPUT_TENSOR \
  --ocr_engine $OCR_ENGINE \
  --offset 0 \
  --max_images 3200 \
  --max_size 1024\
  --batch_size 4


echo "OCR Extraction Completed. Output: $OUTPUT_TENSOR |&| $OUTPUT_JSON "

#sbatch run_ocrextractorWtoken.sh
#--data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \
#--data_dir /home/woody/iwi5/iwi5280h/dataset/small_dataset \