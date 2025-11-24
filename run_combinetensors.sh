#!/bin/bash -l

#SBATCH --job-name=combine_tensor_tesseract        # Job name
#SBATCH --output=logs/%x_%j.out           # Standard output log
#SBATCH --error=logs/%x_%j.err            # Error log
#SBATCH --partition=broadwell512                  # GPU partition name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=16                 # Number of CPU cores per task
#SBATCH --time=23:55:00                   # Time limit hrs:min:sec
#SBATCH --export=NONE                     # Avoid inheriting unwanted environment variables

unset SLURM_EXPORT_ENV

module load cuda/12.6
module load python/3.12-conda
conda activate ocr_env

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export PYTHONPATH=$PYTHONPATH:$(pwd)/FAU-Masters_Thesis-Ahad

echo "Combining Tensor Files"

python utils/combine_tensorTokens.py \
  --inputs /home/woody/iwi5/iwi5280h/dataset/all_prepdataset_ocr_texts_tesseract1.pt /home/woody/iwi5/iwi5280h/dataset/all_prepdataset_ocr_texts_tesseract2.pt /home/woody/iwi5/iwi5280h/dataset/all_prepdataset_ocr_texts_tesseract3.pt /home/woody/iwi5/iwi5280h/dataset/all_prepdataset_ocr_texts_tesseract4.pt /home/woody/iwi5/iwi5280h/dataset/all_prepdataset_ocr_texts_tesseract5.pt /home/woody/iwi5/iwi5280h/dataset/all_prepdataset_ocr_texts_tesseract6.pt /home/woody/iwi5/iwi5280h/dataset/all_prepdataset_ocr_texts_tesseract7.pt /home/woody/iwi5/iwi5280h/dataset/all_prepdataset_ocr_texts_tesseract8.pt \
  --output /home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_tesseract.pt\
  
echo "Tensor combination Completed. Output: $OUTPUT_TENSORS"

#sbatch.tinyfat run_combinetensors.sh
#--data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \
#--data_dir /home/woody/iwi5/iwi5280h/dataset/small_dataset \