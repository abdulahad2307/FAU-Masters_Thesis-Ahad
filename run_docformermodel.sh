#!/bin/bash -l

#SBATCH --job-name=alldocformer_training     # Job name
#SBATCH --output=logs/docformer_all_training_%j.out    # Standard output log
#SBATCH --error=logs/docformer_all_training_%j.err     # Error log
#SBATCH --partition=v100                  # GPU partition name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=4                 # Increased CPU cores for OCR processing
#SBATCH --gres=gpu:v100:1                 # Number of GPUs
#SBATCH --time=23:30:00                   # Time limit hrs:min:sec
#SBATCH --export=NONE                     # Avoid inheriting unwanted environment variables

unset SLURM_EXPORT_ENV

# Load required modules
# Load modules
module load cuda/12.6
module load python/3.12-conda
conda activate mtil

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# Move to the repository folder
export PYTHONPATH=$PYTHONPATH:$(pwd)/FAU-Masters_Thesis-Ahad


echo "Starting Progressive DocFormer Training..."

# Configuration
OCR_ENGINE=${1:-"trocr"} # tesseract, trocr, pero, easyocr, paddleocr
DATA_DIR=${2:-"/home/woody/iwi5/iwi5280h/dataset/small_dataset"}
OUTPUT_DIR="docformer_outputs_$(date +%Y%m%d_%H%M%S)"  
CLASSES="advertisement,budget,email,file_folder,form,handwritten,invoice,letter,memo,news_article,presentation,questionnaire,resume,scientific_publication,scientific_report,specification"

mkdir -p $OUTPUT_DIR
mkdir -p logs

if [ "$EVAL_ONLY" = "true" ]; then
    echo "=== Running Evaluation Only ==="
    
    # Check if there's an existing model to evaluate
    if [ ! -f "$OUTPUT_DIR/best_model.pt" ]; then
        echo "Error: No trained model found at $OUTPUT_DIR/best_model.pt"
        echo "Please train a model first or provide the correct output directory"
        exit 1
    fi
    
    python src/sota_docformer_model.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --ocr_engine "$OCR_ENGINE" \
        --classes "$CLASSES" \
        --evaluate_only
        
else
    echo "=== Starting Training ==="
    
    # Training with progressive stages
    python src/sota_docformer_model.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --ocr_engine "$OCR_ENGINE" \
        --classes "$CLASSES" \
        --text_epochs 3 \
        --visual_epochs 3 \
        --final_epochs 5 \
        --batch_size 4 \
        --finetune_batch_size 2 \
        --learning_rate 5e-5 \
        --finetune_lr 2.5e-5 \
        --num_workers 2 \
        --progressive

    echo "Training completed!"
    
    echo "=== Starting Final Evaluation ==="
    
    # Run evaluation after training
    python src/sota_docformer_model.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --ocr_engine "$OCR_ENGINE" \
        --classes "$CLASSES" \
        --evaluate_only
fi

echo "=== Process Completed ==="
echo "Results saved to: $OUTPUT_DIR"
echo "Training and Evaluation Completed."

# sbatch run_docformermodel.sh
#--data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \
#"/home/woody/iwi5/iwi5280h/dataset/small_dataset"