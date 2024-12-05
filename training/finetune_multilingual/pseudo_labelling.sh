#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --job-name=pseudo_labelling
#SBATCH --output=pseudo_labelling/pseudo_labelling.out
#SBATCH --error=pseudo_labelling/pseudo_labelling.err

# Create output directory
mkdir -p pseudo_labelling

# Load necessary modules
module load WebProxy

# Set proxy if needed
export http_proxy=http://10.72.8.25:8080
export https_proxy=http://10.72.8.25:8080

# Set WandB API key
export WANDB_API_KEY="856a2fc7b95dac753d8efc995d2136072384bb27"

# Run the pseudo labelling script
accelerate launch run_pseudo_labelling.py \
  --model_name_or_path "openai/whisper-large-v3" \
  --dataset_name "./mls_dutch+./common_voice_dutch" \
  --dataset_split_name "train" \
  --output_dir "./pseudo_labeled_dutch" \
  --language "nl" \
  --task "transcribe" \
  --dtype "bfloat16" \
  --attn_implementation "sdpa" \
  --logging_steps 500 \
  --max_label_length 256 \
  --concatenate_audio \
  --return_timestamps \
  --streaming False \
  --generation_num_beams 1 \
  --preprocessing_batch_size 250 \
  --preprocessing_num_workers 8 \
  --dataloader_num_workers 8 \
  --push_to_hub
