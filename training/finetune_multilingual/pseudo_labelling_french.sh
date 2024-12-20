#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=20:00:00
#SBATCH --mem=128G
#SBATCH --job-name=voxpopuli_french
#SBATCH --output=voxpopuli_french/voxpopuli_french.out
#SBATCH --error=voxpopuli_french/voxpopuli_french.err

# Create output directory
mkdir -p voxpopuli_french

# Load necessary modules
module load WebProxy

# Set proxy if needed
export http_proxy=http://10.73.132.63:8080
export https_proxy=http://10.73.132.63:8080
export WANDB_API_KEY=""

# Run the accelerate command with VoxPopuli dataset parameters
accelerate launch run_pseudo_labelling.py \
  --model_name_or_path "openai/whisper-large-v3" \
  --dataset_name "facebook/voxpopuli" \
  --dataset_config_name "fr" \
  --dataset_split_name "train+validation+test" \
  --text_column_name "normalized_text" \
  --id_column_name "audio_id" \
  --output_dir "./voxpopuli_fr_pseudo_labelled" \
  --wandb_project "distil-whisper-labelling-voxpopuli-french" \
  --per_device_eval_batch_size 10 \
  --dtype "bfloat16" \
  --attn_implementation "sdpa" \
  --logging_steps 500 \
  --max_label_length 256 \
  --concatenate_audio \
  --preprocessing_batch_size 500 \
  --preprocessing_num_workers 8 \
  --dataloader_num_workers 8 \
  --report_to "wandb" \
  --language "fr" \
  --task "transcribe" \
  --return_timestamps \
  --streaming False \
  --generation_num_beams 1 \
  --push_to_hub
