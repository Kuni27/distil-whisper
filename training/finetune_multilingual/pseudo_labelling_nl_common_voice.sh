#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=20:00:00
#SBATCH --mem=128G
#SBATCH --job-name=pseudo_labelling_french
#SBATCH --output=pseudo_labelling_french/pseudo_labelling_french.out
#SBATCH --error=pseudo_labelling_french/pseudo_labelling_french.err

# Create output directory
mkdir -p pseudo_labelling_french

# Load necessary modules
module load WebProxy

# Set proxy if needed
export http_proxy=http://10.72.8.25:8080
export https_proxy=http://10.72.8.25:8080

# Set WandB API key
export WANDB_API_KEY=""
export HF_ALLOW_CODE_EXECUTION=1


accelerate launch run_pseudo_labelling.py \
  --model_name_or_path "openai/whisper-large-v3" \
  --dataset_name "mozilla-foundation/common_voice_16_1" \
  --dataset_config_name "nl" \
  --dataset_split_name "train+validation+test" \
  --text_column_name "sentence" \
  --id_column_name "path" \
  --output_dir "./common_voice_16_0_fr_pseudo_labelled" \
  --wandb_project "distil-whisper-labelling-french" \
  --per_device_eval_batch_size 64 \
  --dtype "bfloat16" \
  --attn_implementation "sdpa" \
  --logging_steps 500 \
  --max_label_length 256 \
  --concatenate_audio \
  --preprocessing_batch_size 500 \
  --preprocessing_num_workers 8 \
  --dataloader_num_workers 8 \
  --report_to "wandb" \
  --language "nl" \
  --task "transcribe" \
  --return_timestamps \
  --streaming False \
  --generation_num_beams 1 \
  --push_to_hub

  
  


