#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --job-name=pseudo_labelling_test_italian_common_voice
#SBATCH --output=pseudo_labelling_test_italian_common_voice_logs/pseudo_labelling_test_italian_common_voice.out
#SBATCH --error=pseudo_labelling_test_italian_common_voice_logs/pseudo_labelling_test_italian_common_voice.err

# Create output directory
mkdir -p pseudo_labelling_test_italian_common_voice_logs

# Load necessary modules
module load WebProxy

# Set proxy if needed
export http_proxy=http://10.73.132.63:8080
export https_proxy=http://10.73.132.63:8080

# Set WandB API key
export WANDB_API_KEY=""
export HF_ALLOW_CODE_EXECUTION=1


accelerate launch run_pseudo_labelling.py \
  --model_name_or_path "openai/whisper-large-v3" \
  --dataset_name "deepdml/common_voice_17_0_it_pseudo_labelled-large-v3" \
  --dataset_config_name "it" \
  --dataset_split_name "train+validation+test" \
  --text_column_name "sentence" \
  --id_column_name "path" \
  --output_dir "./common_voice_16_0_it_pseudo_labelled" \
  --wandb_project "distil-whisper-labelling-italian-common-voice" \
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
  --language "it" \
  --task "transcribe" \
  --return_timestamps \
  --streaming False \
  --generation_num_beams 1 \
  --push_to_hub

  
  