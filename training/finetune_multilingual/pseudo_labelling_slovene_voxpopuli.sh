#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --job-name=pseudo_labelling_test_slovene_voxpopuli
#SBATCH --output=pseudo_labelling_test_slovene_voxpopuli_logs/pseudo_labelling_test_slovene_voxpopuli.out
#SBATCH --error=pseudo_labelling_test_slovene_voxpopuli_logs/pseudo_labelling_test_slovene_voxpopuli.err

# Create output directory
mkdir -p pseudo_labelling_test_slovene_voxpopuli_logs

# Load necessary modules
module load WebProxy

# Set proxy if needed
export http_proxy=http://10.72.8.25:8080
export https_proxy=http://10.72.8.25:8080

# Set WandB API key
export WANDB_API_KEY=""
export HF_ALLOW_CODE_EXECUTION=1


# Run the accelerate command with VoxPopuli dataset parameters
accelerate launch run_pseudo_labelling.py \
  --model_name_or_path "openai/whisper-large-v3" \
  --dataset_name "facebook/voxpopuli" \
  --dataset_config_name "sl" \
  --dataset_split_name "train+validation+test" \
  --text_column_name "normalized_text" \
  --id_column_name "audio_id" \
  --output_dir "./voxpopuli_sl_pseudo_labelled" \
  --wandb_project "distil-whisper-labelling-slovene" \
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
  --language "sl" \
  --task "transcribe" \
  --return_timestamps \
  --streaming False \
  --generation_num_beams 1 \
  --push_to_hub
  
  


