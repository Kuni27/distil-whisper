#!/usr/bin/env bash

TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000 python3 run_distillation.py \
  --model_name_or_path "distil-whisper/large-32-2" \
  --teacher_model_name_or_path "openai/whisper-large-v2" \
  --train_dataset_config_name "all+all+all+en+en+ihm+sdm+clean+all+L" \
  --train_dataset_samples "2.9+10.4+14.9+89+18.2+10.9+10.9+288+371.2+192.7" \
  --train_dataset_name "librispeech_asr+librispeech_asr+librispeech_asr+common_voice_13_0+voxpopuli+ami-ihm+ami-sdm+peoples_speech-clean+switchboard-data+spgispeech" \
  --train_split_name "train.clean.100+train.clean.360+train.other.500+train+train+train+train+train+train+train" \
  --eval_dataset_name "distil-whisper/gigaspeech-l+esb/diagnostic-dataset+esb/diagnostic-dataset+esb/diagnostic-dataset+esb/diagnostic-dataset+esb/diagnostic-dataset+esb/diagnostic-dataset+esb/diagnostic-dataset+esb/diagnostic-dataset+esb/diagnostic-dataset+esb/diagnostic-dataset+esb/diagnostic-dataset+esb/diagnostic-dataset" \
  --eval_dataset_config_name "l+librispeech+librispeech+common_voice+common_voice+voxpopuli+voxpopuli+tedlium+tedlium+spgispeech+spgispeech+ami+ami" \
  --eval_split_name "validation+clean+other+clean+other+clean+other+clean+other+clean+other+clean+other" \
  --eval_text_column_name "text+ortho_transcript+ortho_transcript+ortho_transcript+ortho_transcript+ortho_transcript+ortho_transcript+ortho_transcript+ortho_transcript+ortho_transcript+ortho_transcript+ortho_transcript+ortho_transcript" \
  --eval_steps 1250 \
  --save_steps 1250 \
  --warmup_steps 250 \
  --learning_rate 0.0001 \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --save_total_limit 1 \
  --max_steps 20000 \
  --wer_threshold 10 \
  --per_device_train_batch_size 128 \
  --per_device_eval_batch_size 128 \
  --dtype "bfloat16" \
  --precision "full_mixed" \
  --dataloader_num_workers 16 \
  --cache_dir "/fsx/sanchit/.cache" \
  --dataset_cache_dir "/fsx/sanchit/.cache" \
  --output_dir "./" \
  --wandb_name "large-32-2-gpu-flat-lr" \
  --wandb_dir "/fsx/sanchit/.cache" \
  --wandb_project "distil-whisper" \
  --do_train \
  --do_eval \
  --use_scan \
  --gradient_checkpointing \
  --overwrite_output_dir \
  --predict_with_generate \
  --freeze_encoder \
  --streaming \
  --use_auth_token \
  --push_to_hub