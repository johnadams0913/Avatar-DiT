#!/bin/bash
# Stage 3: joint fine-tuning — rank-64 LoRA on backbone attention/FFN (both adapters frozen)
# 8x H100, ~34h. lr 1e-5. Data: splits/camera-hybrid.json (12,761 clips).
DATA_ROOT=${DATA_ROOT:-./datasets}
accelerate launch --config_file ac-ddp train.py \
    -n stage3-joint \
    -d "$DATA_ROOT" \
    -bs 1 -nt 16 -lr 1e-5 -sfs 500 \
    -cfg configs/training/multiview-lora.yaml \
    --epoch 6
