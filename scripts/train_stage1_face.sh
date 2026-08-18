#!/bin/bash
# Stage 1: face control — train FLAME adapter + motion encoder (backbone frozen)
# 12x H100, ~9h. lr 2e-5, bf16, bs 1/GPU, 81-frame clips.
DATA_ROOT=${DATA_ROOT:-./datasets}
accelerate launch --config_file ac-ddp train.py \
    -n stage1-face \
    -d "$DATA_ROOT" \
    -bs 1 -nt 16 -lr 2e-5 -sfs 500 \
    -cfg configs/training/flame.yaml \
    --epoch 6
