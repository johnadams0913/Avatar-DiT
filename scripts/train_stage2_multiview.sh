#!/bin/bash
# Stage 2: multi-view control — train camera modulation + patch embeddings (FLAME adapter frozen)
# 8x H100, ~32h. lr 1e-5.
DATA_ROOT=${DATA_ROOT:-./datasets}
accelerate launch --config_file ac-ddp train.py \
    -n stage2-multiview \
    -d "$DATA_ROOT" \
    -bs 1 -nt 16 -lr 1e-5 -sfs 500 \
    -cfg configs/training/multiview.yaml \
    --epoch 6
