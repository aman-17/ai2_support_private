#!/bin/bash

pip install ai2-olmo datasets wandb

OLMO_DIR="/myfiles/amanr/OLMo"
LOCAL_CHECKPOINT_DIR="/myfiles/amanr/stage2_olmo7b"
DEST_DIR="/myfiles/amanr/unsharded_stage2_olmo7b"
LOGS_DIR="/myfiles/amanr/logs_unsharded_stage2_olmo7b"
TEMP_DIR="/myfiles/ai2-llm/amanr/temp_unsharded_stage2_olmo7b"
BATCH_SIZE=4

mkdir -p "$LOGS_DIR"
mkdir -p "$TEMP_DIR"

ls "$LOCAL_CHECKPOINT_DIR" | \
    grep -Eo 'step[0-9]+' | \
    sed 's/step//' | \
    sort -n | \
    awk '$1 >= 0 && $1 <= 248000' | \
    parallel -j 4 \
    python "$OLMO_DIR/scripts/storage_cleaner.py" unshard \
    "$LOCAL_CHECKPOINT_DIR" "$DEST_DIR" \
    --checkpoint_num {} ">" "$LOGS_DIR/step{}-unshard.log" 2>&1
