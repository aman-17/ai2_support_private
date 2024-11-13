#!/bin/bash

OLMO_DIR="/ai2_support_private/OLMo"
GCS_CHECKPOINT_DIR="gs://ai2-llm/checkpoints/OLMo-medium/peteish13-highlr"  
LOCAL_CHECKPOINT_DIR="/weka/oe-training-default/amanr/peteish13-highlrr/temp" 
DEST_DIR="/weka/oe-training-default/amanr/peteish13-highlrr"  
LOGS_DIR="/weka/oe-training-default/amanr/peteish13-highlrr/logs"  
SERVICE_ACCOUNT_JSON="/service_account.json" 
BATCH_SIZE=4  

if [ ! -f "$SERVICE_ACCOUNT_JSON" ]; then
    echo "Error: Service account JSON file not found at $SERVICE_ACCOUNT_JSON"
    exit 1
fi

export GOOGLE_APPLICATION_CREDENTIALS="$SERVICE_ACCOUNT_JSON"
gcloud auth activate-service-account --key-file="$SERVICE_ACCOUNT_JSON"

mkdir -p "$LOCAL_CHECKPOINT_DIR"
mkdir -p "$LOGS_DIR"
mkdir -p "$DEST_DIR"

get_checkpoint_numbers() {
    gsutil ls "$GCS_CHECKPOINT_DIR" | \
        grep -Eo 'step[0-9]+' | \
        sed 's/step//' | \
        sort -n | \
        awk '$1 >= 100500 && $1 <= 107000'
}

process_batch() {
    local batch_numbers=("$@")
    
    echo "Processing batch: step${batch_numbers[*]}"
    for checkpoint in "${batch_numbers[@]}"; do
        echo "Downloading step$checkpoint"
        gsutil -m cp -r "$GCS_CHECKPOINT_DIR/step$checkpoint" "$LOCAL_CHECKPOINT_DIR/"
    done
    
    printf "%s\n" "${batch_numbers[@]}" | \
        parallel -j $BATCH_SIZE \
        GOOGLE_APPLICATION_CREDENTIALS="$SERVICE_ACCOUNT_JSON" \
        python "$OLMO_DIR/scripts/storage_cleaner.py" unshard \
        "$LOCAL_CHECKPOINT_DIR" "$DEST_DIR" \
        --checkpoint_num {} --delete_sharded \
        ">" "$LOGS_DIR/step{}-unshard.log" 2>&1
    
    for checkpoint in "${batch_numbers[@]}"; do
        echo "Cleaning up local copy of step$checkpoint"
        rm -rf "$LOCAL_CHECKPOINT_DIR/step$checkpoint"
    done
    
    echo "Batch completed: step${batch_numbers[*]}"
}
CHECKPOINT_NUMBERS=($(get_checkpoint_numbers))
TOTAL_CHECKPOINTS=${#CHECKPOINT_NUMBERS[@]}

echo "Found $TOTAL_CHECKPOINTS checkpoints to process"
echo "Will process in batches of $BATCH_SIZE, starting from lowest checkpoint number"

for ((i = 0; i < ${#CHECKPOINT_NUMBERS[@]}; i += BATCH_SIZE)); do
    batch=("${CHECKPOINT_NUMBERS[@]:i:BATCH_SIZE}")
    echo "Starting batch $((i/BATCH_SIZE + 1)) of $((TOTAL_CHECKPOINTS/BATCH_SIZE + 1))"
    process_batch "${batch[@]}"
done
echo "All checkpoints processed successfully"
