#!/bin/bash
pip install torch
pip install -e /myfiles/amanr/transformers/
pip install 'accelerate>=0.26.0'
pip install hf_transfer
pip install ai2-olmo datasets wandb

OLMO_DIR="/myfiles/amanr/OLMo"
GCS_CHECKPOINT_DIR="gs://ai2-llm/checkpoints/OLMo-medium/peteish13-highlr"
LOCAL_CHECKPOINT_DIR="/myfiles/amanr/peteish13-highlrr/temp" #downloading gs unsharded here
DEST_DIR="/myfiles/amanr/peteish13-highlrr"
LOGS_DIR="/myfiles/amanr/peteish13-highlrr/logs"
SERVICE_ACCOUNT_JSON="/myfiles/amanr/service_account.json"
HF_OUTPUT_BASE="/myfiles/amanr/huggingface_peteish13_output"
TOKENIZER="/myfiles/amanr/tokenizer.json"
HF_BATCH_SIZE=2048
SEQ_LENGTH=4096
HF_REPO_ID="allenai/peteish13b"

mkdir -p "$LOCAL_CHECKPOINT_DIR" "$LOGS_DIR" "$DEST_DIR" "$HF_OUTPUT_BASE"

if [ ! -f "$SERVICE_ACCOUNT_JSON" ]; then
    echo "Error: Service account JSON file not found at $SERVICE_ACCOUNT_JSON"
    exit 1
fi

export GOOGLE_APPLICATION_CREDENTIALS="$SERVICE_ACCOUNT_JSON"
gcloud auth activate-service-account --key-file="$SERVICE_ACCOUNT_JSON"
export HUGGING_FACE_HUB_TOKEN="HF_TOKEN"
huggingface-cli login --token $HUGGING_FACE_HUB_TOKEN

checkpoints=($(gsutil ls "$GCS_CHECKPOINT_DIR" | \
    grep -Eo 'step[0-9]+' | \
    sed 's/step//' | \
    sort -n | \
    awk '$1 >= 100500 && $1 <= 103500'))

total=${#checkpoints[@]}
echo "Found $total checkpoints to process"

for ((i=0; i<${#checkpoints[@]}; i++)); do
    checkpoint=${checkpoints[i]}
    echo "Processing checkpoint $((i+1))/$total: step${checkpoint}"

    # Check if unsharded directory already exists
    if [ -d "${DEST_DIR}/step${checkpoint}-unsharded" ]; then
        echo "Unsharded checkpoint already exists for step${checkpoint}, skipping download and unshard"
    else
        # Check if sharded checkpoint exists
        if [ -d "${LOCAL_CHECKPOINT_DIR}/step${checkpoint}" ]; then
            echo "Sharded checkpoint already exists for step${checkpoint}, skipping download"
        else
            # Download checkpoint
            echo "Downloading step${checkpoint}"
            gsutil -m cp -r "${GCS_CHECKPOINT_DIR}/step${checkpoint}" "$LOCAL_CHECKPOINT_DIR/"
        fi

        # Unshard checkpoint
        echo "Unsharding step${checkpoint}"
        python "$OLMO_DIR/scripts/storage_cleaner.py" unshard \
            "$LOCAL_CHECKPOINT_DIR" "$DEST_DIR" \
            --checkpoint_num "$checkpoint" \
            > "$LOGS_DIR/step${checkpoint}-unshard.log" 2>&1
    fi

    # 3. HF conversion
    total_tokens=$((checkpoint * HF_BATCH_SIZE * SEQ_LENGTH))
    tokens_b=$(echo "scale=0; ($total_tokens+999999999) / 1000000000" | bc)
    hf_output_dir="${HF_OUTPUT_BASE}/step${checkpoint}-hf"
    revision="step${checkpoint}-tokens${tokens_b}B"

    # 4. Convert to HF format and upload
    echo "Converting to HF format and uploading (${tokens_b}B tokens)"
    mkdir -p "$hf_output_dir"
    HF_HUB_ENABLE_HF_TRANSFER=1 python /myfiles/amanr/transformers/src/transformers/models/olmo_1124/convert_olmo_1124_weights_to_hf.py \
        --input_dir "${DEST_DIR}/step${checkpoint}-unsharded" \
        --output_dir "$hf_output_dir" \
        --hf_repo_id "$HF_REPO_ID" \
        --hf_repo_revision "$revision" \
        --tokenizer_json_path "$TOKENIZER"

    # 5. Cleanup
    rm -rf "${LOCAL_CHECKPOINT_DIR}/step${checkpoint}"
    rm -rf "${DEST_DIR}/step${checkpoint}-unsharded"
    rm -rf "$hf_output_dir"
    echo "Completed step${checkpoint} ($((i+1))/$total)"
    echo "----------------------------------------"
done

echo "All checkpoints processed successfully"