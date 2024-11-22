#!/bin/bash

set -euxo pipefail

mkdir unshard_and_upload
cd unshard_and_upload/
git clone https://github.com/allenai/OLMo.git
pip install -e ./OLMo/
pip install torch
# pip install -e /myfiles/amanr/transformers/
# git clone https://github.com/huggingface/transformers.git
# git clone -b shanea/add-olmo1124 https://github.com/2015aroras/transformers.git
git clone https://github.com/aman-17/transformers.git
pip install -e ./transformers/
pip install 'accelerate>=0.26.0'
pip install hf_transfer
pip install datasets wandb
wget https://huggingface.co/allenai/dolma2-tokenizer/resolve/main/tokenizer.json

checkpoint=$1
OLMO_DIR="./OLMo"
GCS_CHECKPOINT_DIR="gs://ai2-llm/checkpoints/OLMo-medium/peteish13-highlr"
LOCAL_CHECKPOINT_DIR="/data/unshard_and_upload/peteish13-highlrr/temp" #downloading gs unsharded here
DEST_DIR="/data/unshard_and_upload/peteish13-highlrr"
LOGS_DIR="/data/unshard_and_upload/peteish13-highlrr/logs"
SERVICE_ACCOUNT_JSON="/new_mount_path_1/service_account.json"
HF_OUTPUT_BASE="/data/unshard_and_upload/huggingface_peteish13_output"
TOKENIZER="./tokenizer.json"
HF_BATCH_SIZE=2048
SEQ_LENGTH=4096
HF_REPO_ID="allenai/peteish13b"

mkdir -p "$LOCAL_CHECKPOINT_DIR" "$DEST_DIR" "$LOGS_DIR" "$HF_OUTPUT_BASE"

if [ ! -f "$SERVICE_ACCOUNT_JSON" ]; then
    echo "Error: Service account JSON file not found at $SERVICE_ACCOUNT_JSON"
    exit 1
fi

export GOOGLE_APPLICATION_CREDENTIALS="$SERVICE_ACCOUNT_JSON"
gcloud auth activate-service-account --key-file="$SERVICE_ACCOUNT_JSON"
export HUGGING_FACE_HUB_TOKEN="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"
huggingface-cli login --token $HUGGING_FACE_HUB_TOKEN

echo "Downloading step${checkpoint}"
gsutil -m rsync -r "${GCS_CHECKPOINT_DIR}/step${checkpoint}" "$LOCAL_CHECKPOINT_DIR/"

# 1.c Unshard checkpoint
echo "Unsharding step${checkpoint}"
python "$OLMO_DIR/scripts/storage_cleaner.py" unshard \
   "$LOCAL_CHECKPOINT_DIR" "$DEST_DIR" \
   --checkpoint_num "$checkpoint" \
   > "$LOGS_DIR/step${checkpoint}-unshard.log" 2>&1

cat "$LOGS_DIR/step${checkpoint}-unshard.log"

# 2. Preparing HF conversion
total_tokens=$((checkpoint * HF_BATCH_SIZE * SEQ_LENGTH))
tokens_b=$(echo "scale=0; ($total_tokens+999999999) / 1000000000" | bc)
hf_output_dir="${HF_OUTPUT_BASE}/step${checkpoint}-hf"
revision="step${checkpoint}-tokens${tokens_b}B"

# 3. HF formatting and uploading
echo "Converting to HF format and uploading (${tokens_b}B tokens)"
mkdir -p "$hf_output_dir"
HF_HUB_ENABLE_HF_TRANSFER=1 python ./transformers/src/transformers/models/olmo_1124/convert_olmo_1124_weights_to_hf.py \
    --input_dir "${DEST_DIR}/step${checkpoint}-unsharded" \
    --output_dir "$hf_output_dir" \
    --hf_repo_id "allenai/peteish13b" \
    --hf_repo_revision "$revision" \
    --tokenizer_json_path "$TOKENIZER"

# 3. Cleaning after HF upload
rm -rf "${LOCAL_CHECKPOINT_DIR}/step${checkpoint}"
rm -rf "${DEST_DIR}/step${checkpoint}-unsharded"
rm -rf "$hf_output_dir"
echo "Completed step${checkpoint} ($((i+1))/$total)"
echo "----------------------------------------"

echo "All checkpoints processed successfully"
