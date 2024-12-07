#!/bin/bash

pip install torch
pip install -e /myfiles/amanr/transformers/
pip install 'accelerate>=0.26.0'
pip install hf_transfer
export HUGGING_FACE_HUB_TOKEN="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"
huggingface-cli login --token $HUGGING_FACE_HUB_TOKEN

BASE_DIR="/myfiles/ai2-llm/checkpoints/OLMo-medium/peteish7"
OUTPUT_BASE="/myfiles/amanr/huggingface_peteish7"
TOKENIZER="/myfiles/amanr/tokenizer.json"

BATCH_SIZE=1024
SEQ_LENGTH=4096

for checkpoint in $(ls -d ${BASE_DIR}/step*-unsharded | sort -V); do
    step=$(echo $checkpoint | grep -o 'step[0-9]*' | grep -o '[0-9]*')

    if [ $step -ge 249000 ] && [ $step -le 928646 ]; then
        total_tokens=$((step * BATCH_SIZE * SEQ_LENGTH))
        tokens_b=$(echo "scale=0; ($total_tokens+999999999) / 1000000000" | bc)

        OUTPUT_DIR="${OUTPUT_BASE}/step${step}-hf"
        mkdir -p "$OUTPUT_DIR"

        revision="stage-2-step${step}-tokens${tokens_b}B"

        echo "Processing step ${step}, total tokens: ${tokens_b}B"

        HF_HUB_ENABLE_HF_TRANSFER=1 python /myfiles/amanr/transformers/src/transformers/models/olmo_1124/convert_olmo_1124_weights_to_hf.py \
            --input_dir "$checkpoint" \
            --output_dir "$OUTPUT_DIR" \
            --hf_repo_id "allenai/olmo-peteish7" \
            --hf_repo_revision "$revision" \
            --tokenizer_json_path "$TOKENIZER"
    fi
done