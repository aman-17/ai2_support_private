#!/bin/bash

BASE_DIR="/weka/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish7"
OUTPUT_BASE="/weka/oe-training-default/amanr/huggingface_peteish7"
TOKENIZER="/weka/oe-training-default/amanr/tokenizer.json"

for checkpoint in $(ls -d ${BASE_DIR}/step*-unsharded | sort -V); do
    step=$(echo $checkpoint | grep -o 'step[0-9]*' | grep -o '[0-9]*')

    if [ $step -ge 249000 ] && [ $step -le 928646 ]; then
        OUTPUT_DIR="${OUTPUT_BASE}/step${step}-hf"
        mkdir -p "$OUTPUT_DIR"

        echo "Processing step ${step}..."
        
        python src/transformers/models/olmo_1124/convert_olmo_1124_weights_to_hf.py \
            --input_dir "$checkpoint" \
            --output_dir "$OUTPUT_DIR" \
            --hf_repo_id "allenai/olmo-peteish7" \
            --hf_repo_revision "checkpoint-step${step}" \
            --tokenizer_json_path "$TOKENIZER"
    fi
done