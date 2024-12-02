#!/bin/bash

SOURCE_DIR="/data/input/ai2-llm/checkpoints/OLMo-medium/peteish7"
DEST_DIR="/data/input/amanr/safetens_unshard2"

mkdir -p "$DEST_DIR"

steps=($(ls "$SOURCE_DIR" | grep -Eo 'step[0-9]+-unsharded' | grep -Eo '[0-9]+' | sort -n | awk '$1 >= 248000 && $1 <= 2480000'))
total=${#steps[@]}

echo "Found $total unsharded checkpoints to move"
echo "Moving checkpoints..."

for ((i=0; i<${#steps[@]}; i++)); do
    step=${steps[i]}
    current=$((i + 1))
    percentage=$((current * 100 / total))

    printf "\rProgress: [%-50s] %d%% (%d/%d)" \
        $(printf "#%.0s" $(seq 1 $((percentage/2)))) \
        $percentage $current $total

    mv "$SOURCE_DIR/step${step}-unsharded" "$DEST_DIR/"
done

echo -e "\nMove completed!"
