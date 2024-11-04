
export CHECKPOINT_PATH=$1

export CHECKPOINT_NAME=$(basename $CHECKPOINT_PATH)

#export BASEPATH=$(readlink -f "$(dirname "$0")")
export BASEPATH=~/olmo-release-processes/models

echo "Checkpoint: $CHECKPOINT_NAME"

echo "Adding HF configs ..."
python $BASEPATH/convert_to_hf.py --checkpoint-dir $CHECKPOINT_PATH

