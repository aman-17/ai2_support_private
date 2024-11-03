
export CHECKPOINT_PATH=$1
export HF_LOCAL=$2

export CHECKPOINT_NAME=$(basename $CHECKPOINT_PATH)

export BASEPATH=$(readlink -f "$(dirname "$0")")

echo "Checkpoint: $CHECKPOINT_NAME"
cd $HF_LOCAL

echo "Checkout new branch: $CHECKPOINT_NAME"
git checkout main
git checkout -b $CHECKPOINT_NAME

echo "Moving model checkpoint ..."
cp $CHECKPOINT_PATH/model.pt .
cp $CHECKPOINT_PATH/config.yaml .

echo "Adding HF configs ..."
python $BASEPATH/convert_to_hf.py --checkpoint-dir $HF_LOCAL

echo "Git add"
git add .

echo "Git commit"
git commit -m "add $CHECKPOINT_NAME"

echo "Git push"
git push --set-upstream origin $CHECKPOINT_NAME
