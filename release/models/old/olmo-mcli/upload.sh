
export CHECKPOINT_PATH=$1

# format: stepXXX-tokensYYY
export CHECKPOINT_NAME=$2

export HF_REPO=$3

#export BASEPATH=$(readlink -f "$(dirname "$0")")
export BASEPATH=~/olmo-release-processes/models

echo "Checkpoint: $CHECKPOINT_NAME"
cd $HF_REPO

echo "Checkout new branch: $CHECKPOINT_NAME"
git checkout main
git checkout -b $CHECKPOINT_NAME
git checkout $CHECKPOINT_NAME

echo "New model checkpoint ..."
cp $CHECKPOINT_PATH/config.yaml .
cp $CHECKPOINT_PATH/model.pt .

python $BASEPATH/convert_to_hf.py --checkpoint-dir .

echo "Git add"
git add .

echo "Git commit"
git commit -m "add $CHECKPOINT_NAME"

echo "Git push"
git push --set-upstream origin $CHECKPOINT_NAME
