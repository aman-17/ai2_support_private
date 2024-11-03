
# format: stepXXX-tokensYYY
export CHECKPOINT_NAME=$1
export HF_REPO=$2

#❗Add any extra args required for adding the checkpoint (eg. path to model file).
export MODEL_BASE=$3


#export BASEPATH=$(readlink -f "$(dirname "$0")")
#export BASEPATH=~/olmo-release-processes/models

echo "Checkpoint: $CHECKPOINT_NAME"
cd $HF_REPO

# This is IMPORTANT. If you don't checkout main, then each new branch will contain
# commits from the previous branch as well, which in this case are extremely large
# model files. The size of branch will linearly increase.
GIT_LFS_SKIP_SMUDGE=1 git checkout main

echo "Checkout new branch: $CHECKPOINT_NAME"

# Create new branch. If it already exists, the checkout DOES NOT change the branch.
git checkout -b $CHECKPOINT_NAME

# Extra safety step, in case the branch already exists.
git checkout $CHECKPOINT_NAME

echo "New model checkpoint ..."

#❗Add the checkpoints. Add any args required for adding the checkpoint (eg. path to model file).

# Example: Add model files
cp $MODEL_BASE/$CHECKPOINT_NAME/* .

echo "Git add"
git add .

echo "Git commit"
git commit -m "add $CHECKPOINT_NAME"

echo "Git push"
git push --set-upstream origin $CHECKPOINT_NAME
