
# format: stepXXX-tokensYYY
export CHECKPOINT_NAME=$1

export HF_REPO=$2

#export BASEPATH=$(readlink -f "$(dirname "$0")")
export BASEPATH=~/olmo-release-processes/models

echo "Checkpoint: $CHECKPOINT_NAME"
cd $HF_REPO

echo "Checkout branch: $CHECKPOINT_NAME"
GIT_LFS_SKIP_SMUDGE=1 git checkout main
GIT_LFS_SKIP_SMUDGE=1 git checkout $CHECKPOINT_NAME

git lfs pull

if test -f model.safetensors;
then
    echo "already present"
else
    echo "adding safetensors"
    rm README.md
    cp $BASEPATH/olmo-mcli/updated/config.json .
    python $BASEPATH/convert_to_safetensors.py pytorch_model.bin model.safetensors
fi

echo "Git add"
git add .

echo "Git commit"
git commit -m "add safetensors $CHECKPOINT_NAME"

echo "Git push"
git push
