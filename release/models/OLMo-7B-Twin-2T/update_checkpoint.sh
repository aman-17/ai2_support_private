
# format: stepXXX-tokensYYY
export CHECKPOINT_NAME=$1

export HF_REPO=$2

export BASEPATH=~/olmo-release-processes/models

echo "Checkpoint: $CHECKPOINT_NAME"
cd $HF_REPO

echo "Checkout branch: $CHECKPOINT_NAME"

GIT_LFS_SKIP_SMUDGE=1 git checkout main

GIT_LFS_SKIP_SMUDGE=1 git checkout $CHECKPOINT_NAME

rm README.md # remove unnecessary readme in branch.
cp $BASEPATH/OLMo-7B-Twin-2T/updated_config/config.json .

if test -f model.safetensors;
then
    echo "safetensors file already present"
else
    git lfs pull
    echo "adding safetensors"
    python $BASEPATH/common/convert_to_safetensors.py pytorch_model.bin model.safetensors
fi

echo "Git add"
git add .

echo "Git commit"
git commit -m "add safetensors $CHECKPOINT_NAME"

echo "Git push"
git push
