
# format: stepXXX-tokensYYY
export CHECKPOINT_NAME=$1

export HF_REPO=$2

echo "Checkpoint: $CHECKPOINT_NAME"
cd $HF_REPO

echo "Checkout branch: $CHECKPOINT_NAME"

GIT_LFS_SKIP_SMUDGE=1 git checkout main

GIT_LFS_SKIP_SMUDGE=1 git checkout $CHECKPOINT_NAME

# ❗If your update requires use of the model files,
# then uncomment line below.
# git lfs pull  # uncomment if needed

# ❗Add your updates here

# Example 1: Update config file

# cp $BASEPATH/$MODEL/updated_config/config.json .

# Example 2: Run a conversion script to add safetensors format.
# if test -f model.safetensors;
# then
#     echo "safetensors file already present"
# else
#     git lfs pull
#     echo "adding safetensors"
#     python $BASEPATH/common/convert_to_safetensors.py pytorch_model.bin model.safetensors
# fi

echo "Git add"
git add .

echo "Git commit"
git commit -m "update $CHECKPOINT_NAME"

echo "Git push"
git push
