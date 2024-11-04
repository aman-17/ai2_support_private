export S3_PATH=$1
export LOCAL_PATH=$2
export HF_MODEL_REPO=$3
export FILES_TO_INCLUDE=$4
export HF_GIT_FOLDER=~/models/hf

export HF_MODEL_NAME=$(basename $HF_MODEL_REPO)

aws s3 sync $S3_PATH $LOCAL_PATH --exclude "*" --include "*/config.yaml"
aws s3 sync $S3_PATH $LOCAL_PATH --exclude "*" --include "*/model.pt"

export LATEST_CHECKPOINT=$(ls -1v "$LOCAL_PATH" | tail -n 1)

echo "Latest checkpoint (added to main branch): $LATEST_CHECKPOINT"
export BASEPATH=$(readlink -f "$(dirname "$0")")
echo $BASEPATH

cd $HF_GIT_FOLDER
git clone https://huggingface.co/$HF_MODEL_REPO
huggingface-cli lfs-enable-largefiles $HF_MODEL_NAME

cd $HF_GIT_FOLDER/$HF_MODEL_NAME
git checkout main

cp $BASEPATH/hf_files/* $HF_GIT_FOLDER/$HF_MODEL_NAME

if [ $# -eq 4 ]
then
    cp $BASEPATH/$FILES_TO_INCLUDE/* $HF_GIT_FOLDER/$HF_MODEL_NAME
fi


echo "Copying checkpoint to hf repo locally"
cp $LOCAL_PATH/$LATEST_CHECKPOINT/model.pt $HF_GIT_FOLDER/$HF_MODEL_NAME
cp $LOCAL_PATH/$LATEST_CHECKPOINT/config.yaml $HF_GIT_FOLDER/$HF_MODEL_NAME

echo "Making checkpoint hf-compatible"
python $BASEPATH/convert_to_hf.py --checkpoint-dir $HF_GIT_FOLDER/$HF_MODEL_NAME
git add .
git commit -m "add model"
git push

$BASEPATH/upload_all_checkpoints.sh $LOCAL_PATH .
