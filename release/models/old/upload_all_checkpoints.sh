export CHECKPOINT_FOLDER=$1
export HF_LOCAL=$2

export BASEPATH=$(readlink -f "$(dirname "$0")")
for CHECKPOINT_PATH in $(find $CHECKPOINT_FOLDER -name "*-unsharded"); do $BASEPATH/upload_checkpoint.sh $CHECKPOINT_PATH $HF_LOCAL; done
