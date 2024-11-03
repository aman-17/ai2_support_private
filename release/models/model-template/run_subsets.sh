export BRANCHES=$1
export HF_REPO=$2

#❗Add any extra args required for adding the checkpoint (eg. path to model file).
# export MODEL_BASE=$3

export UPDATE_TRACKER=updated_$BRANCHES

cat $BRANCHES | while read branch; do
  echo "$branch"

  if grep -q "$branch" $UPDATE_TRACKER
    then
      echo "$branch has already been updated. If you think this is a mistake, modify the tracking file: $UPDATE_TRACKER"
    else

      #./upload_checkpoint.sh $branch $HF_REPO $3

      ./update_checkpoint.sh $branch $HF_REPO

      echo $branch >> $UPDATE_TRACKER
    fi
done
