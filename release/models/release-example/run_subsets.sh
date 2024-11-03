export BRANCHES=$1
export HF_REPO=$2

#❗Add any extra args required for adding the checkpoint (eg. path to model file).
# export MODEL_BASE=$3

cat $BRANCHES | while read branch; do
  echo "$branch"

  if grep -q "$branch" updated_branches.txt
    then
      echo "$branch has already been updated"
    else

      #./upload_checkpoint.sh $branch $HF_REPO $3

      ./update_checkpoint.sh $branch $HF_REPO

      echo $branch >> updated_branches.txt
    fi
done
