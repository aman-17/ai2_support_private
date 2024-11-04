export BRANCHES=$1
export HF_REPO=$2

cat $BRANCHES | while read branch; do
  echo "$branch"

  if grep -q "$branch" updated_$BRANCHES
    then
      echo "$branch has already been updated"
    else
      ./update_checkpoint.sh $branch $HF_REPO
      echo $branch >> updated_$BRANCHES
    fi
done
