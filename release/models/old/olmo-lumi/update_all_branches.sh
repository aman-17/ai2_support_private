export HF_REPO=$1
export FILES_TO_UPDATE=$2
export BASEPATH=~/olmo-release-processes/models
echo $BASEPATH

cd $HF_REPO
cat $BASEPATH/olmo-lumi/branch_names.txt | while read branch_name; do
    echo "Branch:" $branch_name
    GIT_LFS_SKIP_SMUDGE=1 git checkout main
    GIT_LFS_SKIP_SMUDGE=1 git checkout $branch_name
    cp $BASEPATH/olmo-lumi/$FILES_TO_UPDATE/* $HF_REPO
    git add .
    git commit -m "update tokenizer"
    git push
done 
