HF_LOCAL=$1
FILES_TO_UPDATE=$2
HF_GIT_FOLDER=~/models/hf

export BASEPATH=$(readlink -f "$(dirname "$0")")
echo $BASEPATH

cd $HF_LOCAL
BRANCHES=($(git branch -r --list))
for item in "${BRANCHES[@]}"; 
do 
    if [[ $item == *"HEAD"* ]] || [[ $item == *"->"* ]]; then
	ignore="Ignoring"
    else
        branch_name="${item#origin/}"
	echo "Branch:" $branch_name
	git checkout $branch_name
	cp $BASEPATH/$FILES_TO_UPDATE/* $HF_LOCAL
	git add .
	git commit -m "update files"
	git push
    fi
done 
