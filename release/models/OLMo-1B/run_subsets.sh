export CHECKPOINTS_PATH=$1
export NUMBER=$2
cat step-token-${NUMBER}.txt | while read step; read tokens; do
  echo "step${step}-tokens${tokens}B"

  if grep -q "step${step}-tokens${tokens}B" current_branches.txt
    then
      echo "present"
    else
      ./upload.sh $CHECKPOINTS_PATH/step${step}-unsharded step${step}-tokens${tokens}B ~/hf/step-token-${NUMBER}/OLMo-1B
    fi
done
