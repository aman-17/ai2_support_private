export NUMBER=$1

cat step-token-$NUMBER.txt | while read step; read tokens; do
  echo "step${step}-tokens${tokens}B"

  if grep -q "step${step}-tokens${tokens}B" current_branches.txt
    then
      echo "present"
    else
      ./upload.sh ~/lumi_checkpoints/step${step}-unsharded step${step}-tokens${tokens}B ~/hf/step-token-$NUMBER/olmo-7b-maple/
    fi
done
