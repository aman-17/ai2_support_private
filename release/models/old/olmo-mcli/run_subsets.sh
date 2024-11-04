export NUMBER=$1

cat step-token-$NUMBER.txt | while read step; read tokens; do
  echo "step${step}-tokens${tokens}B"

  if grep -q "step${step}-tokens${tokens}B" current_branches.txt
    then
      echo "present"
    else
      ./upload.sh /mnt/disks/ckpt/checkpoints/v1_5-mix-mitch-ish/step${step}-unsharded step${step}-tokens${tokens}B /mnt/disks/ckpt/akshitab/hf/step-token-$NUMBER/olmo-7b-pine/
    fi
done
