#cat test-do-step-tokens.txt| parallel -j 1 --bar --max-args=2 ./upload.sh /mnt/disks/ckpt/checkpoints/v1_5-mix-mitch-ish/step{1}-unsharded step{1}-tokens{2}B /mnt/disks/ckpt/akshitab/hf/olmo-7b-pine/

cat do-step-token.txt | while read step; read tokens; do
  echo "step${step}-tokens${tokens}B"
  ./upload.sh /mnt/disks/ckpt/checkpoints/v1_5-mix-mitch-ish/step${step}-unsharded step${step}-tokens${tokens}B /mnt/disks/ckpt/akshitab/hf/olmo-7b-pine/
done
