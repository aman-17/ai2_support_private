
1. Run interactive high-memory session on LUMI:

cd olmo-release-processes/models/olmo-lumi
srun --account=$PROJECT --partition=largemem --time=04:00:00 --nodes=1 --ntasks-per-node=8 --gpus-per-node=0 --pty ./lumi-interactive.sh
