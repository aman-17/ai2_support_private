# olmo-release-processes
Scripts and notes for releasing data and models
```bash
pip3 install ai2-olmo datasets wandb 
```

## Unsharding:
```bash
export OLMO_DIR="/root/program/OLMo"
export LOCAL_CHECKPOINT_DIR="/weka/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish7"
export DEST_DIR="/weka/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish7"
export LOGS_DIR="/root/program/logs"
export TEMP_DIR="/weka/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish7/temp"
mkdir -p $LOGS_DIR
```

Number of checkpoints are in your directory:
```bash
ls $LOCAL_CHECKPOINT_DIR | grep -Eo 'step[0-9]+' | wc -l
```

```bash
ls $LOCAL_CHECKPOINT_DIR | \
    grep -Eo 'step[0-9]+' | \
    sed 's/step//' | \
    parallel -j 4 python $OLMO_DIR/scripts/storage_cleaner.py unshard $LOCAL_CHECKPOINT_DIR $DEST_DIR --checkpoint_num {} ">" $LOGS_DIR/step{}-unshard.log
```