# 1B Baselines from Paloma

## Unsharding
most of the model checkpoints are already unsharded by the training process, but for some reason some are missed. To unshard these we have to use [this script at this specific commit](https://github.com/allenai/LLM/blob/db0756ffa97af3bce38edfa701bfbcd272e94ca2/scripts/unshard.py). I don't understand why but using the newer version of this script just crashes.

## Updating the tokenizer in the config
The model training uses a local copy of the tokenizer to speed things up, but we want to use the identical copy on HF hub now. So we run [this script](scripts/fix_olmo_checkpoints_tokenizer.py) on the model checkpoints.
