
# Upload a new model on huggingface

## Step 0: Initial setup

```
conda create -n release-<model> python=3.10
conda activate release-<model>
pip install -r requirements.txt
```

## Step 1: Create repository

Create private repository for the model in the huggingface organization `allenai`.

* `main` branch will contain our final model.
* Intermediate checkpoints will be stored in branches, eg. `step1000-tokens4B`.

## Step 2: Environment

```
export HF_BASE=~/hf  # this is the folder where the huggingface repository will be cloned.
export BASEPATH=$(pwd) # this repository
huggingface-cli login
```

### Setup HF repo
```
cd $HF_BASE
git lfs install
git clone https://huggingface.co/allenai/<model>
```

### Add common files

Add all the non-model files (i.e., requirements.txt, python files, config files, tokenizer). These are files that are expected to be common across all the checkpoints of the model.
Do not include the large model files. Do not include README.md

```
cp hf_files/* $HF_BASE/<model>
cd $HF_BASE/<model>
git add .
git commit -m "add common files"
git push
```

## Step 3: Add a new folder in THIS repo for the model

This is to track any model-specific uploading scripts, for every model that we upload to huggingface.

```
cd $BASEPATH
cp model-template <model>
```

And add the names of all checkpoints to a text file. These names will be used as branch names.

```
ls $BASEPATH/<model>/revisions.txt
```


The `model-template` folder contains 3 files that **should be updated per model**.

1. `run_subsets.sh` : This is the file that iterates over all the checkpoint names, and uploads/updates the corresponding models to branches of the same name.

2. `upload_checkpoint.sh` : This file uploads a single checkpoint. This includes creating a new branch, adding relevant model files, and pushing it to huggingface. 

3. `update_checkpoint.sh` : Use this in `run_subsets.sh` you need to update already uploaded checkpoints (**try not to do this**).


Finally,

```
cd $BASEPATH/<model>/
./run_subsets.sh revisions.txt $HF_BASE/<model> <any additional args>
```

Note: [common](common) folder contains some scripts that you may find useful for processing your models before uploading, which you can specify in the upload/update scripts. See [release-example](release-example) for an example use of the `model-template`, including model-specific changes to the config file.


## Step 4: Update final model

Update your final model to the `main` once all your checkpoints have been uploaded.
Also add README.md


#### Hacks

1. If you need to upload a large number of checkpoints, you can parallelize your efforts by creating multiple clones of the model repository, and running the above process on multiple batches of checkpoints simultaneously. See `common/split_revisions.py` to split your checkpoints into batches.

2. Git push issues
https://discuss.huggingface.co/t/cant-push-to-new-space/35319/3
