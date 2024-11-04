
## Release 1b models

### Unshard checkpoint

Already done.

### Process

```
conda create -n release-models python=3.10
conda activate release-models

pip install -r requirements.txt
```

### Model upload

Create a private model repository in huggingface organization `olmo-friends`.

```
huggingface-cli login
export AWS_ACCESS_KEY_ID=<access_key_id>
export AWS_SECRET_ACCESS_KEY=<secret_access_key>
./upload_to_hf.sh <s3-model-path> <local-path> <hf-model-repo> <(optional) path-to-folder-containing-files-to-include>
```
