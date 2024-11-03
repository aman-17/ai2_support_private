
## Release PALOMA

* S3 location: s3://ai2-llm/eval-data/perplexity/v3_not_deconned/
* DO NOT upload s3://ai2-llm/eval-data/perplexity/v3_not_deconned/ice_fixed/.
* DO NOT upload s3://ai2-llm/eval-data/perplexity/v3_not_deconned/pile/.


* Upload location: Huggingface olmo-friends repository

## Process

```
conda create -n release-paloma python=3.10
conda activate release-paloma

pip install -r requirements.txt
```

### Dataset upload

```
huggingface-cli login
git lfs install
git clone https://huggingface.co/datasets/olmo-friends/paloma
cp -r <data-files> .
git add .
git commit -m "add data"
git push
```
