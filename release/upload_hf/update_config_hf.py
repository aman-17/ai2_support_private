from huggingface_hub import HfApi
import json
import os
import shutil

def update_config(token, repo_id, branch):
    api = HfApi(token=token)
    temp_dir = f"temp_{branch}"
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    config_path = api.hf_hub_download(
        repo_id=repo_id, 
        filename="config.json", 
        revision=branch,
        local_dir=temp_dir
    )
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    config["architectures"] = ["Olmo2ForCausalLM"]
    config["model_type"] = "olmo2"
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    api.upload_file(
        path_or_fileobj=config_path,
        path_in_repo="config.json",
        repo_id=repo_id,
        revision=branch,
        commit_message="Update model architecture to Olmo2"
    )
    
    shutil.rmtree(temp_dir)

def update_all_branches(token, repo_id):
    branches = [
        "main", "step1000-tokens5B"]#, "step10000-tokens42B", "step101000-tokens424B"]

    
    for branch in branches:
        try:
            update_config(token, repo_id, branch)
            print(f"Updated {branch}")
        except Exception as e:
            print(f"Error updating {branch}: {e}")

if __name__ == "__main__":
    HF_TOKEN = "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"
    REPO_ID = "allenai/olmo-peteish7"
    update_all_branches(HF_TOKEN, REPO_ID)
