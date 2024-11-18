from huggingface_hub import delete_branch, create_tag

# delete_branch("allenai/olmo-peteish7", repo_type="model", branch="main")
delete_branch("allenai/olmo-peteish7", repo_type="model", branch="step249K-tokens-1044.38B")
delete_branch("allenai/olmo-peteish7", repo_type="model", branch="step250K-tokens-1048.57B")
delete_branch("allenai/olmo-peteish7", repo_type="model", branch="step251K-tokens-1052.77B")
delete_branch("allenai/olmo-peteish7", repo_type="model", branch="step252K-tokens-1056.96B")
delete_branch("allenai/olmo-peteish7", repo_type="model", branch="step253K-tokens-1061.15B")
delete_branch("allenai/olmo-peteish7", repo_type="model", branch="step254K-tokens-1065.35B")


# create_tag("bigcode/the-stack", repo_type="dataset", revision="v0.1-release", tag="v0.1.1", tag_message="Bump release version.")