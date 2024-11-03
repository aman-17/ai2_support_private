import sys
import os
import glob
import torch
import pandas as pd
import tqdm
import concurrent.futures
import torch


def get_info(checkpoints_path, i):
    path = os.path.join(checkpoints_path, f"step{i}-unsharded")
    train_path = path + "/train.pt"
    num_tokens = torch.load(train_path, map_location="cpu")["global_train_tokens_seen"]

    return {
        "num_step": i,
        "num_tokens": num_tokens,
    }

def checkpoints_location(checkpoints_path: str):
    
    with concurrent.futures.ThreadPoolExecutor(
        thread_name_prefix="get_info-"
    ) as executor:
        all_checkpoints = {}
        all_steps = []
        for p in glob.glob(f"{checkpoints_path}/*-unsharded"):
            i = int(p.split("/")[-1].replace("step", "").replace("-unsharded", ""))
            all_steps.append(executor.submit(get_info, checkpoints_path, i))
        for future in concurrent.futures.as_completed(all_steps):
            step_info = future.result()
            all_checkpoints[step_info["num_step"]] = step_info

    cdf = pd.DataFrame(list(all_checkpoints.values()))
    cdf.sort_values(by="num_step", inplace=True, ascending=False)

    # validation
    assert cdf[cdf["num_tokens"]>0]["num_tokens"].is_monotonic_decreasing

    cdf.to_csv("olmo_1b_checkpoints.csv", index=False)

if __name__ == "__main__":
    checkpoints_location(sys.argv[1])
