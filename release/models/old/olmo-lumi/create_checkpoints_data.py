import sys
import os
import glob
import torch
import pandas as pd

def read_run_data(run_path: str, location_field: str = "Checkpoints Location"):
    df = pd.read_csv(run_path)
    
    missing_runs = []

    all_checkpoints = {}

    for path in list(df[location_field]):
        if path.startswith("s3:"):
            missing_runs.append(path)
            # only for local paths
            continue
        if not os.path.exists(path):
            print(f"{path} does not exist locally")
            missing_runs.append(path)

        print()

        print("Run: ", path)

        for chkpt in sorted(glob.glob(os.path.join(path, "step*"))):

            if "unsharded" in chkpt:
                continue
            train_info = torch.load(os.path.join(chkpt, "train", "rank0.pt"))
            if "global_train_tokens_seen" not in train_info:
                num_tokens = train_info["global_step"] * train_info["global_train_batch_size"]
            else:
                num_tokens = train_info["global_train_tokens_seen"]

            num_step = int(os.path.basename(chkpt).replace("step", ""))
            all_checkpoints[num_step] = {
                "num_step": num_step,
                "path": chkpt,
                "num_tokens": num_tokens
            }
            #print(f"{os.path.basename(chkpt)}-tokens{num_tokens}")

    #cdf = pd.DataFrame(all_checkpoints)
    #cdf.sort_values(by="num_step", inplace=True, ascending=False)

    all_checkpoints_list = []

    for i in range(0, 444000, 1000):
        if i not in all_checkpoints:
            all_checkpoints_list.append({
                "num_step": i,
                "path": "",
                "num_tokens": -1
            })
        else:
            all_checkpoints_list.append(all_checkpoints[i])

    cdf = pd.DataFrame(all_checkpoints_list)
    cdf.sort_values(by="num_step", inplace=True, ascending=False)

    # validation
    assert cdf[cdf["num_tokens"]>0]["num_tokens"].is_monotonic_decreasing

    cdf.to_csv("olmo_7b_lumi_checkpoints.csv", index=False)

if __name__ == "__main__":
    read_run_data(sys.argv[1])
