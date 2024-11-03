import glob
import torch
path = "/mnt/disks/ckpt/checkpoints/mitchish-lumi/step1000-unsharded/train.pt"
t = torch.load(path, map_location="cpu")
t
t.keys()
t["global_train_tokens_seen"]
t["global_train_tokens_seen"]/10**9
round(t["global_train_tokens_seen"]/10**9)
token_dict = {}
for path in glob.glob("/mnt/disks/ckpt/checkpoints/mitchish-lumi/step*-unsharded/train.pt"):
    t = torch.load(path, map_location="cpu")
    step = path
    break
import os
os.path.basename(path)
path
path.split("/")[-2].replace("-unsharded", "")
for path in glob.glob("/mnt/disks/ckpt/checkpoints/mitchish-lumi/step*-unsharded/train.pt"):
    t = torch.load(path, map_location="cpu")
    step = path.split("/")[-2].replace("-unsharded", "")
    token = round(t["global_train_tokens_seen"]/10**9)
    token_dict[step] = token
len(token_dict)
import pandas as pd
df = pd.DataFrame(token_dict)
df = pd.DataFrame.from_dict?
df = pd.DataFrame.from_dict(token_dict, columns=["num_step", "num_tokens"])
df = pd.DataFrame.from_dict(token_dict, columns=["num_step", "num_tokens"], orient="records")
df = pd.DataFrame.from_dict(token_dict, columns=["num_step", "num_tokens"], orient="index")
df = pd.DataFrame.from_dict(token_dict, columns=["num_step", "num_tokens"], orient="columns")
df = pd.DataFrame.from_dict(token_dict, columns=["num_tokens"], orient="index")
df
df.reindex()
df.reindex?
df.reset_index()
df.reset_index().rename(columns={"index": "num_step"})
df = df.reset_index().rename(columns={"index": "num_step"})
df.sort_values?
df.sort_values?
df.sort_values("num_step", axis=1)
df
df.sort_values("num_step", axis=0)
df.sort_values?
df.sort_values("num_step", axis=0, key=lambda x: int(x.replace("step", "")))
df.sort_values("num_step", axis=0, key=lambda x: x.replace("step", ""))
df
for path in glob.glob("/mnt/disks/ckpt/checkpoints/mitchish-lumi/step*-unsharded/train.pt"):
    t = torch.load(path, map_location="cpu")
    step = int(path.split("/")[-2].replace("-unsharded", "").replace("step", ""))
    token = round(t["global_train_tokens_seen"]/10**9)
    token_dict[step] = token
token_dict
df = pd.DataFrame.from_dict(token_dict, columns=["num_tokens"], orient="index")
df
df = df.reset_index().rename(columns={"index": "num_step"})
df
for path in glob.glob("/mnt/disks/ckpt/checkpoints/mitchish-lumi/step*-unsharded/train.pt"):
    t = torch.load(path, map_location="cpu")
    step = int(path.split("/")[-2].replace("-unsharded", "").replace("step", ""))
    token = round(t["global_train_tokens_seen"]/10**9)
    token_dict[step] = token
token_dict
df = pd.DataFrame.from_dict(token_dict, columns=["num_tokens"], orient="index")
df
token_dict
token_dict = {}
for path in glob.glob("/mnt/disks/ckpt/checkpoints/mitchish-lumi/step*-unsharded/train.pt"):
    t = torch.load(path, map_location="cpu")
    step = int(path.split("/")[-2].replace("-unsharded", "").replace("step", ""))
    token = round(t["global_train_tokens_seen"]/10**9)
    token_dict[step] = token
len(token_dict)
df = pd.DataFrame.from_dict(token_dict, columns=["num_tokens"], orient="index")
df = df.reset_index().rename(columns={"index": "num_step"})
df
df.sort_values("num_step", axis=0)
df.sort_values("num_step", axis=0, ascending=False)
df = df.sort_values("num_step", axis=0, ascending=False)
for i, row in df.iterrows():
    print(row)
    break
for i, row in df.iterrows():
    print(row.num_step)
    break
with open("olmo-lumi/step-token.txt", "w") as f:
    for i, row in df.iterrows():
        f.write(row.num_step+"\n)
with open("olmo-lumi/step-token.txt", "w") as f:
    for i, row in df.iterrows():
        f.write(row.num_step+"\n")
        f.write(row.num_tokens+"\n")
with open("olmo-lumi/step-token.txt", "w") as f:
    for i, row in df.iterrows():
        f.write(str(row.num_step)+"\n")
        f.write(str(row.num_tokens)+"\n")
with open("olmo-lumi/step-token.txt") as f"
with open("olmo-lumi/step-token.txt") as f:
    lines = f.readlines()
len(lines)
with open("olmo-lumi/step-token-1.txt", "w") as f:
    for line in lines[:200]:
        f.write(line+"\n")
with open("olmo-lumi/step-token-1.txt", "w") as f:
    for line in lines[:200]:
        f.write(line)
with open("olmo-lumi/step-token-2.txt", "w") as f:
    for line in lines[200:400]:
        f.write(line)
with open("olmo-lumi/step-token-3.txt", "w") as f:
    for line in lines[400:]:
        f.write(line)
