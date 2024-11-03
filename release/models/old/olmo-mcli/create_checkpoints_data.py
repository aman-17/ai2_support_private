import sys
import os
import glob
import torch
import pandas as pd
import tqdm
import concurrent.futures
import torch
from cached_path import cached_path
from torch.distributed._shard.sharded_tensor import ShardedTensor

# Monkeypatch torch's ShardedTensor, so we can unpickle without having torch.distributed set up.
def _rebuild_from_type_v2_monkey(func, new_type, args, state):
    ret = func(*args)
    if type(ret) is not new_type:
        ret = ret.as_subclass(new_type)

    # Shortcut the construction of ShardedTensor
    # This is in the top 5 of my worst hacks.
    if isinstance(ret, ShardedTensor):
        ret._local_shards, ret._metadata, _, ret._sharding_spec, ret._init_rrefs = state
        return ret

    # The rest of this function ought to be in the top 5 of somebody else's worst hacks.
    # Tensor does define __setstate__ even though it doesn't define
    # __getstate__. So only use __setstate__ if it is NOT the one defined
    # on Tensor
    if getattr(ret.__class__, "__setstate__", torch.Tensor.__setstate__) is not torch.Tensor.__setstate__:
        ret.__setstate__(state)
    else:
        ret = torch._utils._set_obj_state(ret, state)
    return ret


torch._tensor._rebuild_from_type_v2 = _rebuild_from_type_v2_monkey

def get_info(checkpoints_path, i):
    path = checkpoints_path + f"step{i}"
    rank0_path = path + "/rank0.pt"
    num_tokens = torch.load(cached_path(rank0_path), map_location="cpu")["global_train_tokens_seen"]

    return {
        "num_step": i,
        "num_tokens": num_tokens,
        "path": path
    }

def checkpoints_location(checkpoints_path: str):
    
    """
    all_checkpoints = {}
    for i in tqdm.tqdm(range(0, 557000, 1000)):
        path = checkpoints_path + f"step{i}"
        rank0_path = path + "/rank0.pt"
       
        num_tokens = torch.load(cached_path(rank0_path), map_location="cpu")["global_train_tokens_seen"]

        all_checkpoints[i] = {
            "num_step": i,
            "num_tokens": num_tokens,
            "path": path
        }
    """

    with concurrent.futures.ThreadPoolExecutor(
        thread_name_prefix="get_info-"
    ) as executor:
        all_checkpoints = {}
        all_steps = []
        for i in range(0, 557000, 1000):
            all_steps.append(executor.submit(get_info, checkpoints_path, i))
        for future in concurrent.futures.as_completed(all_steps):
            step_info = future.result()
            all_checkpoints[step_info["num_step"]] = step_info

    cdf = pd.DataFrame(list(all_checkpoints.values()))
    cdf.sort_values(by="num_step", inplace=True, ascending=False)

    # validation
    assert cdf[cdf["num_tokens"]>0]["num_tokens"].is_monotonic_decreasing

    cdf.to_csv("olmo_7b_mcli_checkpoints.csv", index=False)

if __name__ == "__main__":
    checkpoints_location(sys.argv[1])
