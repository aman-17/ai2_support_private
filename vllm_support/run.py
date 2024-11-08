from hf_olmo import *
from olmo_new import OlmoNewForCausalLM  # The new implementation we'll use
from vllm.model_executor.utils import set_random_seed
from vllm import ModelRegistry
from vllm import LLM, SamplingParams

checkpoint_path = "model-checkpoints/peteish7/step11931-unsharded-hf/"
ModelRegistry.register_model("OlmoNewForCausalLM", OlmoNewForCausalLM)
sampling_params = SamplingParams(temperature=0.0)
llm = LLM(
    model=checkpoint_path, 
    trust_remote_code=True, 
    gpu_memory_utilization=0.90
)
set_random_seed(0)
# Example generation
prompt = "San Francisco is a"
outputs = llm.generate([prompt], sampling_params=sampling_params)
generated_text = outputs[0].outputs[0].text
print(f"Generated: {generated_text}")

import torch.distributed as dist
if dist.is_initialized():
    dist.destroy_process_group()