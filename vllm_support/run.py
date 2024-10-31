from hf_olmo import *  # For peteish config
from olmo_new import OlmoNewForCausalLM  # The new implementation we'll use
# from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.utils import set_random_seed
from vllm import ModelRegistry
from vllm import LLM, SamplingParams

# Path to your peteish checkpoint
checkpoint_path = "model-checkpoints/peteish7/step11931-unsharded-hf/"

# Register the new OLMo implementation with vLLM
ModelRegistry.register_model("OlmoNewForCausalLM", OlmoNewForCausalLM)


# Create sampling parameters (here using greedy sampling)
sampling_params = SamplingParams(temperature=0.0)

# Initialize vLLM with the model
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

# {'vllm': '23 years old and I live in the United States. I am a student at'}