from hf_olmo import *
# Instead of importing OlmoNewForCausalLM, import your unified class
from olmo_backward_compatible import OlmoForCausalLM  # Your unified implementation
from vllm.model_executor.utils import set_random_seed
from vllm import ModelRegistry
from vllm import LLM, SamplingParams

# Register your unified model
ModelRegistry.register_model("OlmoForCausalLM", OlmoForCausalLM)

checkpoint_path = "model-checkpoints/peteish7/step11931-unsharded-hf/"

# Create LLM instance - it will use your unified implementation
sampling_params = SamplingParams(temperature=0.0)
llm = LLM(
    model=checkpoint_path,
    trust_remote_code=True,
    gpu_memory_utilization=0.90
)

set_random_seed(0)

# Test generation
prompt = "San Francisco is a"
outputs = llm.generate([prompt], sampling_params=sampling_params)
generated_text = outputs[0].outputs[0].text
print(f"Generated: {generated_text}")

# Cleanup
import torch.distributed as dist
if dist.is_initialized():
    dist.destroy_process_group()