from hf_olmo import *  # For peteish config
from transformers import AutoModelForCausalLM
from olmo_new import OlmoNewForCausalLM  # The new implementation we'll use
from vllm.model_executor.models import ModelRegistry
from vllm import LLM, SamplingParams

# Path to your peteish checkpoint
checkpoint_path = "model-checkpoints/peteish7/step11931-unsharded-hf/"

# Load the model using HuggingFace
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    trust_remote_code=True,
)

# Register the new OLMo implementation with vLLM
ModelRegistry.register_model("OLMoForCausalLM", OlmoNewForCausalLM)


# Create sampling parameters (here using greedy sampling)
sampling_params = SamplingParams(temperature=0.0)

# Initialize vLLM with the model
llm = LLM(
    model=checkpoint_path, 
    trust_remote_code=True, 
    gpu_memory_utilization=0.90
)

# Example generation
prompt = "San Francisco is a"
outputs = llm.generate([prompt], sampling_params=sampling_params)
generated_text = outputs[0].outputs[0].text
print(f"Generated: {generated_text}")