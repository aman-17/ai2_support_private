from hf_olmo import *  # For peteish config
from olmo_new import OlmoNewForCausalLM
from vllm import ModelRegistry, LLM, SamplingParams
import torch
from transformers import AutoModelForCausalLM
import os

def inspect_checkpoint(checkpoint_path):
    """Inspect the checkpoint contents"""
    print(f"\nInspecting checkpoint at: {checkpoint_path}")

    print("\nFiles in checkpoint directory:")
    for file in os.listdir(checkpoint_path):
        print(f"  - {file}")
    
    try:
        print("\nTrying to load with transformers...")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True
        )
        
        print("\nModel state dict keys:")
        for key in model.state_dict().keys():
            print(f"  - {key}")
            
    except Exception as e:
        print(f"Error loading with transformers: {str(e)}")
        
    try:
        pt_files = [f for f in os.listdir(checkpoint_path) if f.endswith(('.bin', '.pt'))]
        if pt_files:
            print(f"\nLoading PyTorch checkpoint: {pt_files[0]}")
            state_dict = torch.load(os.path.join(checkpoint_path, pt_files[0]))
            if isinstance(state_dict, dict):
                if 'state_dict' in state_dict:
                    print("\nCheckpoint state dict keys:")
                    for key in state_dict['state_dict'].keys():
                        print(f"  - {key}")
                else:
                    print("\nCheckpoint keys:")
                    for key in state_dict.keys():
                        print(f"  - {key}")
    except Exception as e:
        print(f"Error loading PyTorch checkpoint: {str(e)}")

checkpoint_path = "model-checkpoints/peteish7/step11931-unsharded-hf/"
inspect_checkpoint(checkpoint_path)

ModelRegistry.register_model("OlmoNewForCausalLM", OlmoNewForCausalLM)
sampling_params = SamplingParams(temperature=0.0)

try:
    llm = LLM(
        model=checkpoint_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.90
    )
    
    prompt = "San Francisco is a"
    outputs = llm.generate([prompt], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text
    print(f"Generated: {generated_text}")
    
except Exception as e:
    print(f"Error initializing LLM: {str(e)}")
    
finally:
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()