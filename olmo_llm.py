from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
import torch

# Check if MPS, CUDA, or CPU is available, and set precision accordingly
if torch.backends.mps.is_available():
    device = torch.device("mps")
    dtype = torch.float16  # MPS does not support float16, so we force float32
elif torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.float16  # CUDA supports float16
else:
    device = torch.device("cpu")
    dtype = torch.float32  # Default to float32 for CPU

print(f"Using device: {device} with dtype: {dtype}")

# Load the model and tokenizer
olmo = OLMoForCausalLM.from_pretrained("allenai/OLMo-1B", torch_dtype=dtype)
tokenizer = OLMoTokenizerFast.from_pretrained("allenai/OLMo-1B")

# Move the model to the specified device
olmo = olmo.to(device)

# Tokenize the input message
message = ["Language modeling is"]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False).to(device)

# MPS doesn't support float16, so we use float32 with no autocast
if device.type == 'mps':
    with torch.no_grad():  # Inference only, disable gradients
        response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
else:
    # For CUDA or CPU, we allow mixed precision with autocast if using float16
    with torch.autocast(device_type=device.type, dtype=dtype):
        response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)

# Decode and print the generated response
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])