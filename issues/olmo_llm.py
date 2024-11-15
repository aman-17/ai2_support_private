from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
    dtype = torch.float16 
elif torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.float16 
else:
    device = torch.device("cpu")
    dtype = torch.float32
print(f"Using device: {device} with dtype: {dtype}")


olmo = OLMoForCausalLM.from_pretrained("allenai/OLMo-1B", torch_dtype=dtype)
tokenizer = OLMoTokenizerFast.from_pretrained("allenai/OLMo-1B")

olmo = olmo.to(device)
message = ["Language modeling is"]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False).to(device)
if device.type == 'mps':
    with torch.no_grad():
        response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
else:
    with torch.autocast(device_type=device.type, dtype=dtype):
        response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)

print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])