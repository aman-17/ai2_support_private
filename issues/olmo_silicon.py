from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests
import torch

# First check if MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device")
    device = 'cpu'
else:
    device = 'mps'

# Initialize processor and model with MPS device
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype=torch.float32  # MPS works better with float32
)

model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype=torch.float32
).to(device)

# Process the image
inputs = processor.process(
    images=[Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)],
    text="Describe this image."
)

# Move inputs to MPS device
inputs = {k: v.to(device).unsqueeze(0) for k, v in inputs.items()}

# Generate output
output = model.generate_from_batch(
    inputs,
    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
    tokenizer=processor.tokenizer
)

# Process output
generated_tokens = output[0,inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(generated_text)