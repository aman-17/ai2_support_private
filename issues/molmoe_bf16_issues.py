# from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
# from PIL import Image
# import requests
# import torch

# processor = AutoProcessor.from_pretrained(
#     'allenai/MolmoE-1B-0924',
#     trust_remote_code=True,
#     torch_dtype=torch.float16,
#     device_map='auto'
# )
# model = AutoModelForCausalLM.from_pretrained(
#     'allenai/MolmoE-1B-0924',
#     trust_remote_code=True,
#     torch_dtype=torch.float16,
#     device_map='auto'
# )

# image = Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)
# print("\nImage type:", type(image))

# inputs = processor.process(
#     images=[image],
#     text="Describe this image."
# )

# print("\nInput dtypes:")
# for key, value in inputs.items():
#     if torch.is_tensor(value):
#         print(f"{key}: {value.dtype}")

# inputs = {
#     k: (v.to(torch.float16).to(model.device).unsqueeze(0) 
#        if k in ['images', 'image_masks'] 
#        else v.to(model.device).unsqueeze(0))
#     if torch.is_tensor(v) else v
#     for k, v in inputs.items()
# }

# print("\nInput dtypes after conversion:")
# for key, value in inputs.items():
#     if torch.is_tensor(value):
#         print(f"{key}: {value.dtype}")

# output = model.generate_from_batch(
#     inputs,
#     GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
#     tokenizer=processor.tokenizer
# )

# generated_tokens = output[0,inputs['input_ids'].size(1):]
# generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
# print("\nGenerated text:", generated_text)

# from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
# from PIL import Image
# import requests
# import torch
# import torchvision.transforms as transforms

# processor = AutoProcessor.from_pretrained(
#     'allenai/MolmoE-1B-0924',
#     trust_remote_code=True,
#     torch_dtype=torch.bfloat16,
#     device_map='auto'
# )
# model = AutoModelForCausalLM.from_pretrained(
#     'allenai/MolmoE-1B-0924',
#     trust_remote_code=True,
#     torch_dtype=torch.bfloat16,
#     device_map='auto'
# )
# image = Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)
# inputs = processor.process(
#     images=[image],
#     text="Describe this image."
# )

# print("\nInput dtypes before processing:")
# for key, value in inputs.items():
#     if torch.is_tensor(value):
#         print(f"{key}: {value.dtype}")

# processed_inputs = {}
# for k, v in inputs.items():
#     if not torch.is_tensor(v):
#         processed_inputs[k] = v
#         continue
#     if v.dtype in [torch.int32, torch.int64]:
#         processed_inputs[k] = v.to(model.device).unsqueeze(0)
#     elif k in ['images', 'image_masks']:
#         if k == 'images':
#             v = v / 255.0 if v.max() > 1.0 else v
#         processed_inputs[k] = v.to(torch.bfloat16).to(model.device).unsqueeze(0)
#     else:
#         processed_inputs[k] = v.to(model.device).unsqueeze(0)

# print("\nInput dtypes after processing:")
# for key, value in processed_inputs.items():
#     if torch.is_tensor(value):
#         print(f"{key}: {value.dtype}")
#         if key == 'images':
#             print(f"Image tensor range: [{value.min().item():.3f}, {value.max().item():.3f}]")
# output = model.generate_from_batch(
#     processed_inputs,
#     GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
#     tokenizer=processor.tokenizer
# )
# generated_tokens = output[0,processed_inputs['input_ids'].size(1):]
# generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
# print("\nGenerated text:", generated_text)

# from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
# from PIL import Image
# import requests
# import torch

# processor = AutoProcessor.from_pretrained(
#     'allenai/MolmoE-1B-0924',
#     trust_remote_code=True,
#     torch_dtype=torch.bfloat16,
#     device_map='auto'
# )
# model = AutoModelForCausalLM.from_pretrained(
#     'allenai/MolmoE-1B-0924',
#     trust_remote_code=True,
#     torch_dtype=torch.bfloat16,
#     device_map='auto'
# )

# image = Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)
# print("\nOriginal image size:", image.size)

# image = image.convert('RGB')
# inputs = processor.process(
#     images=[image],
#     text="Describe this image."
# )
# print("\nTensor shapes before processing:")
# for key, value in inputs.items():
#     if torch.is_tensor(value):
#         print(f"{key}: {value.shape}")
# processed_inputs = {}
# for k, v in inputs.items():
#     if not torch.is_tensor(v):
#         processed_inputs[k] = v
#         continue
    
#     # Original processor's image tensor but convert to bfloat16
#     if k == 'images':
#         print(f"\nImage tensor stats before conversion:")
#         print(f"Shape: {v.shape}")
#         print(f"Range: [{v.min().item():.3f}, {v.max().item():.3f}]")
#         processed_inputs[k] = v.to(torch.bfloat16).to(model.device).unsqueeze(0)
#     elif k == 'image_masks':
#         processed_inputs[k] = v.to(torch.bfloat16).to(model.device).unsqueeze(0)
#     elif v.dtype in [torch.int32, torch.int64]:
#         processed_inputs[k] = v.to(model.device).unsqueeze(0)
#     else:
#         processed_inputs[k] = v.to(model.device).unsqueeze(0)
# print("\nTensor shapes and dtypes after processing:")
# for key, value in processed_inputs.items():
#     if torch.is_tensor(value):
#         print(f"{key}:")
#         print(f"  Shape: {value.shape}")
#         print(f"  Dtype: {value.dtype}")
#         if key == 'images':
#             print(f"  Range: [{value.min().item():.3f}, {value.max().item():.3f}]")
# output = model.generate_from_batch(
#     processed_inputs,
#     GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
#     tokenizer=processor.tokenizer
# )
# generated_tokens = output[0,processed_inputs['input_ids'].size(1):]
# generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
# print("\nGenerated text:", generated_text)


from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests
import torch
import torchvision.transforms as transforms

# Load model and processor
processor = AutoProcessor.from_pretrained(
    'allenai/MolmoE-1B-0924',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
model = AutoModelForCausalLM.from_pretrained(
    'allenai/MolmoE-1B-0924',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

# Load and preprocess image
image = Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)
print("\nOriginal image size:", image.size)

# Convert to RGB and resize to a standard size
image = image.convert('RGB')
transform = transforms.Compose([
    transforms.Resize((576, 588)),  # Match the processor's output dimensions
    transforms.ToTensor(),
])
image_tensor = transform(image)
print("Initial tensor shape:", image_tensor.shape)

# Process inputs with the preprocessed image
inputs = processor.process(
    images=[image_tensor],  # Pass tensor instead of PIL image
    text="Describe this image."
)

# Print initial shapes
print("\nTensor shapes after processor:")
for key, value in inputs.items():
    if torch.is_tensor(value):
        print(f"{key}: {value.shape}")

# Process tensors
processed_inputs = {}
for k, v in inputs.items():
    if not torch.is_tensor(v):
        processed_inputs[k] = v
        continue
    
    # Convert to appropriate dtype and move to device
    if k == 'images':
        processed_inputs[k] = v.to(torch.bfloat16).to(model.device).unsqueeze(0)
    elif k == 'image_masks':
        processed_inputs[k] = v.to(torch.bfloat16).to(model.device).unsqueeze(0)
    else:
        processed_inputs[k] = v.to(model.device).unsqueeze(0)

# Print final shapes and dtypes
print("\nFinal tensor shapes and dtypes:")
for key, value in processed_inputs.items():
    if torch.is_tensor(value):
        print(f"{key}:")
        print(f"  Shape: {value.shape}")
        print(f"  Dtype: {value.dtype}")
        if key == 'images':
            print(f"  Range: [{value.min().item():.3f}, {value.max().item():.3f}]")

# Generate output
output = model.generate_from_batch(
    processed_inputs,
    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
    tokenizer=processor.tokenizer
)

generated_tokens = output[0,processed_inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
print("\nGenerated text:", generated_text)