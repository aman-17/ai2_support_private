import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import requests

# Custom Dataset Class
class YourDataset(Dataset):
    def __init__(self, image_urls, texts, transform=None):
        self.image_urls = image_urls
        self.texts = texts
        self.transform = transform

    def __len__(self):
        return len(self.image_urls)

    def __getitem__(self, index):
        # Load image from URL
        image = Image.open(requests.get(self.image_urls[index], stream=True).raw).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        text = self.texts[index]
        return image, text

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to your desired size
    transforms.ToTensor(),           # Convert to tensor
])

# Sample data (replace with your actual image URLs and texts)
image_urls = ["https://picsum.photos/id/237/536/354", "https://picsum.photos/id/238/536/354"]
texts = ["Describe this image.", "Provide details about this image."]

# Create Dataset and DataLoader
dataset = YourDataset(image_urls, texts, transform)
data_loader = DataLoader(dataset, batch_size=2, shuffle=False)

# Load model and processor
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# Inference
for batch in data_loader:
    images, texts = zip(*batch)  # Unzip batch into images and texts
    
    # Convert images to a single tensor
    images = torch.stack([transform(image) for image in images])  # Stack images to create a batch tensor

    inputs = processor(images=images, text=list(texts))

    # Generate output
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=200, stop_strings=["<|endoftext|>"]),
        tokenizer=processor.tokenizer
    )

    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(generated_text)