import numpy as np
import requests
import torch
from PIL import Image, ImageOps
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from typing import List, Dict

processor = AutoProcessor.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="auto",
)
model = AutoModelForCausalLM.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="auto",
)
urls = [
    "https://picsum.photos/id/237/536/354",
    "https://picsum.photos/id/238/536/354",
    "https://picsum.photos/id/239/536/354",
]
prompts = [
    "What breed is this dog?",
    "Describe the colors in this image.",
    "Is this an indoor or outdoor scene?",
]

images_list = []
for url in urls:
    response = requests.get(url)
    image = Image.open(requests.get(url, stream=True).raw)
    images_list.append([image])

texts = ["User: " + prompt + " Assistant:" for prompt in prompts]


def process_batch(
    processor: AutoProcessor,
    texts: List[str],
    images_list: List[List[Image.Image]]
) -> Dict[str, torch.Tensor]:
    """
    Process in batch.
    
    Args:
        processor: The original processor.
        texts: List of text inputs
        images_list: List of lists containing PIL images.
        
    Returns:
        Dict with padded input_ids, images, image_input_idx, image_masks.
    """
    batch_size = len(texts)
    tokens_list = []
    for text in texts:
        tokens = processor.tokenizer.encode(" " + text, add_special_tokens=False)
        tokens_list.append(tokens)
    images_arrays_list = []
    image_idxs_list = []
    for images in images_list:
        if images:
            image_arrays = []
            for image in images:
                if isinstance(image, Image.Image):
                    image = image.convert("RGB")
                    image = ImageOps.exif_transpose(image)
                    image_arrays.append(np.array(image))
                else:
                    assert len(image.shape) == 3 and image.shape[-1] == 3
                    image_arrays.append(image.astype(np.uint8))
            images_arrays_list.append(image_arrays)
            image_idx = [-1] * len(image_arrays)
            image_idxs_list.append(image_idx)
        else:
            images_arrays_list.append(None)
            image_idxs_list.append(None)
    images_kwargs = {
        "max_crops": 12,
        "overlap_margins": [4, 4],
        "base_image_input_size": [336, 336],
        "image_token_length_w": 12,
        "image_token_length_h": 12,
        "image_patch_size": 14,
        "image_padding_mask": True,
    }
    outputs_list = []
    for i in range(batch_size):
        tokens = tokens_list[i]
        images = images_arrays_list[i]
        image_idx = image_idxs_list[i]
        out = processor.image_processor.multimodal_preprocess(
            images=images,
            image_idx=image_idx,
            tokens=np.asarray(tokens).astype(np.int32),
            sequence_length=1536,
            image_patch_token_id=processor.special_token_ids["<im_patch>"],
            image_col_token_id=processor.special_token_ids["<im_col>"],
            image_start_token_id=processor.special_token_ids["<im_start>"],
            image_end_token_id=processor.special_token_ids["<im_end>"],
            **images_kwargs,
        )
        outputs_list.append(out)

    batch_outputs = {}
    for key in outputs_list[0].keys():
        tensors = [torch.from_numpy(out[key]) for out in outputs_list]
        batch_outputs[key] = torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=-1
        )
    bos = processor.tokenizer.bos_token_id or processor.tokenizer.eos_token_id
    batch_outputs["input_ids"] = torch.nn.functional.pad(
        batch_outputs["input_ids"], (1, 0), value=bos
    )
    if "image_input_idx" in batch_outputs:
        image_input_idx = batch_outputs["image_input_idx"]
        batch_outputs["image_input_idx"] = torch.where(
            image_input_idx < 0, image_input_idx, image_input_idx + 1
        )
    return batch_outputs


inputs = process_batch(processor, texts, images_list)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

output = model.generate_from_batch(
    inputs,
    GenerationConfig(
        max_new_tokens=200,
        stop_sequences=["<|endoftext|>"],
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    ),
    tokenizer=processor.tokenizer,
)

generated_texts = processor.tokenizer.batch_decode(
    output[:, inputs["input_ids"].size(1) :], skip_special_tokens=True
)
for prompt, text in zip(prompts, generated_texts):
    print(f"\nPrompt: {prompt}")
    print(f"Response: {text}")