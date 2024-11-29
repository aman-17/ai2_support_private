from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests
import torch
import cv2
import numpy as np
import re

import re
from typing import List, Tuple

def parse_coordinates(text: str) -> List[Tuple[float, float]]:
    """
    Extracts x,y coordinates from XML-style point tags.
    
    Args:
        text (str): Input text containing point tags with x and y attributes
        
    Returns:
        List[Tuple[float, float]]: List of (x, y) coordinate pairs
        
    Example:
        >>> text = '<point x="70.4" y="45.1">tip</point>'
        >>> parse_coordinates(text)
        [(70.4, 45.1)]
    """
    # Instead of trying to match x and y in a specific order, we'll capture the entire
    # tag content and use a separate regex to find x and y attributes individually
    tag_pattern = r'<point\s+([^>]*)>(?:[^<]*)</point>'
    
    # Patterns to find x and y attributes anywhere within the tag content
    x_pattern = r'x="(-?\d*\.?\d+)"'
    y_pattern = r'y="(-?\d*\.?\d+)"'
    
    coordinates = []
    
    # Find all point tags
    for tag_match in re.finditer(tag_pattern, text):
        tag_content = tag_match.group(1)
        
        # Find x and y values within the tag content
        x_match = re.search(x_pattern, tag_content)
        y_match = re.search(y_pattern, tag_content)
        
        # Only proceed if both x and y are found
        if x_match and y_match:
            x = float(x_match.group(1))
            y = float(y_match.group(1))
            coordinates.append((x, y))
    
    return coordinates

# Test cases to verify the parser works correctly
def test_parser():
    """Verify the coordinate parser with various test cases"""
    
    test_cases = [
        # Basic case
        ('<point x="70.4" y="45.1" alt="tip">tip of mountain</point>', 
         [(70.4, 45.1)]),
        
        # Multiple points
        ('<point x="1.0" y="2.0">first</point><point x="3.0" y="4.0">second</point>', 
         [(1.0, 2.0), (3.0, 4.0)]),
        
        # Negative coordinates
        ('<point x="-1.5" y="-2.5">negative</point>', 
         [(-1.5, -2.5)]),
        
        # Different attribute order
        ('<point y="1.0" x="2.0">reversed</point>', 
         [(2.0, 1.0)]),
        
        # With other attributes
        ('<point id="1" x="1.0" class="marker" y="2.0" alt="test">with attrs</point>', 
         [(1.0, 2.0)]),
         
        # With whitespace variations
        ('<point   x="1.0"     y="2.0"   >spaced</point>', 
         [(1.0, 2.0)]),
    ]
    
    for input_text, expected in test_cases:
        result = parse_coordinates(input_text)
        assert result == expected, f"Failed: {input_text}\nExpected: {expected}\nGot: {result}"
    
    print("All test cases passed!")


def locate_point_in_image(image, query, model=None, processor=None):
    """Pipeline to locate and mark a point in an image based on a text query"""
    # Load models if not provided
    if model is None or processor is None:
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

    # Process inputs
    inputs = processor.process(
        images=[image],
        text=query
    )
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # Generate output
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=50, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )

    # Get coordinates from response
    generated_tokens = output[0,inputs['input_ids'].size(1):]
    response = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(f"Response: {response}")
    
    coordinates = parse_coordinates(response)
    
    # Convert PIL Image to cv2 format
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Draw green point
    for (x, y) in coordinates:
        cv2.circle(img_cv2, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    return img_cv2

# Example usage:
if __name__ == "__main__":
    # Load image
    image_url = "https://picsum.photos/id/237/536/354"
    # image_url = "https://picsum.photos/seed/picsum/200/123"
    image = Image.open(requests.get(image_url, stream=True).raw)
    
    # Process query
    query = "Point at the puppy's nose"
    # query = "Point at the tip of the mountain"
    result_image = locate_point_in_image(image, query)

    # Save result instead of displaying
    cv2.imwrite('result.png', result_image)
    
    # # Display result
    # cv2.imshow('Result', result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()