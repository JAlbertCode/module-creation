"""Multimodal input handler for Hugging Face models"""

def get_inference_code(model_id, task):
    return f'''import os
import json
from transformers import pipeline
import torch
from PIL import Image
import argparse

def get_output_dir():
    """Get output directory based on environment"""
    return '/outputs' if os.path.exists('/.dockerenv') else './outputs'

def main():
    parser = argparse.ArgumentParser(description='Run inference on {task} model')
    parser.add_argument('--image_path', help='Path to input image')
    parser.add_argument('--input_text', help='Input text for the model')
    args = parser.parse_args()

    image_path = args.image_path or os.environ.get('IMAGE_PATH')
    input_text = args.input_text or os.environ.get('INPUT_TEXT')
    
    if not image_path or not input_text:
        raise ValueError("Please provide both image_path and input_text via arguments or environment variables")
    
    try:
        print("Loading model...")
        device = 0 if torch.cuda.is_available() else -1
        print(f"Device set to use {{'gpu' if torch.cuda.is_available() else 'cpu'}}")
        
        pipe = pipeline(
            task="{task}", 
            model="{model_id}",
            device=device
        )
        
        print("Running inference...")
        image = Image.open(image_path)
        result = pipe(image=image, text=input_text)
        output = {{"result": result, "status": "success"}}
        print("Inference complete.")
        
    except Exception as e:
        output = {{"error": str(e), "status": "error"}}
        print(f"Error: {{e}}")
    
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'result.json')
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {{output_path}}")

if __name__ == "__main__":
    main()'''

def get_system_packages():
    return [
        "libgl1-mesa-glx",
        "libglib2.0-0"
    ]

def get_requirements():
    return [
        "pillow==10.0.0",
        "numpy<2.0.0"
    ]