[Previous imports and Flask setup remain the same...]

def generate_inference(model_id, model_type):
    """Generate run_inference.py content"""
    if model_type['input'] == 'image':
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
    parser = argparse.ArgumentParser(description='Run inference on {model_type["task"]} model')
    parser.add_argument('--image_path', help='Path to input image')
    args = parser.parse_args()

    input_path = args.image_path or os.environ.get('INPUT_PATH')
    if not input_path:
        raise ValueError("Please provide image path via --image_path or INPUT_PATH environment variable")
    
    try:
        print("Loading model...")
        device = 0 if torch.cuda.is_available() else -1
        print(f"Device set to use {{\'gpu\' if torch.cuda.is_available() else \'cpu\'}}")
        
        pipe = pipeline(
            task="{model_type['task']}", 
            model="{model_id}",
            device=device
        )
        
        print("Running inference...")
        image = Image.open(input_path)
        result = pipe(image)
        output = {{"result": result, "status": "success"}}
        print("Inference complete.")
        
    except Exception as e:
        output = {{"error": str(e), "status": "error"}}
        print(f"Error: {{e}}")
    
    # Use appropriate output directory
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'result.json')
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {{output_path}}")

if __name__ == "__main__":
    main()'''
    else:
        return f'''import os
import json
from transformers import pipeline
import torch
import argparse

def get_output_dir():
    """Get output directory based on environment"""
    return '/outputs' if os.path.exists('/.dockerenv') else './outputs'

def main():
    parser = argparse.ArgumentParser(description='Run inference on {model_type["task"]} model')
    parser.add_argument('--input_text', help='Input text for the model')
    args = parser.parse_args()
    
    input_text = args.input_text
    if not input_text:
        input_path = os.environ.get('INPUT_PATH')
        if input_path:
            with open(input_path, 'r') as f:
                input_text = f.read()
        else:
            raise ValueError("Please provide input via --input_text argument")
    
    try:
        print("Loading model...")
        device = 0 if torch.cuda.is_available() else -1
        print(f"Device set to use {{\'gpu\' if torch.cuda.is_available() else \'cpu\'}}")
        
        pipe = pipeline(
            task="{model_type['task']}", 
            model="{model_id}",
            device=device
        )
        
        print("Running inference...")
        result = pipe(input_text)
        output = {{"result": result, "status": "success"}}
        print("Inference complete.")
        
    except Exception as e:
        output = {{"error": str(e), "status": "error"}}
        print(f"Error: {{e}}")
    
    # Use appropriate output directory
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'result.json')
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {{output_path}}")

if __name__ == "__main__":
    main()'''

[Rest of the file remains the same...]