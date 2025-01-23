"""Simple web application to generate Lilypad modules from Hugging Face models"""

from flask import Flask, request, send_file, render_template, jsonify
from huggingface_hub import HfApi
import os
import json
import zipfile
from io import BytesIO
import traceback

app = Flask(__name__)

def get_model_info(model_url):
    """Extract model information from Hugging Face URL"""
    model_id = model_url.split('huggingface.co/')[-1].strip('/')
    api = HfApi()
    return api.model_info(model_id)

def detect_model_type(model_info):
    """Detect the model type from model info"""
    pipeline_mapping = {
        'text-classification': {'task': 'text-classification', 'input': 'text'},
        'image-classification': {'task': 'image-classification', 'input': 'image'},
        'object-detection': {'task': 'object-detection', 'input': 'image'},
        'question-answering': {'task': 'question-answering', 'input': 'text'}
    }
    
    for tag in model_info.tags:
        if tag in pipeline_mapping:
            return pipeline_mapping[tag]
    
    return {'task': 'text-classification', 'input': 'text'}  # Default

def generate_dockerfile():
    """Generate Dockerfile content"""
    return '''FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /outputs /workspace/input

# Copy inference script
COPY run_inference.py .

# Set entrypoint
ENTRYPOINT ["python", "/workspace/run_inference.py"]'''

def generate_requirements(model_type):
    """Generate requirements.txt content"""
    reqs = [
        "transformers==4.36.0",
        "torch==2.1.0",
        "numpy<2.0.0"
    ]
    
    if model_type['input'] == 'image':
        reqs.extend([
            "pillow==10.0.0",
            "torchvision==0.16.0"
        ])
    
    return "\n".join(reqs)

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
        use_gpu = torch.cuda.is_available()
        device = 0 if use_gpu else -1
        print(f"Device set to use {'gpu' if use_gpu else 'cpu'}")
        
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
        use_gpu = torch.cuda.is_available()
        device = 0 if use_gpu else -1
        print(f"Device set to use {'gpu' if use_gpu else 'cpu'}")
        
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

def generate_module_yaml(model_id, model_type):
    """Generate module.yaml content"""
    return f'''name: {model_id.split('/')[-1]}
version: 1.0.0
description: Hugging Face {model_type['task']} model deployment

resources:
  cpu: 2
  memory: 4Gi
  gpu: 1  # Remove if GPU is not needed

input:
  - name: INPUT_PATH
    description: Path to the input file ({model_type['input']} format)
    type: string
    required: false  # Not required since we can use command line args

output:
  - name: result
    description: Model output
    type: file
    path: /outputs/result.json'''

def generate_readme(model_id, model_type):
    """Generate README.md content"""
    return f'''# Lilypad Module for {model_id}

This module deploys the [{model_id}](https://huggingface.co/{model_id}) model from Hugging Face.

## Setup

1. Build the Docker image:
   ```bash
   docker build -t {model_id.split('/')[-1]} .
   ```

2. Run the module:

   {"For image input:" if model_type["input"] == "image" else "For text input:"}
   ```bash
   # Using command line arguments:
   {"python run_inference.py --image_path=input/image.jpg" if model_type["input"] == "image" else "python run_inference.py --input_text=\"Your text here\""}

   # Using Docker with input directory:
   docker run -v $(pwd)/input:/workspace/input \\
             -e INPUT_PATH=/workspace/input/{"image.jpg" if model_type["input"] == "image" else "input.txt"} \\
             {model_id.split('/')[-1]}
   ```

3. Deploy to Lilypad:
   ```bash
   lilypad module deploy .
   ```

## Input Format

- Type: {model_type['input']}
- {"Place image files in the input directory" if model_type["input"] == "image" else "Provide text directly via --input_text or place in input.txt"}
- Results will be in: output/result.json'''

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        model_url = request.form.get('model_url')
        if not model_url or 'huggingface.co' not in model_url:
            return "Invalid model URL", 400
            
        # Get model info
        model_info = get_model_info(model_url)
        model_type = detect_model_type(model_info)
        
        # Create zip file
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add files
            zf.writestr('Dockerfile', generate_dockerfile())
            zf.writestr('requirements.txt', generate_requirements(model_type))
            zf.writestr('run_inference.py', generate_inference(model_info.id, model_type))
            zf.writestr('module.yaml', generate_module_yaml(model_info.id, model_type))
            zf.writestr('README.md', generate_readme(model_info.id, model_type))
            
            # Add example input directory
            if model_type['input'] == 'image':
                zf.writestr('input/README.md', 'Place your image files (JPG/PNG) here')
            else:
                zf.writestr('input/README.md', 'Place your text files here if not using command line input')
        
        memory_file.seek(0)
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'lilypad-{model_info.id.split("/")[-1]}.zip'
        )
        
    except Exception as e:
        # Get the full error traceback
        error_traceback = traceback.format_exc()
        print(f"Error: {error_traceback}")  # Print to console for debugging
        return f"Error: {str(e)}\n\nTraceback: {error_traceback}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)