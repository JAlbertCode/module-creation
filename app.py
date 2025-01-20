from flask import Flask, request, send_file, render_template
from huggingface_hub import HfApi
import os
import json
import zipfile
from io import BytesIO
import tempfile

app = Flask(__name__)

def get_model_info(model_url):
    """Extract model information from Hugging Face URL"""
    model_id = model_url.split('huggingface.co/')[-1].strip('/')
    api = HfApi()
    return api.model_info(model_id)

def generate_dockerfile(model_type):
    """Generate Dockerfile content"""
    return f"""FROM python:3.9-slim

# Set working directory
WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create output directory
RUN mkdir -p /outputs

# Copy inference script
COPY run_inference.py .

# Set entrypoint
ENTRYPOINT ["python", "/workspace/run_inference.py"]"""

def generate_requirements(model_info):
    """Generate requirements.txt content"""
    return """transformers==4.36.0
torch==2.1.0
numpy<2.0.0
pillow==10.0.0
huggingface-hub==0.19.4"""

def generate_run_inference(model_id, model_type):
    """Generate run_inference.py content"""
    return f'''import os
import json
from transformers import pipeline
import torch

def main():
    # Get input parameters from environment variables
    input_path = os.environ.get('INPUT_PATH')
    if not input_path:
        raise ValueError("INPUT_PATH environment variable must be set")
    
    try:
        # Initialize the pipeline
        pipe = pipeline("{model_type}", model="{model_id}")
        
        # Load input
        with open(input_path, 'r') as f:
            input_data = f.read()
        
        # Run inference
        result = pipe(input_data)
        
        # Format output
        output = {{
            'result': result,
            'status': 'success'
        }}
        
    except Exception as e:
        output = {{
            'error': str(e),
            'status': 'error'
        }}
    
    # Save output
    os.makedirs('/outputs', exist_ok=True)
    with open('/outputs/result.json', 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()'''

def generate_module_yaml(model_id):
    """Generate module.yaml content"""
    return f"""name: {model_id.split('/')[-1]}
version: 1.0.0
description: Hugging Face model deployment for {model_id}
resources:
  cpu: 1
  memory: 4Gi
  gpu: 1  # Remove if GPU is not needed
input:
  - name: INPUT_PATH
    description: Path to the input file
    type: string
output:
  - name: result
    description: Model output
    type: file
    path: /outputs/result.json"""

def create_zip_file(model_info):
    """Create zip file containing all necessary files"""
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        model_type = 'text-classification'  # Default type, can be detected from model_info
        model_id = model_info.id
        
        # Add files to zip
        zf.writestr('Dockerfile', generate_dockerfile(model_type))
        zf.writestr('requirements.txt', generate_requirements(model_info))
        zf.writestr('run_inference.py', generate_run_inference(model_id, model_type))
        zf.writestr('module.yaml', generate_module_yaml(model_id))
        
        # Add setup instructions
        instructions = f"""# Setup Instructions

1. Extract the files
2. Build the Docker image:
   ```bash
   docker build -t {model_id.split('/')[-1]} .
   ```

3. Run locally:
   ```bash
   docker run -v $(pwd)/input:/workspace/input \\
             -e INPUT_PATH=/workspace/input/input.txt \\
             {model_id.split('/')[-1]}
   ```

4. Deploy to Lilypad:
   ```bash
   lilypad module deploy .
   ```"""
        zf.writestr('README.md', instructions)
    
    memory_file.seek(0)
    return memory_file

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        model_url = request.form.get('model_url')
        if not model_url:
            return "Model URL is required", 400
            
        model_info = get_model_info(model_url)
        zip_file = create_zip_file(model_info)
        
        return send_file(
            zip_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'lilypad-{model_info.id.split("/")[-1]}.zip'
        )
        
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)