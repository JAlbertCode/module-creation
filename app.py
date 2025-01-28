"""Simple web application to generate Lilypad modules from Hugging Face models"""

from flask import Flask, request, send_file, render_template, jsonify
from huggingface_hub import HfApi
import os
import json
import zipfile
from io import BytesIO
import traceback

from modules import handlers, utils, model_types, validation

app = Flask(__name__)

def get_model_info(model_url):
    """Extract model information from Hugging Face URL"""
    model_id = model_url.split('huggingface.co/')[-1].strip('/')
    api = HfApi()
    return api.model_info(model_id)

def detect_model_type(model_info):
    """Detect the model type from model info"""
    return model_types.detect_model_type(model_info)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/validate', methods=['POST'])
def validate_model():
    """Validate a model before generation"""
    try:
        model_url = request.form.get('model_url')
        if not model_url or 'huggingface.co' not in model_url:
            return jsonify({
                'valid': False,
                'message': 'Invalid model URL. Please provide a valid Hugging Face model URL.'
            })
        
        # Get model info
        model_info = get_model_info(model_url)
        model_type = detect_model_type(model_info)
        
        # Validate model
        validation_result = validation.check_model_compatibility(model_info)
        message = validation.format_validation_message(validation_result)
        
        # Get input requirements
        input_requirements = validation.InputValidator.get_input_requirements(model_type['input'])
        
        return jsonify({
            'valid': validation_result.is_valid,
            'message': message,
            'model_type': model_type,
            'requirements': validation_result.requirements,
            'input_requirements': input_requirements
        })
        
    except Exception as e:
        return jsonify({
            'valid': False,
            'message': f'Error validating model: {str(e)}'
        })

@app.route('/generate', methods=['POST'])
def generate():
    """Generate Lilypad module files"""
    try:
        model_url = request.form.get('model_url')
        if not model_url or 'huggingface.co' not in model_url:
            return "Invalid model URL", 400
            
        # Get model info
        model_info = get_model_info(model_url)
        model_type = detect_model_type(model_info)
        
        # Validate model
        validation_result = validation.check_model_compatibility(model_info)
        if not validation_result.is_valid:
            message = validation.format_validation_message(validation_result)
            return message, 400
        
        # Create zip file
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add files
            zf.writestr('Dockerfile', utils.generate_dockerfile(model_type))
            zf.writestr('requirements.txt', utils.generate_requirements(model_type))
            
            # Get inference code from appropriate handler
            inference_code = handlers.get_inference_code(model_type['input'], model_info.id, model_type['task'])
            if inference_code:
                zf.writestr('run_inference.py', inference_code)
            else:
                # Use default text handler if no specific handler exists
                zf.writestr('run_inference.py', handlers.get_inference_code('text', model_info.id, model_type['task']))
            
            # Add validation script
            validation_script = f'''#!/bin/bash
# Validate input files before running inference

INPUT_TYPE="{model_type['input']}"
REQUIRED_FORMATS="{", ".join(validation.InputValidator.SUPPORTED_FORMATS.get(model_type['input'], {}).get('extensions', []))}"
MAX_SIZE="{validation.InputValidator.SUPPORTED_FORMATS.get(model_type['input'], {}).get('max_size', 0)}"

function validate_file() {{
    local file=$1
    
    # Check file exists
    if [ ! -f "$file" ]; then
        echo "Error: File not found: $file"
        exit 1
    fi
    
    # Check file extension
    if [[ ! $file =~ \\.({"|".join(validation.InputValidator.SUPPORTED_FORMATS.get(model_type['input'], {}).get('extensions', [])).replace(".", "")})$ ]]; then
        echo "Error: Invalid file format. Supported formats: $REQUIRED_FORMATS"
        exit 1
    fi
    
    # Check file size
    size=$(stat -f%z "$file")
    if [ $size -gt $MAX_SIZE ]; then
        echo "Error: File too large. Maximum size: $(($MAX_SIZE/1024/1024))MB"
        exit 1
    fi
}}

# Validate input directory
if [ -d "/workspace/input" ]; then
    for file in /workspace/input/*; do
        if [ -f "$file" ]; then
            validate_file "$file"
        fi
    done
fi

echo "Input validation passed"
'''
            zf.writestr('validate_input.sh', validation_script)
            
            zf.writestr('lilypad_module.json.tmpl', generate_module_template(model_info.id, model_type))
            zf.writestr('README.md', generate_readme(model_info.id, model_type, validation_result))
            
            # Add example input directory and instructions based on type
            input_readme = {
                'text': 'Place your text files here if not using command line input',
                'image': 'Place your image files (JPG/PNG) here',
                'audio': 'Place your audio files (WAV/MP3) here',
                'video': 'Place your video files (MP4) here',
                'point-cloud': 'Place your 3D files (PLY/OBJ/STL) here',
                'time-series': 'Place your time series data (CSV/JSON) here',
                'tabular': 'Place your data files (CSV/JSON/Excel) here',
                'document-text-pair': 'Place your document files (PDF) here',
                'image-text-pair': 'Place your image files here',
            }.get(model_type['input'], 'Place input files here')
            
            zf.writestr('input/README.md', input_readme)

            # Add test script
            test_script = f'''#!/bin/bash
set -e

# Validate inputs
./validate_input.sh

# Build the Docker image
echo "Building Docker image..."
docker build -t {model_info.id.split("/")[-1]} .

# Run inference with example input
echo "Running inference..."
docker run -v $(pwd)/input:/workspace/input \\
          -v $(pwd)/output:/outputs \\
          {model_info.id.split("/")[-1]}

echo "Checking results..."
cat output/result.json

echo "Test complete."'''
            zf.writestr('test.sh', test_script)
            
            # Add GitHub Actions workflow for testing
            github_workflow = f'''name: Test Module

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Docker
      uses: docker/setup-buildx-action@v1
    
    - name: Build image
      run: docker build -t test-module .
    
    - name: Validate example inputs
      run: |
        chmod +x validate_input.sh
        ./validate_input.sh
    
    - name: Run test inference
      run: |
        mkdir -p output
        docker run -v ${{{{ github.workspace }}}}/input:/workspace/input \\
                  -v ${{{{ github.workspace }}}}/output:/outputs \\
                  test-module
        
    - name: Check results
      run: cat output/result.json'''
            
            zf.writestr('.github/workflows/test.yml', github_workflow)
            
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