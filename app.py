"""Simple web application to generate Lilypad modules from Hugging Face models"""

from flask import Flask, request, send_file, render_template, jsonify
from huggingface_hub import HfApi
import os
import json
import zipfile
from io import BytesIO
import traceback

from modules import handlers, utils, model_types

app = Flask(__name__)

def get_model_info(model_url):
    """Extract model information from Hugging Face URL"""
    model_id = model_url.split('huggingface.co/')[-1].strip('/')
    api = HfApi()
    return api.model_info(model_id)

def detect_model_type(model_info):
    """Detect the model type from model info"""
    return model_types.detect_model_type(model_info)

def generate_module_template(model_id, model_type):
    """Generate lilypad_module.json.tmpl content"""
    input_vars = []
    
    # Add input variables based on input type
    if model_type['input'] in ['text', 'text-pair']:
        input_vars.append('{{ if .input_text }}"INPUT_TEXT={{ js .input_text }}"{{ end }}')
    
    if model_type['input'] in ['image', 'video', 'audio', 'document-text-pair', 'image-text-pair']:
        input_vars.append('{{ if .input_path }}"INPUT_PATH={{ js .input_path }}"{{ else }}"INPUT_PATH=/workspace/input/default_input"{{ end }}')

    if model_type['input'] in ['point-cloud', '3d']:
        input_vars.append('{{ if .model_path }}"MODEL_PATH={{ js .model_path }}"{{ else }}"MODEL_PATH=/workspace/input/model.ply"{{ end }}')

    if model_type['input'] == 'time-series':
        input_vars.extend([
            '{{ if .time_column }}"TIME_COLUMN={{ js .time_column }}"{{ end }}',
            '{{ if .value_column }}"VALUE_COLUMN={{ js .value_column }}"{{ end }}',
            '{{ if .window_size }}"WINDOW_SIZE={{ js .window_size }}"{{ end }}'
        ])

    if model_type['input'] == 'tabular':
        input_vars.extend([
            '{{ if .table_path }}"TABLE_PATH={{ js .table_path }}"{{ end }}',
            '{{ if .column }}"COLUMN={{ js .column }}"{{ end }}'
        ])

    env_vars = ',\n                    '.join(input_vars)

    # Determine GPU and memory requirements based on task
    gpu_tasks = ['text-to-image', 'text-to-video', 'image-to-image', '3d-mesh-reconstruction']
    high_memory_tasks = ['video-classification', 'point-cloud-segmentation', 'protein-structure-prediction']

    gpu_required = model_type['task'] in gpu_tasks
    memory = 16000 if model_type['task'] in high_memory_tasks else 8000

    return f'''{{
    "machine": {{
        "gpu": {1 if gpu_required else 0},
        "cpu": 1000,
        "ram": {memory}
    }},
    "job": {{
        "APIVersion": "V1beta1",
        "Spec": {{
            "Deal": {{
                "Concurrency": 1
            }},
            "Docker": {{
                "Entrypoint": ["python", "/workspace/run_inference.py"],
                "WorkingDirectory": "/workspace",
                "EnvironmentVariables": [
                    {env_vars}
                ],
                "Image": "username/{model_id.split('/')[-1].lower()}:latest"
            }},
            "Engine": "Docker",
            "Network": {{
                "Type": "None"
            }},
            "Outputs": [
                {{
                    "Name": "outputs",
                    "Path": "/outputs"
                }}
            ],
            "PublisherSpec": {{
                "Type": "ipfs"
            }},
            "Resources": {{
                "GPU": "{1 if gpu_required else 0}"
            }},
            "Timeout": 1800
        }}
    }}
}}'''

def generate_readme(model_id, model_type):
    """Generate README.md with type-specific instructions"""
    input_instructions = {
        'text': "Provide text directly via --input_text or in input/text.txt file",
        'image': "Place image files (JPG/PNG) in the input directory",
        'audio': "Place audio files (WAV/MP3) in the input directory",
        'video': "Place video files (MP4) in the input directory",
        'point-cloud': "Place 3D files (PLY/OBJ/STL) in the input directory",
        'time-series': "Place time series data (CSV/JSON) in the input directory",
        'tabular': "Place data files (CSV/JSON/Excel) in the input directory",
        'graph': "Place graph data (JSON/GraphML) in the input directory",
        'image-text-pair': "Provide both image file and text input",
        'document-text-pair': "Provide both document file and text input",
        'audio-video-pair': "Provide both audio and video files",
        'multimodal': "Provide all required input files in their respective formats"
    }

    cli_examples = {
        'text': f'python run_inference.py --input_text="Your text here"',
        'image': 'python run_inference.py --image_path=input/image.jpg',
        'audio': 'python run_inference.py --audio_path=input/audio.wav',
        'video': 'python run_inference.py --video_path=input/video.mp4',
        'point-cloud': 'python run_inference.py --model_path=input/model.ply',
        'time-series': 'python run_inference.py --input_path=input/data.csv --time_column=timestamp --value_column=value',
        'tabular': 'python run_inference.py --table_path=input/data.csv --column=feature',
        'image-text-pair': 'python run_inference.py --image_path=input/image.jpg --input_text="Your text here"',
        'document-text-pair': 'python run_inference.py --document_path=input/document.pdf --input_text="Your question here"'
    }

    # Get description and example usage
    task_desc = model_type['task'].replace('-', ' ').title()
    example = cli_examples.get(model_type['input'], cli_examples['text'])
    instruction = input_instructions.get(model_type['input'], input_instructions['text'])

    # Determine hardware requirements
    gpu_required = model_type['task'] in ['text-to-image', 'text-to-video', 'image-to-image']
    min_memory = "16GB" if model_type['task'] in ['video-classification', 'point-cloud-segmentation'] else "8GB"

    return f'''# Lilypad Module for {model_id}

This module deploys [{model_id}](https://huggingface.co/{model_id}) from Hugging Face for {task_desc} using Lilypad.

## Task Description

This model performs {task_desc} and accepts {model_type['input']} input type.

## Setup

1. Build the Docker image:
   ```bash
   docker build -t {model_id.split('/')[-1]} .
   docker tag {model_id.split('/')[-1]} username/{model_id.split('/')[-1]}:latest
   docker push username/{model_id.split('/')[-1]}:latest
   ```

2. Test locally:
   ```bash
   # Example usage:
   {example}

   # Using Docker:
   docker run -v $(pwd)/input:/workspace/input -v $(pwd)/output:/outputs username/{model_id.split('/')[-1]}:latest
   ```

3. Run on Lilypad:
   ```bash
   lilypad run username/{model_id.split('/')[-1]} [input parameters]
   ```

## Input Format

- Type: {model_type['input']}
- {instruction}
- Results will be saved to: output/result.json

## System Requirements

- {"GPU with at least 8GB VRAM" if gpu_required else "CPU only"}
- Minimum RAM: {min_memory}
- Docker installed

## Advanced Configuration

The lilypad_module.json.tmpl file can be modified to adjust:
- GPU/CPU allocation
- Memory requirements
- Network access
- Environment variables
- Timeout duration

## Error Handling

The module provides detailed error messages in case of failures:
```json
{{
    "error": "Error description",
    "status": "error"
}}
```

## Model Information

Visit the [model page]({model_info.url}) for detailed information about:
- Model architecture
- Training data
- Usage guidelines
- License information
'''

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
            zf.writestr('Dockerfile', utils.generate_dockerfile(model_type))
            zf.writestr('requirements.txt', utils.generate_requirements(model_type))
            
            # Get inference code from appropriate handler
            inference_code = handlers.get_inference_code(model_type['input'], model_info.id, model_type['task'])
            if inference_code:
                zf.writestr('run_inference.py', inference_code)
            else:
                # Use default text handler if no specific handler exists
                zf.writestr('run_inference.py', handlers.get_inference_code('text', model_info.id, model_type['task']))
            
            zf.writestr('lilypad_module.json.tmpl', generate_module_template(model_info.id, model_type))
            zf.writestr('README.md', generate_readme(model_info.id, model_type))
            
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

            # Add example test script
            test_script = f'''#!/bin/bash
# Build the Docker image
docker build -t {model_info.id.split("/")[-1]} .

# Run a test with example input
{cli_examples.get(model_type['input'], cli_examples['text'])}

# Check the output
cat output/result.json'''
            zf.writestr('test.sh', test_script)
            
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