"""Simple web application to generate Lilypad modules from Hugging Face models"""

from flask import Flask, request, send_file, render_template, jsonify
from huggingface_hub import HfApi
import os
import json
import zipfile
from io import BytesIO
import traceback

from modules import handlers, utils

app = Flask(__name__)

def get_model_info(model_url):
    """Extract model information from Hugging Face URL"""
    model_id = model_url.split('huggingface.co/')[-1].strip('/')
    api = HfApi()
    return api.model_info(model_id)

def detect_model_type(model_info):
    """Detect the model type from model info"""
    pipeline_mapping = {
        # Text Tasks
        'text-classification': {'task': 'text-classification', 'input': 'text'},
        'text-generation': {'task': 'text-generation', 'input': 'text'},
        'question-answering': {'task': 'question-answering', 'input': 'text'},
        'summarization': {'task': 'summarization', 'input': 'text'},
        'translation': {'task': 'translation', 'input': 'text'},
        'text2text-generation': {'task': 'text2text-generation', 'input': 'text'},
        'sentence-similarity': {'task': 'sentence-similarity', 'input': 'text-pair'},
        'token-classification': {'task': 'token-classification', 'input': 'text'},
        
        # Image Tasks
        'image-classification': {'task': 'image-classification', 'input': 'image'},
        'object-detection': {'task': 'object-detection', 'input': 'image'},
        'image-segmentation': {'task': 'image-segmentation', 'input': 'image'},
        'image-to-text': {'task': 'image-to-text', 'input': 'image'},
        'text-to-image': {'task': 'text-to-image', 'input': 'text'},
        'image-to-image': {'task': 'image-to-image', 'input': 'image'},
        
        # Audio Tasks
        'automatic-speech-recognition': {'task': 'speech-recognition', 'input': 'audio'},
        'audio-classification': {'task': 'audio-classification', 'input': 'audio'},
        'text-to-speech': {'task': 'text-to-speech', 'input': 'text'},
        
        # Video Tasks
        'video-classification': {'task': 'video-classification', 'input': 'video'},
        'text-to-video': {'task': 'text-to-video', 'input': 'text'},
        
        # Code Tasks
        'text-to-code': {'task': 'text-to-code', 'input': 'text'},
        'code-to-text': {'task': 'code-to-text', 'input': 'text'},
        
        # Multimodal Tasks
        'visual-question-answering': {'task': 'vqa', 'input': 'image-text-pair'},
        'document-question-answering': {'task': 'document-qa', 'input': 'document-text-pair'}
    }
    
    for tag in model_info.tags:
        if tag in pipeline_mapping:
            return pipeline_mapping[tag]
    
    # If no specific tag is found, try to infer from model card
    if model_info.cardData:
        if 'audio' in str(model_info.cardData).lower():
            return {'task': 'audio-classification', 'input': 'audio'}
        if 'video' in str(model_info.cardData).lower():
            return {'task': 'video-classification', 'input': 'video'}
        if 'image' in str(model_info.cardData).lower():
            return {'task': 'image-classification', 'input': 'image'}
    
    return {'task': 'text-classification', 'input': 'text'}  # Default

def generate_module_template(model_id, model_type):
    """Generate lilypad_module.json.tmpl content"""
    input_vars = []
    
    if model_type['input'] in ['text', 'text-pair']:
        input_vars.append('{{ if .input_text }}"INPUT_TEXT={{ js .input_text }}"{{ end }}')
    
    if model_type['input'] in ['image', 'video', 'audio', 'document-text-pair', 'image-text-pair']:
        input_vars.append('{{ if .input_path }}"INPUT_PATH={{ js .input_path }}"{{ else }}"INPUT_PATH=/workspace/input/default_input"{{ end }}')

    env_vars = ',\n                    '.join(input_vars)

    return f'''{{
    "machine": {{
        "gpu": 1,
        "cpu": 1000,
        "ram": 8000
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
                "GPU": "1"
            }},
            "Timeout": 1800
        }}
    }}
}}'''

def generate_readme(model_id, model_type):
    """Generate README.md content with type-specific instructions"""
    input_instructions = {
        'text': "Provide text directly via --input_text or in input/text.txt file",
        'image': "Place image files (JPG/PNG) in the input directory",
        'audio': "Place audio files (WAV/MP3) in the input directory",
        'video': "Place video files (MP4) in the input directory",
        'image-text-pair': "Provide both image file and text input",
        'document-text-pair': "Provide both PDF document and text input"
    }

    cli_examples = {
        'text': f'python run_inference.py --input_text="Your text here"',
        'image': 'python run_inference.py --image_path=input/image.jpg',
        'audio': 'python run_inference.py --audio_path=input/audio.wav',
        'video': 'python run_inference.py --video_path=input/video.mp4',
        'image-text-pair': 'python run_inference.py --image_path=input/image.jpg --input_text="Your question here"',
        'document-text-pair': 'python run_inference.py --document_path=input/document.pdf --input_text="Your question here"'
    }

    docker_examples = {
        'text': f'docker run -e INPUT_TEXT="Your text here" {model_id.split("/")[-1]}',
        'image': f'docker run -v $(pwd)/input:/workspace/input {model_id.split("/")[-1]}',
        'audio': f'docker run -v $(pwd)/input:/workspace/input {model_id.split("/")[-1]}',
        'video': f'docker run -v $(pwd)/input:/workspace/input {model_id.split("/")[-1]}',
        'image-text-pair': f'docker run -v $(pwd)/input:/workspace/input -e INPUT_TEXT="Your question here" {model_id.split("/")[-1]}',
        'document-text-pair': f'docker run -v $(pwd)/input:/workspace/input -e INPUT_TEXT="Your question here" {model_id.split("/")[-1]}'
    }

    lilypad_examples = {
        'text': f'lilypad run {model_id.split("/")[-1]} -i input_text="Your text here"',
        'image': f'lilypad run {model_id.split("/")[-1]} -i input_path=/workspace/input/image.jpg',
        'audio': f'lilypad run {model_id.split("/")[-1]} -i input_path=/workspace/input/audio.wav',
        'video': f'lilypad run {model_id.split("/")[-1]} -i input_path=/workspace/input/video.mp4',
        'image-text-pair': f'lilypad run {model_id.split("/")[-1]} -i input_path=/workspace/input/image.jpg -i input_text="Your question here"',
        'document-text-pair': f'lilypad run {model_id.split("/")[-1]} -i input_path=/workspace/input/document.pdf -i input_text="Your question here"'
    }

    return f'''# Lilypad Module for {model_id}

This module deploys [{model_id}](https://huggingface.co/{model_id}) from Hugging Face for {model_type['task']} using Lilypad.

## Setup

1. Build the Docker image:
   ```bash
   docker build -t {model_id.split('/')[-1]} .
   docker tag {model_id.split('/')[-1]} username/{model_id.split('/')[-1]}:latest
   docker push username/{model_id.split('/')[-1]}:latest
   ```

2. Test locally:
   ```bash
   # Using Python directly:
   {cli_examples.get(model_type['input'], cli_examples['text'])}

   # Using Docker:
   {docker_examples.get(model_type['input'], docker_examples['text'])}
   ```

3. Run on Lilypad:
   ```bash
   {lilypad_examples.get(model_type['input'], lilypad_examples['text'])}
   ```

## Input Format

- Type: {model_type['input']}
- {input_instructions.get(model_type['input'], input_instructions['text'])}
- Results will be saved to: output/result.json

## Example Output

The model will output JSON in this format:
```json
{{
    "result": <model specific output>,
    "status": "success"
}}
```

## Error Handling

If an error occurs, the output JSON will contain:
```json
{{
    "error": "Error message",
    "status": "error"
}}
```

## System Requirements

- GPU with at least 8GB VRAM
- CUDA support
- Docker installed

## Advanced Configuration

You can modify lilypad_module.json.tmpl to adjust:
- GPU/CPU/RAM requirements
- Timeout duration
- Network access
- Environment variables'''

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
                'document-text-pair': 'Place your document files (PDF) here',
                'image-text-pair': 'Place your image files here'
            }.get(model_type['input'], 'Place input files here')
            
            zf.writestr('input/README.md', input_readme)
        
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