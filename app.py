"""Simple web application to generate Lilypad modules from Hugging Face models"""

from flask import Flask, request, send_file, render_template, jsonify
from huggingface_hub import HfApi
import os
import json
import zipfile
from io import BytesIO
import traceback

from modules import handlers, utils, model_types, validation, config

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
    """Validate a model and return configuration options"""
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
        
        # Get available configurations
        configurations = config.get_model_configs(model_type)
        
        return jsonify({
            'valid': validation_result.is_valid,
            'message': message,
            'model_type': model_type,
            'requirements': validation_result.requirements,
            'input_requirements': input_requirements,
            'configurations': configurations
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
        
        # Get model configurations from form
        model_configs = {}
        for key, value in request.form.items():
            if key.startswith('config_'):
                model_configs[key[7:]] = value
        
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
            # Add configuration file
            zf.writestr('config.json', json.dumps(model_configs, indent=2))
            
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
            
            # Add Lilypad module template
            zf.writestr('lilypad_module.json.tmpl', generate_module_template(model_info.id, model_type, model_configs))
            
            # Add README with configuration info
            zf.writestr('README.md', generate_readme(model_info.id, model_type, validation_result, model_configs))
            
            # Add example input directory
            input_readme = {
                'text': 'Place text files here',
                'image': 'Place image files (JPG/PNG) here',
                'audio': 'Place audio files (WAV/MP3) here',
                'video': 'Place video files (MP4) here'
            }.get(model_type['input'], 'Place input files here')
            zf.writestr('input/README.md', input_readme)
            
            # Add test script
            test_script = f'''#!/bin/bash
# Build and test the module
docker build -t {model_info.id.split("/")[-1]} .
docker run -v $(pwd)/input:/workspace/input {model_info.id.split("/")[-1]}'''
            zf.writestr('test.sh', test_script)
            
        memory_file.seek(0)
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'lilypad-{model_info.id.split("/")[-1]}.zip'
        )
        
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error: {error_traceback}")
        return f"Error: {str(e)}\n\nTraceback: {error_traceback}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)