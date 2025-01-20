from flask import Flask, request, send_file, render_template, jsonify
from huggingface_hub import HfApi
import os
import json
import zipfile
from io import BytesIO
import tempfile
import torch
from transformers import pipeline
from error_handler import validate_model_url, handle_error, ModelGenerationError

app = Flask(__name__)

# Previous functions remain the same...
[Previous functions from app.py]

@app.route('/preview', methods=['POST'])
def preview_model():
    """Preview model with sample input"""
    try:
        model_url = request.form.get('model_url')
        validate_model_url(model_url)
        model_info = get_model_info(model_url)
        model_type = detect_model_type(model_info)
        
        # Get sample input based on model type
        sample_input = get_sample_input(model_type)
        
        # Run quick inference
        device = -1  # Use CPU for preview
        pipe = pipeline(model_type['task'], model=model_info.id, device=device)
        result = pipe(sample_input)
        
        return jsonify({
            'model_info': {
                'name': model_info.id,
                'type': model_type['task'],
                'input_format': model_type['input_type'],
            },
            'sample_input': sample_input,
            'sample_output': result
        })
        
    except Exception as e:
        return jsonify({'error': handle_error(e)}), 400

def get_sample_input(model_type):
    """Get appropriate sample input based on model type"""
    samples = {
        'text-classification': "This is an amazing example of how to use transformer models!",
        'image-classification': None,  # We'll handle images separately
        'object-detection': None,
        'question-answering': {
            'question': "What are transformer models?",
            'context': "Transformer models are neural networks that specialize in processing sequential data."
        }
    }
    
    return samples.get(model_type['task'], "Sample input text")

@app.route('/files', methods=['POST'])
def preview_files():
    """Preview generated files without downloading"""
    try:
        model_url = request.form.get('model_url')
        validate_model_url(model_url)
        model_info = get_model_info(model_url)
        model_type = detect_model_type(model_info)
        
        # Generate files content
        files = {
            'Dockerfile': generate_dockerfile(),
            'requirements.txt': generate_requirements(model_info, model_type),
            'run_inference.py': generate_run_inference(model_info.id, model_type),
            'module.yaml': generate_module_yaml(model_info.id, model_type),
            'README.md': generate_readme(model_info.id, model_type)
        }
        
        return jsonify(files)
        
    except Exception as e:
        return jsonify({'error': handle_error(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)