from flask import Flask, request, send_file, render_template, jsonify
from huggingface_hub import HfApi
import os
import json
import zipfile
from io import BytesIO
import tempfile
from config_handler import ConfigHandler
from error_handler import validate_model_url, handle_error, ModelGenerationError

app = Flask(__name__)

MODEL_PRESETS = {
    'efficient': {
        'resources': {'cpu': 1, 'memory': '2Gi', 'gpu': None},
        'model': {'batch_size': 1},
        'description': 'Minimum resource usage, suitable for testing'
    },
    'balanced': {
        'resources': {'cpu': 2, 'memory': '4Gi', 'gpu': None},
        'model': {'batch_size': 4},
        'description': 'Balanced performance and resource usage'
    },
    'performance': {
        'resources': {'cpu': 4, 'memory': '8Gi', 'gpu': 1},
        'model': {'batch_size': 8},
        'description': 'Maximum performance, requires more resources'
    }
}

def get_model_info(model_url):
    """Extract model information from Hugging Face URL"""
    model_id = model_url.split('huggingface.co/')[-1].strip('/')
    api = HfApi()
    return api.model_info(model_id)

def detect_model_type(model_info):
    """Detect the model type from model info"""
    pipeline_mapping = {
        'text-classification': {
            'task': 'text-classification',
            'input_type': 'text',
            'example': 'This movie was fantastic!'
        },
        'image-classification': {
            'task': 'image-classification',
            'input_type': 'image',
            'example': 'image.jpg'
        },
        'object-detection': {
            'task': 'object-detection',
            'input_type': 'image',
            'example': 'image.jpg'
        },
        'question-answering': {
            'task': 'question-answering',
            'input_type': 'json',
            'example': '{"question": "What is...", "context": "..."}'
        }
    }
    
    for tag in model_info.tags:
        if tag in pipeline_mapping:
            return pipeline_mapping[tag]
    
    return pipeline_mapping['text-classification']

def create_zip_file(model_info, config):
    """Create zip file containing all necessary files"""
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        model_type = detect_model_type(model_info)
        config_handler = ConfigHandler(model_info, model_type)
        
        # Validate and update configuration
        validated_config = config_handler.update_config(config)
        config_handler.validate_config(validated_config)
        
        # Generate files with configuration
        files = {
            'Dockerfile': generate_dockerfile(validated_config),
            'requirements.txt': generate_requirements(model_info, validated_config),
            'run_inference.py': generate_run_inference(model_info.id, model_type, validated_config),
            'module.yaml': config_handler.generate_yaml(validated_config),
            'README.md': generate_readme(model_info.id, model_type, validated_config)
        }
        
        # Add files to zip
        for filename, content in files.items():
            zf.writestr(filename, content)
        
        # Add example input
        add_example_input(zf, model_type)
    
    memory_file.seek(0)
    return memory_file

def add_example_input(zf, model_type):
    """Add appropriate example input files"""
    input_dir = 'input/'
    if model_type['input_type'] == 'text':
        zf.writestr(f"{input_dir}input.txt", 'Example input text')
    elif model_type['input_type'] == 'json':
        example = {
            'question': 'What is machine learning?',
            'context': 'Machine learning is a subset of artificial intelligence...'
        }
        zf.writestr(f"{input_dir}input.json", json.dumps(example, indent=2))
    else:  # image
        zf.writestr(f"{input_dir}README.md", 'Place your images here (JPEG, PNG, or WebP format)')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/config/defaults', methods=['POST'])
def get_config_defaults():
    """Get default configuration for a model"""
    try:
        data = request.json
        model_url = data.get('model_url')
        validate_model_url(model_url)
        
        model_info = get_model_info(model_url)
        model_type = detect_model_type(model_info)
        
        config_handler = ConfigHandler(model_info, model_type)
        default_config = config_handler.generate_base_config()
        
        return jsonify({
            'config': default_config,
            'presets': MODEL_PRESETS
        })
        
    except Exception as e:
        return jsonify({'error': handle_error(e)}), 400

@app.route('/config/validate', methods=['POST'])
def validate_config():
    """Validate a configuration"""
    try:
        data = request.json
        model_url = data.get('model_url')
        config = data.get('config')
        
        validate_model_url(model_url)
        model_info = get_model_info(model_url)
        model_type = detect_model_type(model_info)
        
        config_handler = ConfigHandler(model_info, model_type)
        config_handler.validate_config(config)
        
        return jsonify({'status': 'valid'})
        
    except Exception as e:
        return jsonify({'error': handle_error(e)}), 400

@app.route('/generate', methods=['POST'])
def generate():
    """Generate and download the module package"""
    try:
        model_url = request.form.get('model_url')
        config = json.loads(request.form.get('config', '{}'))
        
        validate_model_url(model_url)
        model_info = get_model_info(model_url)
        
        zip_file = create_zip_file(model_info, config)
        
        return send_file(
            zip_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'lilypad-{model_info.id.split("/")[-1]}.zip'
        )
        
    except Exception as e:
        return str(handle_error(e)), 500

@app.route('/preview', methods=['POST'])
def preview_model():
    """Preview model with sample input"""
    try:
        model_url = request.form.get('model_url')
        validate_model_url(model_url)
        
        model_info = get_model_info(model_url)
        model_type = detect_model_type(model_info)
        
        return jsonify({
            'model_info': {
                'name': model_info.id,
                'type': model_type['task'],
                'input_format': model_type['input_type']
            },
            'sample_input': model_type['example'],
            'configuration': ConfigHandler(model_info, model_type).generate_base_config()
        })
        
    except Exception as e:
        return jsonify({'error': handle_error(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)