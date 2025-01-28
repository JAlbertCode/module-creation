"""
Web application for converting Hugging Face models to Lilypad modules
"""

from flask import Flask, request, send_file, render_template, jsonify
from huggingface_hub import HfApi
import os
import json
import zipfile
from io import BytesIO
import traceback

from modules import handlers, utils, model_types, validation, config

app = Flask(__name__)

def get_model_info(model_url: str):
    """Extract model information from Hugging Face URL"""
    try:
        model_id = model_url.split('huggingface.co/')[-1].strip('/')
        if not utils.validate_model_id(model_id):
            raise ValueError("Invalid model ID format")
        
        api = HfApi()
        model_info = api.model_info(model_id)
        return model_info
    except Exception as e:
        raise ValueError(f"Failed to fetch model information: {str(e)}")

@app.route('/')
def index():
    """Render the main page"""
    return render_template('input_form.html')

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
        
        # Detect model type and task
        model_type = model_types.detect_model_type(model_info)
        
        # Validate model compatibility
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
        
        # Get model info and detect type
        model_info = get_model_info(model_url)
        model_type = model_types.detect_model_type(model_info)
        
        # Validate model
        validation_result = validation.check_model_compatibility(model_info)
        if not validation_result.is_valid:
            message = validation.format_validation_message(validation_result)
            return message, 400
        
        # Create zip file with module files
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add configuration file
            zf.writestr('config.json', json.dumps(model_configs, indent=2))
            
            # Add Dockerfile
            zf.writestr('Dockerfile', utils.generate_dockerfile(model_type))
            
            # Add requirements.txt
            zf.writestr('requirements.txt', utils.generate_requirements(model_type))
            
            # Get inference code from appropriate handler
            inference_code = handlers.get_inference_code(
                model_type['input'], 
                model_info.id,
                model_type['task']
            )
            zf.writestr('run_inference.py', inference_code)
            
            # Add Lilypad module template
            module_template = config.generate_module_template(
                model_info.id,
                model_type,
                model_configs
            )
            zf.writestr('lilypad_module.json.tmpl', module_template)
            
            # Add README
            readme = utils.create_model_card(
                model_info.id,
                model_type['task'],
                model_info
            )
            zf.writestr('README.md', readme)
            
            # Add example input directory
            input_readme = utils.get_input_description(model_type['task'])
            zf.writestr('input/README.md', input_readme)
            
            # Add test script
            test_script = utils.generate_test_script(model_info.id, model_type)
            zf.writestr('test.sh', test_script)
            
        # Return zip file
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