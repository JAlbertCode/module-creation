from flask import Flask, request, jsonify, send_file, render_template
from huggingface_hub import HfApi
import os
import json
import zipfile
from io import BytesIO
import tempfile
import docker
from model_templates import ModelTemplate
from config_validator import ConfigValidator

app = Flask(__name__)

def get_model_info(model_url):
    """Extract model information from Hugging Face URL"""
    model_id = model_url.split('huggingface.co/')[-1].strip('/')
    api = HfApi()
    return api.model_info(model_id)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_model():
    """Analyze the model and return its characteristics"""
    try:
        model_url = request.json.get('modelUrl')
        if not model_url:
            return jsonify({'error': 'No model URL provided'}), 400

        model_info = get_model_info(model_url)
        template = ModelTemplate(model_info)

        return jsonify({
            'modelId': model_info.id,
            'pipelineType': template.pipeline_type,
            'resources': template.resources,
            'description': model_info.description,
            'tags': model_info.tags
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate_files():
    """Generate module files based on model information"""
    try:
        model_url = request.json.get('modelUrl')
        if not model_url:
            return jsonify({'error': 'No model URL provided'}), 400

        model_info = get_model_info(model_url)
        template = ModelTemplate(model_info)

        files = {
            'Dockerfile': template.generate_dockerfile(),
            'requirements.txt': template.generate_requirements(),
            'run_inference.py': template.generate_run_inference(),
            'module.yaml': template.generate_module_yaml(),
            'README.md': template.generate_readme()
        }

        # Validate configurations
        validator = ConfigValidator()
        validation_results = validator.validate_all(files)

        return jsonify({
            'files': files,
            'validation': validation_results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test', methods=['POST'])
def test_module():
    """Test the generated module locally with provided input"""
    try:
        data = request.json
        model_url = data.get('modelUrl')
        test_input = data.get('input')

        if not model_url or not test_input:
            return jsonify({'error': 'Missing required parameters'}), 400

        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate files
            model_info = get_model_info(model_url)
            template = ModelTemplate(model_info)

            # Write files
            files = {
                'Dockerfile': template.generate_dockerfile(),
                'requirements.txt': template.generate_requirements(),
                'run_inference.py': template.generate_run_inference(),
                'input.txt': test_input
            }

            for filename, content in files.items():
                with open(os.path.join(tmpdir, filename), 'w') as f:
                    f.write(content)

            # Build and run Docker container
            client = docker.from_env()
            image_tag = f"lilypad-test-{model_info.id.split('/')[-1]}"

            # Build image
            image, _ = client.images.build(
                path=tmpdir,
                tag=image_tag,
                rm=True
            )

            # Run container
            container = client.containers.run(
                image_tag,
                environment={
                    'INPUT_PATH': '/workspace/input.txt'
                },
                volumes={
                    tmpdir: {
                        'bind': '/workspace',
                        'mode': 'rw'
                    }
                },
                detach=True
            )

            # Wait for container to finish and get logs
            container.wait()
            logs = container.logs().decode('utf-8')

            # Read results
            output_file = os.path.join(tmpdir, 'outputs', 'result.json')
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    results = json.load(f)
            else:
                results = {'error': 'No output generated'}

            # Cleanup
            container.remove()

            return jsonify({
                'results': results,
                'logs': logs
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download', methods=['POST'])
def download_files():
    """Generate and download zip file with all module files"""
    try:
        model_url = request.json.get('modelUrl')
        if not model_url:
            return jsonify({'error': 'No model URL provided'}), 400

        model_info = get_model_info(model_url)
        template = ModelTemplate(model_info)

        # Create zip file in memory
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            files = {
                'Dockerfile': template.generate_dockerfile(),
                'requirements.txt': template.generate_requirements(),
                'run_inference.py': template.generate_run_inference(),
                'module.yaml': template.generate_module_yaml(),
                'README.md': template.generate_readme()
            }

            for filename, content in files.items():
                zf.writestr(filename, content)

        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'lilypad-{model_info.id.split("/")[-1]}.zip'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
