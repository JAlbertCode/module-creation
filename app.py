# Previous imports...
from templates.config_templates import TemplateManager

app = Flask(__name__)
template_manager = TemplateManager()

# Previous routes...

@app.route('/templates', methods=['GET'])
def list_templates():
    """List available configuration templates"""
    model_type = request.args.get('model_type')
    templates = template_manager.list_templates(model_type)
    return jsonify(templates)

@app.route('/templates', methods=['POST'])
def save_template():
    """Save a new configuration template"""
    try:
        data = request.json
        template = template_manager.save_template(
            name=data['name'],
            description=data['description'],
            config=data['config'],
            model_type=data['model_type']
        )
        return jsonify({
            'name': template.name,
            'description': template.description,
            'model_type': template.model_type
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/templates/<name>', methods=['GET'])
def load_template(name):
    """Load a configuration template"""
    try:
        template = template_manager.load_template(name)
        return jsonify({
            'name': template.name,
            'description': template.description,
            'config': template.config,
            'model_type': template.model_type
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/templates/<name>', methods=['DELETE'])
def delete_template(name):
    """Delete a configuration template"""
    try:
        template_manager.delete_template(name)
        return '', 204
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/templates/export/<name>', methods=['GET'])
def export_template(name):
    """Export a template to a file"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp:
            template_manager.export_template(name, temp.name)
            return send_file(
                temp.name,
                mimetype='application/json',
                as_attachment=True,
                download_name=f"{name}_config.json"
            )
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/templates/import', methods=['POST'])
def import_template():
    """Import a template from a file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp:
            file.save(temp.name)
            template = template_manager.import_template(temp.name)
            os.unlink(temp.name)  # Clean up
            
        return jsonify({
            'name': template.name,
            'description': template.description,
            'model_type': template.model_type
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Rest of the file remains the same...