# Hugging Face to Lilypad Module Converter

This tool helps you convert any Hugging Face model into a Lilypad module that can be run on the Lilypad network. It automatically handles:

- Model validation and compatibility checking
- Generation of Dockerfile and requirements
- Creation of inference code
- Configuration of Lilypad module template
- Documentation generation

## Features

- Support for multiple model types:
  - Text models (classification, generation, translation)
  - Image models (classification, detection, segmentation)
  - Audio models (speech recognition, text-to-speech)
  - Video models (classification)
  - Multi-modal models (VQA, document QA)
  - Specialized models (point clouds, graphs)

- Automatic detection of:
  - Model task and architecture
  - Hardware requirements
  - Framework dependencies
  - Input/output formats

- Generated module includes:
  - Dockerfile with all dependencies
  - Inference code tailored to model type
  - Lilypad module configuration
  - Documentation and examples
  - Test scripts

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/huggingface-lilypad-converter.git
cd huggingface-lilypad-converter
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the web interface:
```bash
python app.py
```

2. Open your browser and navigate to http://localhost:5000

3. Enter the Hugging Face model URL (e.g., https://huggingface.co/bert-base-uncased)

4. Click "Validate" to check model compatibility

5. Configure model settings if needed

6. Click "Generate" to create the Lilypad module

7. Download and extract the generated zip file

8. Follow the instructions in the generated README.md to test and deploy your module

## Command Line Interface

Coming soon!

## Development

To modify or extend the converter:

1. Model Types: Add new model types in `modules/model_types.py`
2. Handlers: Add input/output handlers in `modules/handlers.py`
3. Validation: Add validation rules in `modules/validation.py`
4. Configuration: Modify module configs in `modules/config.py`
5. Templates: Update UI templates in `templates/`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co) for their amazing models and APIs
- [Lilypad Network](https://lilypad.tech) for the decentralized compute platform
- All the open source libraries used in this project