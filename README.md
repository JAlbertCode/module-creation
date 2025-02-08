# Hugging Face to Lilypad Module Converter

This tool helps you convert any Hugging Face model into a Lilypad module that can be run on the Lilypad network. It automatically handles:

- Model validation and compatibility checking
- Generation of Dockerfile and requirements
- Creation of inference code
- Configuration of Lilypad module template
- Documentation generation

## Current Status

### Completed Features
- ✅ Model type detection system
- ✅ Basic validation framework
- ✅ Template system with Jinja2
- ✅ Download and caching manager
- ✅ Project structure and organization
- ✅ Handlers implementation:
  - Text models (classification, generation)
  - Image models (classification, generation)
  - Vision-Language models (UI understanding, instruction models)
  - Audio models (speech recognition, text-to-speech, classification)
  - Video models (classification, text-to-video generation, captioning)
  - Specialized vision models (detection, segmentation, depth estimation, pose)
  - Multimodal models (VQA, document understanding)
  - Point cloud models (classification, segmentation)
- ✅ Templates for model tasks:
  - Text generation and classification
  - Image generation and classification
  - Audio processing (ASR, TTS, classification)
  - Video processing (classification, generation, captioning)
  - Specialized vision (object detection, segmentation, depth, pose)
  - Multimodal (visual question answering, document QA)
  - Point cloud processing (classification, segmentation)
- ✅ Command-line interface:
  - Model conversion
  - Docker image building
  - Module testing

### In Progress
- 🟡 Graph neural network support
- 🟡 Time series model support
- 🟡 Model type implementations:
  - Pose estimation
  - Multi-modal tasks
  - Point cloud processing
- 🟡 Input/output validation for each task type
- 🟡 Error handling and progress reporting
- 🟡 Testing framework setup and initial tests

### Upcoming Tasks
- 📝 Web interface development
- 📝 Additional specialized task support:
  - Point cloud processing
  - Graph neural networks
  - Custom model architectures
- 📝 Automated testing and CI/CD:
  - Unit tests for all components
  - Integration tests for module generation
  - Docker build testing
  - Performance benchmarking
- 📝 Documentation generation system:
  - API documentation
  - Usage guides
  - Model compatibility matrix
  - Best practices guide
- 📝 Additional Features:
  - Model fine-tuning support
  - Custom preprocessing pipelines
  - Batch processing support
  - Resource usage optimization
  - Model quantization options

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

### Command Line Interface

Convert a Hugging Face model to a Lilypad module:
```bash
python cli.py convert bert-base-uncased --output ./my-module
```

Build a Docker image for the module:
```bash
python cli.py build ./my-module --image-name my-model:latest
```

Run module tests:
```bash
python cli.py test ./my-module
```

### Web Interface

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

## Development

To modify or extend the converter:

1. Model Types: Add new model types in `modules/model_types.py`
2. Handlers: Add input/output handlers in `modules/handlers/`
3. Validation: Add validation rules in `modules/utils/validation.py`
4. Configuration: Modify module configs in `modules/utils/config.py`
5. Templates: Update templates in `templates/`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Here are some ways you can help:

1. Add support for new model types
2. Create templates for specialized tasks
3. Improve validation and error handling
4. Enhance documentation
5. Add examples and test cases
6. Report bugs and suggest features

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co) for their amazing models and APIs
- [Lilypad Network](https://lilypad.tech) for the decentralized compute platform
- All the open source libraries used in this project