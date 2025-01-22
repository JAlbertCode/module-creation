# Lilypad Module Generator

A web-based tool that generates Lilypad deployment modules from Hugging Face models. Simply paste a Hugging Face model URL, and get a ready-to-use Lilypad module.

## Features Checklist

- [x] Core Functionality
  - [x] Accept Hugging Face model URL input
  - [x] Detect model type automatically
  - [x] Generate appropriate files
  - [x] Package files into zip
  - [x] Provide clear setup instructions

- [x] Input Method Support
  - [x] Command line interface for text input (--input_text)
  - [x] File system input for images
  - [x] Environment variable support (INPUT_PATH)
  - [x] Flexible input handling based on model type

- [x] File Generation
  - [x] Dockerfile with proper dependencies
  - [x] requirements.txt based on model type
  - [x] run_inference.py with CLI support
  - [x] module.yaml for Lilypad configuration
  - [x] README with usage instructions

- [x] Model Type Support
  - [x] Text Classification
  - [x] Image Classification
  - [x] Object Detection
  - [x] Question Answering

- [x] Error Handling
  - [x] Invalid URL validation
  - [x] Model compatibility check
  - [x] Input validation
  - [x] Clear error messages

- [ ] Future Enhancements
  - [ ] Support for more model types
  - [ ] Custom model configuration options
  - [ ] Batch processing support
  - [ ] Performance optimization presets

## Usage

1. Start the server:
   ```bash
   python app.py
   ```

2. Open in browser:
   ```
   http://localhost:5000
   ```

3. Enter Hugging Face model URL and generate module

4. For generated modules:
   ```bash
   # For text models:
   python run_inference.py --input_text="Your text here"

   # For image models:
   python run_inference.py --image_path=input/image.jpg
   ```

## Generated Files

Each module includes:
- `Dockerfile` - Container configuration
- `requirements.txt` - Python dependencies
- `run_inference.py` - Inference code with CLI support
- `module.yaml` - Lilypad configuration
- `README.md` - Setup instructions
- `input/` - Directory for input files

## Prerequisites

- Python 3.9+
- Flask
- Hugging Face Hub library

## Development

Install dependencies:
```bash
pip install -r requirements.txt
```

## Contributing

Contributions welcome! Please check existing issues or create new ones.

## License

MIT License - See LICENSE file for details
