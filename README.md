# Lilypad Module Generator

A web-based tool that generates Lilypad deployment modules from Hugging Face models.

## Features Checklist

- [x] Core Functionality
  - [x] Model URL input
  - [x] File generation
  - [x] ZIP packaging
  - [x] Download capability

- [x] Model Support
  - [x] Text Classification
  - [x] Image Classification
  - [x] Object Detection
  - [x] Question Answering
  - [x] Automatic model type detection

- [x] File Generation
  - [x] Dockerfile
  - [x] requirements.txt
  - [x] run_inference.py
  - [x] module.yaml
  - [x] README.md

- [x] Preview Features
  - [x] Model information display
  - [x] Sample input/output preview
  - [x] File content preview
  - [x] Markdown rendering

- [x] Error Handling
  - [x] URL validation
  - [x] Model compatibility check
  - [x] User-friendly error messages
  - [x] Preview error handling

- [ ] Additional Features
  - [ ] Custom configuration options
  - [ ] Interactive testing interface
  - [ ] Multiple model support
  - [ ] Batch processing

## Quick Start

1. Start the server:
   ```bash
   python app.py
   ```

2. Open in browser:
   ```
   http://localhost:5000
   ```

3. Use the interface:
   - Paste a Hugging Face model URL
   - Preview the model (optional)
   - Generate and download the module
   - Follow the setup instructions

## Supported Models

- Text Classification Models
- Image Classification Models
- Object Detection Models
- Question Answering Models
- And more (automatically detected)

## Generated Files

Each generated module includes:
- `Dockerfile` - Container configuration
- `requirements.txt` - Python dependencies
- `run_inference.py` - Model inference code
- `module.yaml` - Lilypad configuration
- `README.md` - Setup instructions

## Usage Example

```bash
# After downloading and extracting the module:

# Build the Docker image
docker build -t my-model .

# Run locally
docker run -v $(pwd)/input:/workspace/input \
          -e INPUT_PATH=/workspace/input/input.txt \
          my-model

# Deploy to Lilypad
lilypad module deploy .
```

## Development

### Prerequisites
- Python 3.9+
- Flask
- Hugging Face Hub library

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
python app.py
```

## Contributing

Contributions are welcome! See CONTRIBUTING.md for guidelines.

## License

MIT License - See LICENSE file for details