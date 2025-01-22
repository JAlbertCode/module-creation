# Lilypad Module Generator

A web-based tool that generates Lilypad deployment modules from Hugging Face models. Simply paste a Hugging Face model URL, and get a ready-to-use Lilypad module.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/JAlbertCode/module-creation.git
   cd module-creation
   ```

2. Create and activate virtual environment:
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate on Mac/Linux
   source venv/bin/activate

   # Activate on Windows
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Open in browser:
   ```
   http://localhost:5000
   ```

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

## Using Generated Modules

After downloading and extracting the generated module:

1. Navigate to module directory:
   ```bash
   cd lilypad-your-model-name
   ```

2. Run inference:
   ```bash
   # For text models:
   python run_inference.py --input_text="Your text here"

   # For image models:
   python run_inference.py --image_path=input/image.jpg
   ```

## Required Files

Your directory structure should look like this:
```
module-creation/
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
└── README.md
```

## Troubleshooting

Common issues:

1. "No such file or directory":
   - Make sure you're in the correct directory (module-creation)
   - Verify all files are present using `ls` or `dir`

2. Module not found errors:
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt` again

3. Port already in use:
   - Change port in app.py to something else (e.g., 5001)
   - Kill process using current port

## Contributing

Contributions welcome! Please check existing issues or create new ones.

## License

MIT License - See LICENSE file for details