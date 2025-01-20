# Image Classification Module for Lilypad

This repository contains a generalized image classification module that can deploy and run various Hugging Face image classification models on Lilypad.

## Setup Checklist

- [x] Create basic repository structure
  - [x] Create Dockerfile
  - [x] Create requirements.txt
  - [x] Create run_inference.py script
  - [x] Create module.yaml for Lilypad configuration
  
- [x] Implement core functionality
  - [x] Add image loading and preprocessing
  - [x] Add model loading from Hugging Face
  - [x] Add inference pipeline
  - [x] Add proper error handling
  - [x] Add output formatting
  
- [x] Add support for common image models
  - [x] ResNet support
  - [x] ViT support
  - [x] DeiT support
  - [x] ConvNeXT support
  
- [x] Create documentation
  - [x] Add usage examples
  - [x] Add API documentation
  - [x] Add troubleshooting guide
  
- [x] Testing
  - [x] Test with different image formats
  - [x] Test with different model architectures
  - [x] Test error cases
  - [x] Test on Lilypad network

## Setup Instructions

1. First-time setup:
   ```bash
   # Clone the repository
   git clone [repository-url]
   cd Image-Classification

   # Create a Python virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

2. Testing the Module:
   ```bash
   # Run the test suite
   python run_tests.py
   
   # View test logs
   cat test_logs/test_run_*.log
   
   # Run Lilypad integration tests
   python -m pytest tests/test_lilypad_integration.py
   ```

3. Build the Docker image:
   ```bash
   docker build -t image-classification .
   ```

4. Run locally for testing:
   ```bash
   docker run -e MODEL_ID=microsoft/resnet-50 -e INPUT_PATH=/path/to/image.jpg image-classification
   ```

5. Deploy to Lilypad:
   ```bash
   lilypad module deploy .
   ```

## Usage

To use this module, you'll need to:

1. Prepare your image file
2. Choose a Hugging Face image classification model
3. Set up the required environment variables:
   - `MODEL_ID`: The Hugging Face model ID (e.g., "microsoft/resnet-50")
   - `INPUT_PATH`: Path to the input image
   - `OUTPUT_PATH`: (Optional) Path where results should be saved

## Expected Input Format

- Supported image formats: JPG, PNG, WebP
- Maximum image size: 1024x1024 pixels (larger images will be automatically resized)
- Images should be in RGB format

## Output Format

The module will output a JSON file containing:
```json
{
    "model_id": "microsoft/resnet-50",
    "predictions": [
        {
            "label": "predicted_class",
            "score": 0.95
        }
    ],
    "status": "success",
    "processing_time": "1.23s"
}
```

## Testing

The module includes comprehensive tests:
- Image format support (JPG, PNG, WebP)
- Model architecture compatibility
- Error handling scenarios
- Resource requirement validation
- Lilypad network integration

Run the test suite using:
```bash
python run_tests.py
```

Test logs are stored in the `test_logs` directory.

## Continuous Integration

The repository includes GitHub Actions workflows for:
- Automated testing
- Code coverage reporting
- Integration testing with Lilypad network

## Error Handling

The module includes robust error handling for:
- Invalid image formats
- Network issues during model download
- Memory constraints
- Invalid model architectures
- Lilypad deployment issues

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting a pull request.

## License

MIT License - See LICENSE file for details