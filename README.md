# Image Classification Module for Lilypad

This repository contains a generalized image classification module that can deploy and run various Hugging Face image classification models on Lilypad.

## Setup Checklist

- [ ] Create basic repository structure
  - [ ] Create Dockerfile
  - [ ] Create requirements.txt
  - [ ] Create run_inference.py script
  - [ ] Create module.yaml for Lilypad configuration
  
- [ ] Implement core functionality
  - [ ] Add image loading and preprocessing
  - [ ] Add model loading from Hugging Face
  - [ ] Add inference pipeline
  - [ ] Add proper error handling
  - [ ] Add output formatting
  
- [ ] Add support for common image models
  - [ ] ResNet support
  - [ ] ViT support
  - [ ] DeiT support
  - [ ] ConvNeXT support
  
- [ ] Create documentation
  - [ ] Add usage examples
  - [ ] Add API documentation
  - [ ] Add troubleshooting guide
  
- [ ] Testing
  - [ ] Test with different image formats
  - [ ] Test with different model architectures
  - [ ] Test error cases
  - [ ] Test on Lilypad network

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

2. Prepare your environment:
   - Ensure you have Docker installed
   - Have your Lilypad network credentials ready
   - Have the Hugging Face model ID you want to use

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

## Error Handling

The module includes robust error handling for:
- Invalid image formats
- Network issues during model download
- Memory constraints
- Invalid model architectures

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting a pull request.

## License

MIT License - See LICENSE file for details