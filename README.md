# Hugging Face to Lilypad Module Converter

Universal converter that transforms any Hugging Face model into a Lilypad module.

## Documentation Structure

This project maintains two key documents:

1. README.md (this file)
   - Project overview
   - Current capabilities
   - Implementation status
   - Basic usage
   - Project structure

2. SETUP.md
   - Detailed installation steps
   - Environment configuration
   - Troubleshooting guides

Both files should be kept in sync when making changes. When adding new model support or features:
1. Update Implementation Status in README.md
2. Update relevant setup instructions in SETUP.md if needed

## Current Capabilities

- Analyzes any model to detect:
  - Input/output types
  - Hardware requirements
  - Dependencies
  - Model configuration

- Generates required files:
  - Dockerfile with dependencies
  - Inference code 
  - Module configuration
  - Requirements file

## Implementation Status

✅ Done:
- Universal model analysis framework
- Text generation models (GPT, LLaMA)
- Text classification (BERT, RoBERTa)
- Token classification models
- Question answering models
- Summarization models (BART, T5)
- Text-to-Image models (Stable Diffusion)

🏗️ In Progress:
- Vision Models:
  - Image classification
  - Object detection
  - Image segmentation
- Audio Models:
  - Speech recognition
  - Audio classification
- Video Models
- Multimodal Models

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Convert a model
python cli.py convert MODEL_ID --output ./modules

# Run on Lilypad
lilypad run ./modules/MODEL_NAME
```

## For Contributors

1. Add Model Support:
- Add architecture mappings in modules/analyzer.py
- Add input/output processors
- Update tests for new model types

2. Track Progress:
- Update Implementation Status section above
- Add test cases
- Update architecture mappings

3. Code Guidelines:
- Universal over custom solutions
- Follow existing patterns
- Add thorough tests

## Testing

```bash
# Run tests
pytest tests/

# Test specific component
pytest tests/test_analyzer.py
```

## Project Structure

```
/
├── modules/           # Core functionality
│   ├── analyzer.py    # Model analysis
│   └── utils/        # Utilities
├── templates/        # Templates
└── tests/           # Test suite
```