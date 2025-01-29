# Project Structure

This document outlines the organization and purpose of each component in the Hugging Face to Lilypad module converter.

## Directory Structure

```
modules/
├── handlers/              # Core handlers for different model types
│   ├── __init__.py       
│   ├── base.py           # Base handler class with common functionality
│   ├── text.py           # Text model handlers (classification, generation, etc.)
│   ├── image.py          # Image model handlers (classification, detection, etc.)
│   ├── audio.py          # Audio model handlers (speech, music, etc.)
│   ├── video.py          # Video model handlers (generation, understanding)
│   ├── multimodal.py     # Multi-input model handlers (vision-language, etc.)
│   └── structured.py     # Structured data handlers (graphs, point clouds)
│
├── types/                # Type definitions and model categorization
│   ├── __init__.py
│   ├── model_types.py    # Model type detection and configuration
│   └── task_types.py     # Task-specific configurations
│
├── utils/                # Utility functions and helpers
│   ├── __init__.py
│   ├── validation.py     # Model compatibility validation
│   ├── config.py         # Configuration management
│   └── io_utils.py       # Input/output utilities
│
└── __init__.py

templates/              # HTML templates for web interface
├── index.html
└── input_form.html

static/                # Static assets for web interface
├── css/
└── js/
```

## Key Components

### Handlers

Each handler in the `modules/handlers/` directory follows a consistent class-based structure:

1. **BaseHandler** (`base.py`):
   - Common functionality for all handlers
   - Input/output methods
   - Error handling
   - Resource management

2. **Specialized Handlers**:
   - Text: Text processing models (BERT, GPT, etc.)
   - Image: Image processing models (ResNet, YOLO, etc.)
   - Audio: Audio processing models (Wav2Vec, etc.)
   - Video: Video processing models (TimeSformer, etc.)
   - Multimodal: Multi-input models (CLIP, etc.)
   - Structured: Graph and point cloud models

Each handler implements:
- `generate_imports()`: Required import statements
- `generate_inference()`: Model-specific inference code
- `generate_setup()`: Environment setup code
- `generate_output_handling()`: Output processing code

### Types

1. **model_types.py**:
   - Model type detection
   - Input/output configurations
   - Hardware requirements
   - Framework dependencies

2. **task_types.py**:
   - Task-specific configurations
   - Default parameters
   - Evaluation metrics

### Utils

1. **validation.py**:
   - Model compatibility checking
   - Hardware requirement validation
   - Input format validation

2. **config.py**:
   - Configuration management
   - Environment variables
   - Default settings

3. **io_utils.py**:
   - File handling
   - Data format conversion
   - Input/output validation

## Web Interface

The web interface provides a user-friendly way to convert Hugging Face models to Lilypad modules:

1. **Templates**:
   - `index.html`: Landing page
   - `input_form.html`: Model conversion form

2. **Static Assets**:
   - CSS: Styling
   - JavaScript: Form handling and validation

## Usage

1. **Direct Usage**:
```python
from modules.handlers import TextHandler, ImageHandler
from modules.types import detect_model_type
from modules.utils import validate_model

# Create appropriate handler
model_type = detect_model_type(model_info)
handler = TextHandler(model_id, task) if model_type == 'text' else ImageHandler(model_id, task)

# Generate module files
inference_code = handler.generate_inference()
```

2. **Web Interface**:
```bash
python app.py  # Run Flask server
# Navigate to http://localhost:5000
```

## Adding New Handlers

To add support for a new model type:

1. Create a new handler class in `modules/handlers/`
2. Implement required methods (see BaseHandler)
3. Add type detection in `model_types.py`
4. Add validation rules in `validation.py`
5. Update configuration options in `config.py`

## Dependencies

Dependencies are managed at two levels:

1. **Project Dependencies** (`requirements.txt`):
   - Core dependencies for the converter

2. **Generated Module Dependencies**:
   - Each handler specifies model-specific dependencies
   - Generated in module's requirements.txt

## Testing

Test files follow the same structure as source files:

```
tests/
├── handlers/
│   ├── test_text.py
│   ├── test_image.py
│   └── ...
├── types/
│   └── test_model_types.py
└── utils/
    └── test_validation.py
```