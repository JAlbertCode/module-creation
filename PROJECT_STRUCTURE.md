# Project Structure

This document outlines the organization and purpose of each component in the Hugging Face to Lilypad module converter.

## Directory Structure

```
modules/
├── handlers/              # Core handlers for different model types
│   ├── __init__.py       
│   ├── base.py           # Base handler class with common functionality
│   ├── text.py           # Text processing (classification, generation, QA)
│   ├── image.py          # Image processing (classification, detection, generation)
│   ├── audio.py          # Audio processing (ASR, TTS, diarization)
│   ├── video.py          # Video processing (generation, understanding)
│   ├── multimodal.py     # Multi-input models (CLIP, VL models)
│   ├── structured.py     # Graph and table processing
│   ├── point_cloud.py    # 3D point cloud processing
│   └── time_series.py    # Time series data processing
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

## Handler Classes

The handler system uses inheritance to provide consistent functionality across different model types:

### Base Handler

**BaseHandler** (`base.py`):
- Abstract base class that defines common interface
- Core functionalities:
  - Model initialization and setup
  - Environment configuration
  - Error handling
  - Resource management
  - Output formatting

### Specialized Handlers

1. **TextHandler** (`text.py`):
   - Tasks:
     - Text Classification
     - Text Generation
     - Translation
     - Summarization
     - Question Answering
     - Token Classification (NER, POS)
   - Models: BERT, GPT, T5, RoBERTa

2. **ImageHandler** (`image.py`):
   - Tasks:
     - Image Classification
     - Object Detection
     - Semantic Segmentation
     - Image Generation
     - Image-to-Image Translation
     - Visual Question Answering
   - Models: ResNet, YOLO, ViT, Stable Diffusion

3. **AudioHandler** (`audio.py`):
   - Tasks:
     - Speech Recognition (ASR)
     - Text-to-Speech (TTS)
     - Speaker Diarization
     - Audio Classification
     - Voice Conversion
   - Models: Wav2Vec2, Whisper, HuBERT

4. **VideoHandler** (`video.py`):
   - Tasks:
     - Video Classification
     - Action Recognition
     - Video Generation
     - Video Captioning
     - Object Tracking
   - Models: TimeSformer, VideoMAE

5. **MultimodalHandler** (`multimodal.py`):
   - Tasks:
     - Vision-Language Processing
     - Audio-Visual Understanding
     - Document Analysis
     - Multi-sensor Fusion
   - Models: CLIP, LayoutLM, AudioCLIP

6. **StructuredHandler** (`structured.py`):
   - Tasks:
     - Graph Classification
     - Node Classification
     - Link Prediction
     - Table Understanding
   - Models: GNN, GAT, GraphSAGE

7. **PointCloudHandler** (`point_cloud.py`):
   - Tasks:
     - Point Cloud Classification
     - Object Detection in 3D
     - Point Cloud Segmentation
     - Point Cloud Registration
   - Models: PointNet, PointNet++

8. **TimeSeriesHandler** (`time_series.py`):
   - Tasks:
     - Time Series Classification
     - Forecasting
     - Anomaly Detection
     - Pattern Recognition
   - Models: Transformer-Time, TimesNet

Each handler implements:
- `generate_imports()`: Required import statements
- `generate_inference()`: Task-specific inference code
- `get_requirements()`: Package dependencies
- `requires_gpu()`: GPU requirements
- Custom methods for specialized tasks

## Type System

The type system helps identify and configure models:

### Model Types

`model_types.py` defines:
- Model requirements (GPU, RAM, packages)
- Framework detection (PyTorch, TensorFlow)
- Model capabilities and limitations

### Task Types

`task_types.py` defines:
- Task categories and requirements
- Input/output specifications
- Default parameters
- Evaluation metrics

## Utilities

### Validation

`validation.py` provides:
- Model compatibility checking
- Hardware requirement validation
- Input format validation

### Configuration

`config.py` handles:
- Module configuration
- Environment variables
- Docker configuration
- Default settings

### I/O Utilities

`io_utils.py` provides:
- File handling for all data types
- Format conversion
- Batch processing
- Caching

## Usage Examples

1. Basic Usage:
```python
from modules.handlers import TextHandler
from modules.types import detect_model_type
from modules.utils import validate_model

# Create handler
model_type = detect_model_type(model_info)
handler = TextHandler(model_id="bert-base-uncased", task="text-classification")

# Generate module
inference_code = handler.generate_code()
```

2. With Configuration:
```python
from modules.utils.config import ModuleConfig

# Create configuration
config = ModuleConfig(model_type)
config.save_config("/path/to/output")
```

## Adding New Handlers

To add support for a new model type:

1. Create new handler class:
```python
from modules.handlers.base import BaseHandler

class NewHandler(BaseHandler):
    def generate_imports(self) -> str:
        # Add required imports
        
    def generate_inference(self) -> str:
        # Add inference code
```

2. Add type detection in `model_types.py`
3. Add task configuration in `task_types.py`
4. Add validation rules in `validation.py`

## Dependencies

Dependencies are managed at two levels:

1. Project Dependencies:
   - Core requirements in `requirements.txt`
   - Development requirements in `requirements-dev.txt`

2. Generated Module Dependencies:
   - Model-specific requirements
   - System dependencies
   - Framework dependencies

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

Each test file includes:
- Unit tests for public methods
- Integration tests for handler combinations
- Validation tests for edge cases