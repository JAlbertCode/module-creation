# Project Structure

This document outlines the organization and purpose of each component in the project.

## Directory Structure

```
├── app.py                      # Web application entry point
├── modules/
│   ├── analyzer.py            # Model analysis system
│   ├── converter.py           # Main conversion interface
│   ├── template_generator.py  # Dynamic template generation
│   ├── types/                # Type definitions
│   │   ├── __init__.py
│   │   └── model_types.py    # Model type definitions
│   └── utils/                # Utility functions
│       ├── __init__.py
│       ├── config.py         # Configuration management
│       ├── download.py       # Model downloading
│       └── validation.py     # Input validation
├── templates/                # Base templates
│   ├── Dockerfile.jinja2     # Base Dockerfile template
│   ├── inference.py.jinja2   # Base inference script template
│   ├── module_config.json.tmpl # Base module config template
│   └── requirements.txt.jinja2 # Base requirements template
├── tests/                    # Test files
│   ├── __init__.py
│   ├── test_analyzer.py      # Model analyzer tests
│   ├── test_converter.py     # Converter tests
│   └── test_templates.py     # Template tests
└── requirements.txt          # Python dependencies
```

## Core Components

### Model Analysis (modules/analyzer.py)

Comprehensive model analysis system:
- Task and architecture detection
- Framework identification
- Hardware requirements analysis
- Dependency resolution
- Generation parameter optimization
- Model-specific configuration detection

### Universal Converter (modules/converter.py)

Main interface for model conversion:
- Hugging Face to Lilypad conversion
- Automatic file generation
- Model download script creation
- Validation and testing
- Error handling

### Template Generator (modules/template_generator.py)

Dynamic template generation system:
- Model-specific adaptations
- Input/output handling
- Error handling generation
- Resource optimization
- Performance monitoring

### Templates (templates/)

Base templates that adapt to model requirements:
- Base Dockerfile with dynamic dependencies
- Universal inference script template
- Adaptive module configuration
- Dynamic requirements generation

## Dependencies

### Core Requirements
- torch>=2.0.0
- transformers>=4.36.0
- diffusers>=0.25.0
- accelerate>=0.25.0
- safetensors>=0.4.0
- jinja2>=3.0.0
- pyyaml>=6.0.0
- huggingface-hub>=0.20.0

### Development Requirements
- pytest>=7.0.0
- black>=23.0.0
- isort>=5.0.0
- mypy>=1.0.0
- pylint>=3.0.0

## Features

### Implemented
- Universal model conversion
- Automatic analysis system
- Dynamic template generation
- Hardware requirement detection
- Dependency resolution
- Error handling
- Performance optimization

### In Progress
- Testing framework
- Model optimization tools
- Performance monitoring
- Resource usage optimization

### Planned
- CLI interface
- Batch processing
- Web interface
- Documentation system
- CI/CD pipeline

## Development

### Setting Up Development Environment

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\\venv\\Scripts\\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

### Running Tests

```bash
pytest tests/  # Run all tests
pytest tests/test_analyzer.py  # Test specific component
```

### Building Documentation

```bash
mkdocs serve  # Start documentation server
mkdocs build  # Build documentation site
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Run tests
5. Submit pull request

See CONTRIBUTING.md for detailed guidelines.