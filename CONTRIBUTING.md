# Contributing to Lilypad Model Generator

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code passes the linter
6. Issue that pull request

## Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Image-Classification.git
   cd Image-Classification
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Local Development Server

1. Start the development server:
   ```bash
   python server.py
   ```

2. Run tests:
   ```bash
   python -m pytest tests/
   ```

3. Run benchmarks:
   ```bash
   python -m benchmark.cli --model YOUR_MODEL_ID
   ```

## Project Structure

```
Image-Classification/
├── website/                 # Web interface code
│   ├── templates/          # HTML templates
│   ├── static/            # Static assets
│   ├── benchmark/         # Benchmarking tools
│   └── tests/             # Test files
├── docs/                  # Documentation
└── examples/             # Example configurations
```

## Adding New Features

### Adding Model Support

1. Update `model_templates.py`:
   ```python
   def _detect_model_type(self):
       # Add your model type detection logic
       pass
   ```

2. Add model-specific templates in `templates/`:
   ```python
   def generate_model_config(self):
       # Add your model configuration
       pass
   ```

3. Update tests in `tests/`:
   ```python
   def test_new_model_type(self):
       # Add your test cases
       pass
   ```

### Adding Monitoring Metrics

1. Update `monitor.py`:
   ```python
   def collect_metrics(self):
       # Add your metrics collection
       pass
   ```

2. Add visualization in `visualize.py`:
   ```python
   def create_new_plot(self):
       # Add your visualization
       pass
   ```

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the CHANGELOG.md with a note describing your changes
3. The PR will be merged once you have the sign-off of two maintainers

## Any contributions you make will be under the MIT Software License
In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issue tracker](https://github.com/OWNER/Image-Classification/issues)
We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/OWNER/Image-Classification/issues/new).

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Use a Consistent Coding Style

* 4 spaces for indentation rather than tabs
* 80 character line length
* Run `black` for code formatting
* Run `flake8` for style guide enforcement

## License
By contributing, you agree that your contributions will be licensed under its MIT License.