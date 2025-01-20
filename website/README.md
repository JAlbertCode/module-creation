# Lilypad Module Generator Website

This web application allows users to easily generate Lilypad modules from any Hugging Face model. Simply paste a Hugging Face model URL and get a complete, deployable Lilypad module.

## Features

- Automatic model type detection
- Custom file generation based on model requirements
- Preview generated files before download
- Step-by-step setup instructions
- Downloadable zip file with all required files
- Support for various model types:
  - Image Classification
  - Text Classification
  - Object Detection
  - Sentiment Analysis
  - And more...

## Development Setup

1. Clone the repository and navigate to the website directory:
```bash
cd website
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the development server:
```bash
python server.py
```

5. Visit http://localhost:5000 in your browser

## Docker Setup

1. Build the Docker image:
```bash
docker build -t lilypad-generator .
```

2. Run the container:
```bash
docker run -p 5000:5000 lilypad-generator
```

## Usage

1. Open the website in your browser
2. Paste a Hugging Face model URL (e.g., https://huggingface.co/microsoft/resnet-50)
3. Click "Generate Module Files"
4. Review the generated files in the preview
5. Download the zip file
6. Follow the provided setup instructions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.