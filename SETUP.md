# Setup Guide

## Prerequisites

1. Python 3.9 or later
2. pip (Python package installer)
3. Virtual environment (recommended)

## Installation Steps

1. Create and activate virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate on Unix/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Flask application:
```bash
# Make sure you're in the project directory
python app.py
```

2. Open in web browser:
```
http://localhost:5000
```

## Troubleshooting

Common issues and solutions:

1. ModuleNotFoundError:
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt

# If you get SSL errors during installation, try:
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

2. Port already in use:
```bash
# Change the port in app.py to something else like 5001
app.run(debug=True, port=5001)
```

3. Permission errors:
```bash
# On Unix/Linux, you might need:
sudo python app.py
```

## Directory Structure

Make sure your directory structure looks like this:
```
Image-Classification/
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
└── README.md
```

## Example Usage

The application supports various Lilypad modules. Here are some example model URLs you can use:

1. ResNet (Image Classification):
   - https://huggingface.co/microsoft/resnet-50

2. BERT (Text Classification):
   - https://huggingface.co/bert-base-uncased

3. [Add other example URLs from your updated instructions]

## Testing

After generating the module:

1. Extract the files:
```bash
unzip lilypad-module.zip
cd lilypad-module
```

2. Build and test:
```bash
docker build -t test-model .
docker run -v $(pwd)/input:/workspace/input test-model
```