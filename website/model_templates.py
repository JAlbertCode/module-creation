from typing import Dict, Any
import json

class ModelTemplate:
    """Generic model template generator that handles any Hugging Face model"""
    
    def __init__(self, model_info):
        self.model_info = model_info
        self.pipeline_type = self._detect_pipeline_type()
        self.resources = self._estimate_resources()
        
    def _detect_pipeline_type(self) -> str:
        """Detect the appropriate pipeline type from model tags and metadata"""
        # Common pipeline types in Hugging Face
        pipeline_mapping = {
            'text-classification': 'text-classification',
            'sentiment-analysis': 'text-classification',
            'image-classification': 'image-classification',
            'object-detection': 'object-detection',
            'text-generation': 'text-generation',
            'automatic-speech-recognition': 'automatic-speech-recognition',
            'image-to-text': 'image-to-text',
            'text-to-image': 'text-to-image',
            'translation': 'translation',
            'summarization': 'summarization',
            'question-answering': 'question-answering',
            'token-classification': 'token-classification',
            'zero-shot-classification': 'zero-shot-classification',
            'feature-extraction': 'feature-extraction'
        }
        
        # Check model tags
        for tag in self.model_info.tags:
            if tag in pipeline_mapping:
                return pipeline_mapping[tag]
            
        # Fallback to feature-extraction if no specific task is identified
        return 'feature-extraction'
    
    def _estimate_resources(self) -> Dict[str, Any]:
        """Estimate required resources based on model size and type"""
        size_mb = self.model_info.size_in_bytes / (1024 * 1024)
        
        resources = {
            'cpu': '1',
            'memory': '2Gi',
            'gpu': None
        }
        
        # Adjust based on model size
        if size_mb > 1000:  # > 1GB
            resources.update({
                'cpu': '2',
                'memory': '8Gi',
                'gpu': '1'
            })
        elif size_mb > 500:  # > 500MB
            resources.update({
                'cpu': '2',
                'memory': '4Gi',
                'gpu': '1'
            })
            
        return resources
    
    def generate_run_inference(self) -> str:
        """Generate the inference script based on model type"""
        return f'''import os
import json
from transformers import pipeline
import torch

def load_input_data(input_path):
    """Load and preprocess input data based on model type"""
    if "{self.pipeline_type}" in ["image-classification", "object-detection", "image-to-text"]:
        from PIL import Image
        return Image.open(input_path)
    elif "{self.pipeline_type}" == "automatic-speech-recognition":
        import librosa
        audio, _ = librosa.load(input_path)
        return audio
    else:
        # Text-based inputs
        with open(input_path, 'r') as f:
            return f.read()

def main():
    try:
        # Get input path from environment
        input_path = os.environ.get('INPUT_PATH')
        if not input_path:
            raise ValueError("INPUT_PATH environment variable must be set")
        
        # Get optional parameters
        params = json.loads(os.environ.get('MODEL_PARAMS', '{{}}'))
        
        # Determine device
        device = 0 if torch.cuda.is_available() else -1
        
        # Load the model with the appropriate pipeline
        pipe = pipeline(
            task="{self.pipeline_type}",
            model="{self.model_info.id}",
            device=device
        )
        
        # Load and preprocess input
        input_data = load_input_data(input_path)
        
        # Run inference
        result = pipe(input_data, **params)
        
        # Format output
        output = {{
            'model_id': "{self.model_info.id}",
            'pipeline_type': "{self.pipeline_type}",
            'result': result,
            'status': 'success'
        }}
        
    except Exception as e:
        output = {{
            'error': str(e),
            'status': 'error'
        }}
    
    # Save output
    os.makedirs('/outputs', exist_ok=True)
    with open('/outputs/result.json', 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
'''

    def generate_requirements(self) -> str:
        """Generate requirements.txt based on model type"""
        requirements = [
            "transformers>=4.36.0",
            "torch>=2.1.0",
            "numpy<2.0.0",
        ]
        
        # Add task-specific requirements
        if self.pipeline_type in ["image-classification", "object-detection", "image-to-text"]:
            requirements.extend([
                "pillow>=10.0.0",
                "torchvision>=0.16.0"
            ])
        elif self.pipeline_type == "automatic-speech-recognition":
            requirements.extend([
                "librosa>=0.10.1"
            ])
            
        return "\n".join(requirements)
    
    def generate_dockerfile(self) -> str:
        """Generate Dockerfile with appropriate base image and dependencies"""
        return f'''FROM python:3.9-slim

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create output directory
RUN mkdir -p /outputs

# Copy inference script
COPY run_inference.py .

# Set entrypoint
ENTRYPOINT ["python", "/workspace/run_inference.py"]
'''

    def generate_module_yaml(self) -> str:
        """Generate Lilypad module.yaml configuration"""
        return f'''name: {self.model_info.id.split("/")[-1]}
version: 1.0.0
description: Hugging Face model deployment for {self.model_info.id}
author: Generated by Lilypad Module Generator

resources:
  cpu: {self.resources['cpu']}
  memory: {self.resources['memory']}
  {f'gpu: {self.resources["gpu"]}' if self.resources["gpu"] else ''}

input:
  - name: INPUT_PATH
    description: Path to the input file
    type: file
  - name: MODEL_PARAMS
    description: Optional model parameters as JSON string
    type: string
    required: false

output:
  - name: result
    description: Model output
    type: file
    path: /outputs/result.json
'''

    def generate_readme(self) -> str:
        """Generate README with setup and usage instructions"""
        return f'''# {self.model_info.id.split("/")[-1]} Lilypad Module

This module deploys the [{self.model_info.id}](https://huggingface.co/{self.model_info.id}) model from Hugging Face on Lilypad.

## Model Information
- Task: {self.pipeline_type}
- Model ID: {self.model_info.id}
- Description: {self.model_info.description}

## Setup Instructions

1. Build the Docker image:
   ```bash
   docker build -t {self.model_info.id.split("/")[-1]} .
   ```

2. Test locally:
   ```bash
   # For text input:
   echo "Your input text" > input.txt
   
   # Run the container:
   docker run -v $(pwd)/input.txt:/workspace/input.txt \\
             -e INPUT_PATH=/workspace/input.txt \\
             {self.model_info.id.split("/")[-1]}
   ```

3. Deploy to Lilypad:
   ```bash
   lilypad module deploy .
   ```

## Input Format
- The model expects input through the INPUT_PATH environment variable
- For this {self.pipeline_type} model:
  {self._get_input_format_description()}

## Output Format
The model will generate a JSON file containing:
```json
{{
    "model_id": "{self.model_info.id}",
    "pipeline_type": "{self.pipeline_type}",
    "result": [
        // Model-specific output format
    ],
    "status": "success"
}}
```

## Additional Parameters
You can pass additional parameters to the model using the MODEL_PARAMS environment variable as a JSON string.

## Resource Requirements
- CPU: {self.resources['cpu']} cores
- Memory: {self.resources['memory']}
{f"- GPU: {self.resources['gpu']}" if self.resources['gpu'] else ""}
'''

    def _get_input_format_description(self) -> str:
        """Get description of input format based on pipeline type"""
        formats = {
            'text-classification': '- Text file containing the text to classify',
            'image-classification': '- Image file (JPG, PNG, etc.)',
            'object-detection': '- Image file (JPG, PNG, etc.)',
            'text-generation': '- Text file containing the prompt',
            'automatic-speech-recognition': '- Audio file (WAV, MP3, etc.)',
            'image-to-text': '- Image file (JPG, PNG, etc.)',
            'text-to-image': '- Text file containing the image description',
            'translation': '- Text file containing the text to translate',
            'summarization': '- Text file containing the text to summarize',
            'question-answering': '- JSON file containing question and context fields',
            'token-classification': '- Text file containing the text to analyze',
            'zero-shot-classification': '- JSON file containing text and candidate labels',
            'feature-extraction': '- Text file containing the input text'
        }
        return formats.get(self.pipeline_type, '- Text file with appropriate input for the model')