"""
Pytest fixtures for testing Hugging Face to Lilypad conversion
"""

import os
import shutil
import pytest
from typing import Dict, Any

@pytest.fixture
def test_models() -> Dict[str, Dict[str, Any]]:
    """Sample models for testing different types"""
    return {
        "text_generation": {
            "model_id": "gpt2",
            "task": "text-generation",
            "architecture": "GPT2LMHeadModel",
            "inputs": ["text"],
            "outputs": ["text"]
        },
        "image_classification": {
            "model_id": "google/vit-base-patch16-224",
            "task": "image-classification",
            "architecture": "ViTForImageClassification",
            "inputs": ["image"],
            "outputs": ["classification"]
        },
        "text_to_image": {
            "model_id": "CompVis/stable-diffusion-v1-4",
            "task": "text-to-image",
            "architecture": "StableDiffusionPipeline",
            "inputs": ["text"],
            "outputs": ["image"]
        },
        "speech_recognition": {
            "model_id": "openai/whisper-small",
            "task": "automatic-speech-recognition",
            "architecture": "WhisperForConditionalGeneration",
            "inputs": ["audio"],
            "outputs": ["text"]
        }
    }

@pytest.fixture
def temp_output_dir(tmpdir):
    """Temporary directory for generated files"""
    output_dir = os.path.join(str(tmpdir), "output")
    os.makedirs(output_dir, exist_ok=True)
    yield output_dir
    # Cleanup
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

@pytest.fixture
def mock_model_info():
    """Mock model info for testing without API calls"""
    def _make_info(model_id: str, task: str, architecture: str) -> Dict[str, Any]:
        return {
            "id": model_id,
            "pipeline_tag": task,
            "config": {
                "architectures": [architecture],
                "model_type": architecture.split("For")[0].lower(),
                "use_cache": True,
                "num_attention_heads": 12,
                "hidden_size": 768
            },
            "cardData": f"""
---
language: en
tags:
- {task}
- pytorch
license: apache-2.0
---

# Model Card

## Model Details
- Architecture: {architecture}
- Task: {task}
- Training Data: Example Dataset
- Hardware Requirements: GPU recommended
            """
        }
    return _make_info
    
@pytest.fixture
def mock_file_contents():
    """Expected file contents for validation"""
    return {
        "Dockerfile": [
            "FROM nvidia/cuda:",
            "WORKDIR /app",
            "COPY requirements.txt .",
            "RUN pip install -r requirements.txt",
            "COPY . ."
        ],
        "run_inference.py": [
            "import torch",
            "import os",
            "def main():",
            "if __name__ == \"__main__\":"
        ],
        "lilypad_module.json.tmpl": [
            "\"machine\": {",
            "\"gpu\":",
            "\"Spec\": {",
            "\"Docker\": {"
        ],
        "requirements.txt": [
            "torch>=",
            "transformers>=",
            "accelerate>="
        ]
    }

@pytest.fixture
def validation_checks():
    """Common validation checks for generated files"""
    def _validate_files(module_dir: str, expected_files: Dict[str, list]):
        results = {}
        for filename, expected_content in expected_files.items():
            file_path = os.path.join(module_dir, filename)
            
            # Check file exists
            results[filename] = {
                "exists": os.path.exists(file_path)
            }
            
            if results[filename]["exists"]:
                with open(file_path, 'r') as f:
                    content = f.read()
                    # Check for expected content
                    results[filename]["content_checks"] = [
                        line in content for line in expected_content
                    ]
                    
        return results
    return _validate_files