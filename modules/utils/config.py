"""Configuration utilities for Lilypad modules"""

import json
from typing import Dict, Any, List, Optional
import os
from huggingface_hub.hf_api import ModelInfo

def generate_module_template(
    model_id: str,
    model_type: Dict[str, Any],
    model_configs: Dict[str, Any]
) -> str:
    """Generate Lilypad module template"""
    
    gpu_count = 1 if model_type.get("requires_gpu", False) else 0
    memory = model_type.get("memory_requirements", 8000)
    
    # Get environment variables based on task
    env_variables = get_task_env_variables(model_type["task"])
    
    # Add model configurations as environment variables
    for key, value in model_configs.items():
        if value is not None:
            env_variables.append(f'"{key.upper()}={value}"')
    
    # Base template
    template = {
        "machine": {
            "gpu": gpu_count,
            "cpu": 1000,
            "ram": memory
        },
        "job": {
            "APIVersion": "V1beta1",
            "Spec": {
                "Deal": {
                    "Concurrency": 1
                },
                "Docker": {
                    "Entrypoint": ["python", "/app/run_inference.py"],
                    "WorkingDirectory": "/app",
                    "EnvironmentVariables": [
                        "HF_HUB_OFFLINE=1",
                        "TRANSFORMERS_OFFLINE=1"
                    ] + env_variables,
                    "Image": "" # To be filled by module generator
                },
                "Engine": "Docker",
                "Network": {
                    "Type": "None"
                },
                "Outputs": [
                    {
                        "Name": "outputs",
                        "Path": "/outputs"
                    }
                ],
                "PublisherSpec": {
                    "Type": "ipfs"
                },
                "Resources": {
                    "GPU": str(gpu_count)
                },
                "Timeout": 1800
            }
        }
    }
    
    return json.dumps(template, indent=4)

def get_task_env_variables(task: str) -> List[str]:
    """Get environment variables needed for a task"""
    
    common_vars = [
        # Common model settings
        '"MODEL_DTYPE=float16"',
        '"USE_SAFETENSORS=1"'
    ]
    
    task_vars = {
        "text-generation": [
            '"MAX_NEW_TOKENS=256"',
            '"TEMPERATURE=0.7"',
            '"TOP_P=0.9"',
            '"DO_SAMPLE=1"'
        ],
        "text-to-image": [
            '"HEIGHT=1024"',
            '"WIDTH=1024"',
            '"NUM_INFERENCE_STEPS=50"',
            '"GUIDANCE_SCALE=7.5"'
        ],
        "image-classification": [
            '"TOP_K=5"',
            '"THRESHOLD=0.5"'
        ],
        "visual-question-answering": [
            '"MAX_NEW_TOKENS=100"',
            '"NUM_BEAMS=4"'
        ]
    }
    
    return common_vars + task_vars.get(task, [])

def generate_dockerfile(
    model_type: Dict[str, Any],
    requirements: Optional[List[str]] = None,
    system_packages: Optional[List[str]] = None
) -> str:
    """Generate Dockerfile for module"""
    
    if requirements is None:
        requirements = []
    if system_packages is None:
        system_packages = []
        
    # Add task-specific system packages
    if model_type["task"] in ["text-to-image", "image-classification"]:
        system_packages.extend(["libgl1-mesa-glx", "libglib2.0-0"])
    elif model_type["task"] in ["text-to-speech", "speech-recognition"]:
        system_packages.extend(["libsndfile1", "ffmpeg"])
        
    # Generate Dockerfile content
    dockerfile = [
        "FROM python:3.9-slim",
        "",
        "WORKDIR /app",
        "",
        "# Install system dependencies",
        "RUN apt-get update && apt-get install -y \\",
        "    " + " \\\n    ".join(system_packages) + " \\",
        "    && rm -rf /var/lib/apt/lists/*",
        "",
        "# Install Python packages",
        "RUN pip install --no-cache-dir \\",
        "    " + " \\\n    ".join(requirements),
        "",
        "# Create directories",
        "RUN mkdir -p /cache/huggingface",
        "RUN mkdir -p /outputs",
        "",
        "# Set environment variables", 
        "ENV HF_HOME=/cache/huggingface",
        "ENV PYTHONUNBUFFERED=1",
        "",
        "# Copy model files",
        "COPY ./model /app/model",
        "",
        "# Copy inference script",
        "COPY run_inference.py /app/",
        "",
        "# Set output directory as volume",
        "VOLUME /outputs",
        "",
        "# Run inference script",
        "CMD [\"python\", \"/app/run_inference.py\"]"
    ]
    
    return "\n".join(dockerfile)

def get_model_configs(model_type: Dict[str, Any]) -> Dict[str, Any]:
    """Get available configurations for a model type"""
    
    base_configs = {
        "model_dtype": {
            "description": "Model precision",
            "options": ["float16", "float32", "bfloat16"],
            "default": "float16"
        },
        "use_safetensors": {
            "description": "Use safetensors format",
            "options": [True, False],
            "default": True
        }
    }
    
    task_configs = {
        "text-generation": {
            "max_new_tokens": {
                "description": "Maximum number of tokens to generate",
                "range": [1, 2048],
                "default": 256
            },
            "temperature": {
                "description": "Sampling temperature",
                "range": [0.1, 2.0],
                "default": 0.7
            },
            "top_p": {
                "description": "Top-p sampling",
                "range": [0.1, 1.0],
                "default": 0.9
            },
            "do_sample": {
                "description": "Use sampling instead of greedy decoding",
                "options": [True, False],
                "default": True
            }
        },
        "text-to-image": {
            "height": {
                "description": "Image height",
                "options": [512, 768, 1024],
                "default": 1024
            },
            "width": {
                "description": "Image width", 
                "options": [512, 768, 1024],
                "default": 1024
            },
            "num_inference_steps": {
                "description": "Number of denoising steps",
                "range": [1, 100],
                "default": 50
            },
            "guidance_scale": {
                "description": "Guidance scale for image generation",
                "range": [1.0, 20.0],
                "default": 7.5
            }
        },
        "image-classification": {
            "top_k": {
                "description": "Number of top predictions to return",
                "range": [1, 100],
                "default": 5
            },
            "threshold": {
                "description": "Classification threshold",
                "range": [0.0, 1.0],
                "default": 0.5
            }
        }
    }
    
    configs = base_configs.copy()
    if model_type["task"] in task_configs:
        configs.update(task_configs[model_type["task"]])
        
    return configs

def load_model_config(config_path: str) -> Dict[str, Any]:
    """Load model configuration from file"""
    if not os.path.exists(config_path):
        return {}
        
    with open(config_path) as f:
        return json.load(f)

def save_model_config(config: Dict[str, Any], config_path: str) -> None:
    """Save model configuration to file"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def validate_model_config(config: Dict[str, Any], model_type: Dict[str, Any]) -> bool:
    """Validate model configuration against requirements"""
    available_configs = get_model_configs(model_type)
    
    for key, value in config.items():
        if key not in available_configs:
            return False
            
        config_spec = available_configs[key]
        if "options" in config_spec:
            if value not in config_spec["options"]:
                return False
        elif "range" in config_spec:
            min_val, max_val = config_spec["range"]
            if value < min_val or value > max_val:
                return False
                
    return True

def get_input_template(task: str) -> Dict[str, Any]:
    """Get input template for a task"""
    templates = {
        "text-generation": {
            "format": "text",
            "example": "Write a story about an adventurous cat.",
            "max_length": 2048
        },
        "text-to-image": {
            "format": "text",
            "example": "A beautiful sunset over mountains, digital art style",
            "max_length": 500
        },
        "image-classification": {
            "format": "image",
            "supported_formats": ["jpg", "png", "jpeg"],
            "max_size": "4096x4096",
            "example_path": "/inputs/image.jpg"
        },
        "visual-question-answering": {
            "format": "multimodal",
            "components": {
                "image": {
                    "format": "image",
                    "supported_formats": ["jpg", "png", "jpeg"]
                },
                "question": {
                    "format": "text",
                    "max_length": 500
                }
            },
            "example": {
                "image": "/inputs/image.jpg",
                "question": "What is the main object in this image?"
            }
        }
    }
    
    return templates.get(task, {"format": "unknown"})