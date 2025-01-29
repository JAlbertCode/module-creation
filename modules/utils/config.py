"""
Configuration management for Hugging Face to Lilypad conversion
"""

from typing import Dict, Any, Optional
import os
import json
from pathlib import Path
from ..types import ModelType, TaskType

class ModuleConfig:
    """Configuration management for Lilypad modules"""
    
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self._load_defaults()

    def _load_defaults(self) -> None:
        """Load default configuration values"""
        self.defaults = {
            'machine': {
                'gpu': int(self.model_type.requirements.min_gpu_memory > 0),
                'cpu': 1000,  # millicpus
                'ram': max(8000, self.model_type.requirements.min_ram)  # MB
            },
            'job': {
                'APIVersion': 'V1beta1',
                'Spec': {
                    'Deal': {
                        'Concurrency': 1
                    },
                    'Docker': {
                        'Entrypoint': ['python', '/workspace/run_inference.py'],
                        'WorkingDirectory': '/workspace',
                        'Image': f'huggingface/{self.model_type.name}'
                    },
                    'Engine': 'Docker',
                    'Network': {
                        'Type': 'None'
                    },
                    'PublisherSpec': {
                        'Type': 'ipfs'
                    },
                    'Timeout': 1800
                }
            }
        }

    def generate_module_config(self, custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate Lilypad module configuration
        
        Args:
            custom_config: Optional custom configuration to override defaults
            
        Returns:
            Dictionary containing module configuration
        """
        config = self.defaults.copy()
        
        # Add task-specific environment variables
        env_vars = [
            f'MODEL_ID={self.model_type.task.name}',
            f'TASK={self.model_type.task.name}'
        ]
        
        # Add framework-specific variables
        if self.model_type.framework == 'pytorch':
            env_vars.extend([
                'TORCH_CUDA_ARCH_LIST=7.5',
                'TORCH_CUDA_VERSION=11.8'
            ])
        
        # Add quantization variables if needed
        if self.model_type.quantization:
            env_vars.append(f'QUANTIZATION={self.model_type.quantization}')
        
        # Add special inputs if any
        if self.model_type.special_inputs:
            for key, value in self.model_type.special_inputs.items():
                env_vars.append(f'{key.upper()}={value}')
        
        config['job']['Spec']['Docker']['EnvironmentVariables'] = env_vars
        
        # Update with custom config if provided
        if custom_config:
            self._deep_update(config, custom_config)
        
        return config

    def generate_dockerfile(self) -> str:
        """
        Generate Dockerfile content
        
        Returns:
            String containing Dockerfile content
        """
        base_image = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime" if self.model_type.framework == 'pytorch' \
            else "tensorflow/tensorflow:2.14.0-gpu"
        
        dockerfile = [
            f"FROM {base_image}",
            "",
            "WORKDIR /workspace",
            "",
            "# Install system dependencies",
            "RUN apt-get update && apt-get install -y \\",
            "    git \\",
            "    python3-pip \\"
        ]
        
        # Add task-specific system packages
        if self.model_type.requirements.system_packages:
            for package in self.model_type.requirements.system_packages:
                dockerfile.append(f"    {package} \\")
        
        dockerfile.extend([
            "    && rm -rf /var/lib/apt/lists/*",
            "",
            "# Install Python dependencies",
            "COPY requirements.txt .",
            "RUN pip install --no-cache-dir -r requirements.txt",
            "",
            "# Create directories",
            "RUN mkdir -p /workspace/input /workspace/output",
            "",
            "# Copy inference code",
            "COPY run_inference.py .",
            "",
            "# Set environment variables",
            "ENV PYTHONUNBUFFERED=1",
            "ENV TRANSFORMERS_OFFLINE=1"
        ])
        
        # Add framework-specific environment variables
        if self.model_type.framework == 'pytorch':
            dockerfile.extend([
                "ENV TORCH_CUDA_ARCH_LIST=7.5",
                "ENV TORCH_CUDA_VERSION=11.8"
            ])
        
        dockerfile.extend([
            "",
            "# Default command",
            'ENTRYPOINT ["python", "run_inference.py"]'
        ])
        
        return "\n".join(dockerfile)

    def generate_requirements(self) -> str:
        """
        Generate requirements.txt content
        
        Returns:
            String containing requirements.txt content
        """
        requirements = self.model_type.requirements.required_packages.copy()
        
        # Add task-specific requirements
        if 'vision' in self.model_type.task.category:
            requirements.extend([
                'pillow>=10.0.0',
                'torchvision>=0.16.0'
            ])
        elif 'audio' in self.model_type.task.category:
            requirements.extend([
                'librosa>=0.10.1',
                'soundfile>=0.12.1'
            ])
        elif 'video' in self.model_type.task.category:
            requirements.extend([
                'decord>=0.6.0',
                'av>=10.0.0'
            ])
        
        return "\n".join(sorted(set(requirements)))

    def save_config(self, output_dir: str) -> None:
        """
        Save all configuration files
        
        Args:
            output_dir: Directory to save configuration files in
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save module config
        with open(output_path / 'lilypad_module.json', 'w') as f:
            json.dump(self.generate_module_config(), f, indent=2)
        
        # Save Dockerfile
        with open(output_path / 'Dockerfile', 'w') as f:
            f.write(self.generate_dockerfile())
        
        # Save requirements
        with open(output_path / 'requirements.txt', 'w') as f:
            f.write(self.generate_requirements())

    @staticmethod
    def _deep_update(base_dict: Dict, update_dict: Dict) -> None:
        """Recursively update a dictionary"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                ModuleConfig._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value