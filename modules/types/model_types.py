"""
Model type definitions and detection for Hugging Face models
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from .task_types import TaskType, detect_task_type

@dataclass
class ModelRequirements:
    """Hardware and software requirements for a model"""
    min_gpu_memory: int
    min_ram: int
    required_packages: List[str]
    system_packages: List[str]
    cuda_version: Optional[str] = None

@dataclass
class ModelType:
    """Complete model type information"""
    name: str
    task: TaskType
    requirements: ModelRequirements
    framework: str
    quantization: Optional[str] = None
    special_inputs: Optional[Dict[str, str]] = None

def detect_model_type(model_info: Dict) -> ModelType:
    """
    Detect model type and requirements from model info
    
    Args:
        model_info: Dictionary containing model information from Hugging Face
        
    Returns:
        ModelType object with complete configuration
    """
    # Detect task type
    task = detect_task_type(model_info)
    
    # Determine framework
    config = model_info.get('config', {})
    if 'torch_dtype' in config:
        framework = 'pytorch'
    elif any('tensorflow' in str(f).lower() for f in model_info.get('library_name', [])):
        framework = 'tensorflow'
    else:
        framework = 'pytorch'  # default to PyTorch
    
    # Determine hardware requirements
    model_size = model_info.get('size_in_bytes', 0)
    gpu_memory = max(
        task.requirements.memory_requirements['gpu_ram'],
        int(model_size * 1.5 / (1024 * 1024 * 1024))  # 1.5x model size in GB
    )
    
    ram_requirements = max(
        task.requirements.memory_requirements['ram'],
        int(model_size * 2.5 / (1024 * 1024 * 1024))  # 2.5x model size in GB
    )
    
    # Determine required packages
    required_packages = [
        'torch>=2.1.0' if framework == 'pytorch' else 'tensorflow>=2.14.0',
        'transformers>=4.36.0',
        'safetensors>=0.4.0'
    ]
    
    # Add task-specific packages
    if 'vision' in task.category:
        required_packages.extend(['pillow>=10.0.0', 'torchvision>=0.16.0'])
    elif 'audio' in task.category:
        required_packages.extend(['librosa>=0.10.1', 'soundfile>=0.12.1'])
    elif 'video' in task.category:
        required_packages.extend(['decord>=0.6.0', 'av>=10.0.0'])
    
    # Determine system packages
    system_packages = []
    if 'audio' in task.category:
        system_packages.extend(['libsndfile1', 'ffmpeg'])
    elif 'video' in task.category:
        system_packages.append('ffmpeg')
    
    # Check for quantization
    quantization = None
    if any('quantized' in tag.lower() for tag in model_info.get('tags', [])):
        quantization = '8bit'  # or detect specific quantization
        required_packages.append('bitsandbytes>=0.41.1')
    
    # Create requirements
    requirements = ModelRequirements(
        min_gpu_memory=gpu_memory * 1024,  # Convert to MB
        min_ram=ram_requirements * 1024,    # Convert to MB
        required_packages=required_packages,
        system_packages=system_packages,
        cuda_version='11.8' if framework == 'pytorch' else '11.2'
    )
    
    # Check for special inputs
    special_inputs = {}
    if config.get('use_cache') is True:
        special_inputs['use_cache'] = 'true'
    if config.get('torch_dtype') == 'float16':
        special_inputs['dtype'] = 'float16'
    
    return ModelType(
        name=model_info.get('model_id', '').split('/')[-1],
        task=task,
        requirements=requirements,
        framework=framework,
        quantization=quantization,
        special_inputs=special_inputs if special_inputs else None
    )