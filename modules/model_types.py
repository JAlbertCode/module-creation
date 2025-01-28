"""
Comprehensive model type detection and configuration for Hugging Face models.
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass

@dataclass
class ModelIOConfig:
    """Configuration for model inputs and outputs"""
    input_types: List[str]
    output_types: List[str]
    supports_batching: bool
    required_libraries: List[str]
    
@dataclass
class ModelType:
    """Complete model type information"""
    task: str
    architecture: str
    input_config: ModelIOConfig
    output_config: ModelIOConfig
    pipeline_tag: str
    special_requirements: Optional[Dict] = None

# Comprehensive mapping of task types to their IO configurations
TASK_CONFIGS = {
    # Text Processing Tasks
    'text-classification': ModelIOConfig(
        input_types=['text'],
        output_types=['label', 'scores'],
        supports_batching=True,
        required_libraries=['transformers', 'torch']
    ),
    'text-generation': ModelIOConfig(
        input_types=['text'],
        output_types=['text'],
        supports_batching=True,
        required_libraries=['transformers', 'torch']
    ),
    'translation': ModelIOConfig(
        input_types=['text', 'source_lang', 'target_lang'],
        output_types=['text'],
        supports_batching=True,
        required_libraries=['transformers', 'torch']
    ),
    
    # Image Processing Tasks
    'image-classification': ModelIOConfig(
        input_types=['image'],
        output_types=['label', 'scores'],
        supports_batching=True,
        required_libraries=['transformers', 'torch', 'pillow']
    ),
    'image-segmentation': ModelIOConfig(
        input_types=['image'],
        output_types=['masks', 'boxes', 'scores'],
        supports_batching=True,
        required_libraries=['transformers', 'torch', 'pillow']
    ),
    'object-detection': ModelIOConfig(
        input_types=['image'],
        output_types=['boxes', 'labels', 'scores'],
        supports_batching=True,
        required_libraries=['transformers', 'torch', 'pillow']
    ),
    
    # Audio Processing Tasks
    'automatic-speech-recognition': ModelIOConfig(
        input_types=['audio'],
        output_types=['text'],
        supports_batching=False,
        required_libraries=['transformers', 'torch', 'librosa']
    ),
    'text-to-speech': ModelIOConfig(
        input_types=['text'],
        output_types=['audio'],
        supports_batching=False,
        required_libraries=['transformers', 'torch', 'soundfile']
    ),
    
    # Video Processing Tasks
    'video-classification': ModelIOConfig(
        input_types=['video'],
        output_types=['label', 'scores'],
        supports_batching=False,
        required_libraries=['transformers', 'torch', 'decord']
    ),
    
    # Multi-modal Tasks
    'visual-question-answering': ModelIOConfig(
        input_types=['image', 'text'],
        output_types=['text'],
        supports_batching=True,
        required_libraries=['transformers', 'torch', 'pillow']
    ),
    'document-question-answering': ModelIOConfig(
        input_types=['image', 'text'],
        output_types=['text', 'boxes'],
        supports_batching=True,
        required_libraries=['transformers', 'torch', 'pillow']
    ),
    
    # Specialized Tasks
    'point-cloud': ModelIOConfig(
        input_types=['point_cloud'],
        output_types=['classification', 'segmentation'],
        supports_batching=True,
        required_libraries=['transformers', 'torch', 'numpy']
    ),
    'graph-processing': ModelIOConfig(
        input_types=['graph'],
        output_types=['node_features', 'edge_features'],
        supports_batching=True,
        required_libraries=['transformers', 'torch', 'networkx']
    )
}

def detect_model_type(model_info: Dict) -> ModelType:
    """
    Detect the complete model type and configuration from model info
    
    Args:
        model_info: Dictionary containing model information from Hugging Face
        
    Returns:
        ModelType object with complete configuration
    """
    # Extract task and architecture information
    pipeline_tag = model_info.get('pipeline_tag')
    if not pipeline_tag:
        # Attempt to detect from model config and tags
        tags = model_info.get('tags', [])
        config = model_info.get('config', {})
        architecture = config.get('architectures', ['Unknown'])[0]
        
        # Logic to detect task from tags and architecture
        if any('classification' in tag.lower() for tag in tags):
            pipeline_tag = 'text-classification'
        elif any('generation' in tag.lower() for tag in tags):
            pipeline_tag = 'text-generation'
        # Add more detection logic here
    
    # Get task configuration
    task_config = TASK_CONFIGS.get(pipeline_tag, TASK_CONFIGS['text-classification'])
    
    return ModelType(
        task=pipeline_tag,
        architecture=model_info.get('config', {}).get('architectures', ['Unknown'])[0],
        input_config=task_config,
        output_config=task_config,
        pipeline_tag=pipeline_tag,
        special_requirements=detect_special_requirements(model_info)
    )

def detect_special_requirements(model_info: Dict) -> Dict:
    """
    Detect any special requirements for the model
    
    Args:
        model_info: Dictionary containing model information from Hugging Face
        
    Returns:
        Dictionary of special requirements
    """
    requirements = {}
    
    # Check model size for GPU requirements
    if model_info.get('size_in_bytes', 0) > 10 * 1024 * 1024 * 1024:  # 10GB
        requirements['gpu'] = True
        requirements['min_gpu_memory'] = '16GB'
    
    # Check for quantization
    if any('quantized' in tag.lower() for tag in model_info.get('tags', [])):
        requirements['quantization'] = True
    
    # Check for specific framework requirements
    if 'pytorch' in model_info.get('library_name', '').lower():
        requirements['framework'] = 'pytorch'
    elif 'tensorflow' in model_info.get('library_name', '').lower():
        requirements['framework'] = 'tensorflow'
    
    return requirements