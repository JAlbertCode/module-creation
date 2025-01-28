"""
Validation utilities for checking model compatibility
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class ValidationResult:
    """Results of model validation"""
    is_valid: bool
    messages: List[str]
    requirements: Dict[str, any]

class InputValidator:
    """Validates model input requirements"""
    
    @staticmethod
    def get_input_requirements(input_type: str) -> Dict[str, any]:
        """Get requirements for input type"""
        requirements = {
            'text': {
                'formats': ['txt'],
                'max_length': 512,
                'supports_batching': True
            },
            'image': {
                'formats': ['jpg', 'jpeg', 'png'],
                'max_size': [1024, 1024],
                'supports_batching': True
            },
            'audio': {
                'formats': ['wav', 'mp3'],
                'max_duration': 30,
                'supports_batching': False
            },
            'video': {
                'formats': ['mp4'],
                'max_duration': 30,
                'supports_batching': False
            },
            'point_cloud': {
                'formats': ['ply', 'pcd'],
                'max_points': 100000,
                'supports_batching': True
            },
            'graph': {
                'formats': ['json'],
                'max_nodes': 1000,
                'supports_batching': True
            }
        }
        return requirements.get(input_type, {})

def check_model_compatibility(model_info: Dict) -> ValidationResult:
    """
    Check if a model is compatible with Lilypad infrastructure
    
    Args:
        model_info: Dictionary containing model information from Hugging Face
        
    Returns:
        ValidationResult with compatibility information
    """
    messages = []
    requirements = {}
    
    # Check model size
    model_size = model_info.get('size_in_bytes', 0)
    if model_size > 10 * 1024 * 1024 * 1024:  # 10GB
        messages.append(f"Model size ({model_size/1024/1024/1024:.1f}GB) requires GPU acceleration")
        requirements['gpu'] = True
        requirements['min_gpu_memory'] = '16GB'
    
    # Check framework compatibility
    framework = model_info.get('library_name', '').lower()
    if 'pytorch' in framework:
        requirements['framework'] = 'pytorch'
    elif 'tensorflow' in framework:
        requirements['framework'] = 'tensorflow'
    else:
        messages.append(f'Unsupported framework: {framework}')
        return ValidationResult(False, messages, requirements)
    
    # Check if model has pipeline tag
    pipeline_tag = model_info.get('pipeline_tag')
    if not pipeline_tag:
        messages.append('No pipeline tag found, task detection may be unreliable')
        # Try to detect from tags
        tags = model_info.get('tags', [])
        if any('classification' in tag.lower() for tag in tags):
            pipeline_tag = 'text-classification'
        elif any('generation' in tag.lower() for tag in tags):
            pipeline_tag = 'text-generation'
        requirements['detected_task'] = pipeline_tag
    
    # Check for quantization
    if any('quantized' in tag.lower() for tag in model_info.get('tags', [])):
        requirements['quantization'] = True
        messages.append('Model is quantized, reduced memory requirements')
    
    # Check for special architecture requirements
    architecture = model_info.get('config', {}).get('architectures', ['Unknown'])[0]
    if 'Llama' in architecture:
        requirements['min_gpu_memory'] = '24GB'
        messages.append('Llama architecture requires minimum 24GB GPU memory')
    elif 'GPT' in architecture:
        requirements['min_gpu_memory'] = '16GB'
        messages.append('GPT architecture requires minimum 16GB GPU memory')
    
    # Check for dataset requirements
    if model_info.get('dataset_tags'):
        requirements['datasets'] = model_info['dataset_tags']
        messages.append(f'Model requires specific datasets: {model_info["dataset_tags"]}')
    
    # Check license
    license_info = model_info.get('license', '')
    if license_info:
        requirements['license'] = license_info
        if any(restricted in license_info.lower() for restricted in ['proprietary', 'commercial']):
            messages.append('Warning: Model has licensing restrictions')
    
    # Return validation result
    is_valid = True  # Can be made more strict based on requirements
    return ValidationResult(is_valid, messages, requirements)

def format_validation_message(validation_result: ValidationResult) -> str:
    """Format validation results into a human-readable message"""
    message_parts = []
    
    if validation_result.is_valid:
        message_parts.append('\u2705 Model is compatible with Lilypad')
    else:
        message_parts.append('\u274c Model is not compatible with Lilypad')
    
    message_parts.extend(validation_result.messages)
    
    # Add requirements summary
    if validation_result.requirements:
        message_parts.append('\nRequirements:')
        for key, value in validation_result.requirements.items():
            message_parts.append(f'- {key}: {value}')
    
    return '\n'.join(message_parts)