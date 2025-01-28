"""Validation utilities for Hugging Face models"""

import re
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from huggingface_hub import HfApi, ModelInfo
import torch
from transformers import AutoConfig

@dataclass
class ValidationResult:
    """Results of model validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    requirements: Dict[str, Any]

def estimate_model_size(model_info: ModelInfo) -> Optional[float]:
    """Estimate model size in GB from model info"""
    try:
        # Check if size is directly available
        if hasattr(model_info, 'modelSize'):
            return model_info.modelSize / (1024 * 1024 * 1024)  # Convert to GB
        
        # Try to find size in tags
        for tag in model_info.tags:
            # Look for patterns like "7B", "13B", etc.
            match = re.search(r'(\d+)B', tag)
            if match:
                return float(match.group(1))
        
        # Default sizes based on architecture
        architecture_sizes = {
            'gpt2': 1.5,
            'bert-base': 0.4,
            'bert-large': 1.2,
            'roberta': 1.5,
            't5': 3.0,
            'stable-diffusion': 4.0,
            'vit': 0.3
        }
        
        for arch, size in architecture_sizes.items():
            if arch in str(model_info.modelId).lower():
                return size
                
        return None
    except Exception:
        return None

def check_gpu_requirements(model_info: ModelInfo) -> Dict[str, Any]:
    """Determine GPU requirements for the model"""
    size = estimate_model_size(model_info)
    if not size:
        # Conservative defaults if we can't estimate
        return {
            "required": True,
            "min_vram": 8,
            "recommended_vram": 16
        }
    
    # Estimate VRAM requirements (typically 2-3x model size for inference)
    min_vram = max(8, int(size * 2))
    recommended_vram = max(16, int(size * 3))
    
    return {
        "required": size > 0.5,  # Require GPU for models over 500MB
        "min_vram": min_vram,
        "recommended_vram": recommended_vram
    }

def check_model_compatibility(model_info: ModelInfo) -> ValidationResult:
    """Check if a model can be used with Lilypad"""
    errors = []
    warnings = []
    requirements = {}
    
    try:
        # Check if model has required files
        if not model_info.siblings:
            errors.append("Model files not found")
        
        # Check for config file
        has_config = any(file.rfilename == 'config.json' for file in model_info.siblings)
        if not has_config:
            warnings.append("Model missing config.json - may have compatibility issues")
        
        # Check if model architecture is supported
        config = AutoConfig.from_pretrained(model_info.modelId)
        architecture = config.architectures[0] if config.architectures else None
        
        # List of known supported architectures
        supported_architectures = [
            'GPT2', 'BERT', 'RoBERTa', 'T5', 'ViT', 'Wav2Vec2', 'CLIPModel',
            'StableDiffusion', 'WhisperModel', 'DetrModel', 'ResNet'
        ]
        
        if architecture and not any(arch in architecture for arch in supported_architectures):
            warnings.append(f"Architecture {architecture} not explicitly tested - may have issues")
        
        # Check hardware requirements
        gpu_reqs = check_gpu_requirements(model_info)
        requirements['gpu'] = gpu_reqs
        
        # Memory requirements
        model_size = estimate_model_size(model_info)
        if model_size:
            requirements['ram'] = {
                'minimum': max(8, int(model_size * 4)),  # At least 4x model size
                'recommended': max(16, int(model_size * 6))  # 6x for safety
            }
        else:
            requirements['ram'] = {'minimum': 8, 'recommended': 16}
        
        # Storage requirements
        storage_needed = sum(f.size for f in model_info.siblings if hasattr(f, 'size')) / (1024 * 1024 * 1024)
        requirements['storage'] = max(5, int(storage_needed * 2))  # At least 5GB, or 2x model files
        
        # Check license
        if hasattr(model_info, 'license') and model_info.license:
            if 'proprietary' in model_info.license.lower():
                errors.append("Model has proprietary license - cannot be used")
            elif not any(license in model_info.license.lower() for license in ['mit', 'apache', 'bsd', 'cc', 'public']):
                warnings.append(f"License '{model_info.license}' may have restrictions")
        else:
            warnings.append("No license information found")
        
        # Check for pipeline tag
        if not any(tag.endswith('-pipeline') for tag in model_info.tags):
            warnings.append("Model may not support pipeline interface")
            
    except Exception as e:
        errors.append(f"Error validating model: {str(e)}")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        requirements=requirements
    )

def format_validation_message(result: ValidationResult) -> str:
    """Format validation results into a user-friendly message"""
    messages = []
    
    if not result.is_valid:
        messages.append("❌ Model validation failed:")
        for error in result.errors:
            messages.append(f"  • Error: {error}")
    else:
        messages.append("✅ Model validation passed")
    
    if result.warnings:
        messages.append("\n⚠️ Warnings:")
        for warning in result.warnings:
            messages.append(f"  • {warning}")
    
    messages.append("\n💻 System Requirements:")
    gpu_reqs = result.requirements.get('gpu', {})
    ram_reqs = result.requirements.get('ram', {})
    
    if gpu_reqs.get('required', False):
        messages.append(f"  • GPU: Required")
        messages.append(f"    - Minimum VRAM: {gpu_reqs['min_vram']}GB")
        messages.append(f"    - Recommended VRAM: {gpu_reqs['recommended_vram']}GB")
    else:
        messages.append("  • GPU: Optional")
    
    messages.append(f"  • RAM: {ram_reqs.get('minimum')}GB minimum, {ram_reqs.get('recommended')}GB recommended")
    messages.append(f"  • Storage: {result.requirements.get('storage', 5)}GB free space required")
    
    return "\n".join(messages)
