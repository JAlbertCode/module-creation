"""Validation utilities for model compatibility and inputs"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import os
import re
import logging
import torch
from huggingface_hub.hf_api import ModelInfo
from .. import model_types

@dataclass
class ValidationResult:
    is_valid: bool
    requirements: Dict[str, Any]  # Required resources/dependencies
    errors: List[str]  # List of validation errors
    warnings: List[str]  # List of warning messages
    missing_dependencies: List[str]  # Missing required dependencies

def check_model_compatibility(model_info: ModelInfo) -> ValidationResult:
    """Check if a model is compatible with Lilypad"""
    
    errors = []
    warnings = []
    missing_dependencies = []
    
    try:
        # Detect model type
        model_type = model_types.detect_model_type(model_info)
        requirements = model_types.get_task_requirements(model_type.task)
        
        # Check if model files can be downloaded
        if not model_info.siblings:
            errors.append("No model files found - model may be gated or private")
            
        # Check model size
        if model_type.model_size > 100000:  # 100GB
            errors.append("Model is too large (>100GB)")
        elif model_type.model_size > 50000:  # 50GB
            warnings.append("Model is very large (>50GB) - may have long loading times")
            
        # Check hardware requirements
        if model_type.requires_gpu:
            if not torch.cuda.is_available():
                missing_dependencies.append("CUDA GPU")
            else:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                min_vram = requirements.get("min_vram", 4096)  # Default 4GB
                if gpu_memory < min_vram * 1024 * 1024:  # Convert MB to bytes
                    warnings.append(f"Available GPU memory ({gpu_memory/(1024*1024*1024):.1f}GB) is less than recommended ({min_vram/1024:.1f}GB)")
        
        # Check system requirements
        required_packages = requirements.get("system_packages", [])
        for package in required_packages:
            if not check_system_package(package):
                missing_dependencies.append(f"System package: {package}")
                
        # Check Python package compatibility
        python_packages = requirements.get("python_packages", [])
        for package in python_packages:
            if not check_python_package(package):
                missing_dependencies.append(f"Python package: {package}")
                
        # Check CUDA version if GPU is required
        if model_type.requires_gpu:
            cuda_version = requirements.get("cuda_version", ">=11.7")
            if not check_cuda_version(cuda_version):
                missing_dependencies.append(f"CUDA {cuda_version}")
                
        is_valid = len(errors) == 0 and len(missing_dependencies) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            requirements=requirements,
            errors=errors,
            warnings=warnings,
            missing_dependencies=missing_dependencies
        )
        
    except Exception as e:
        logging.error(f"Error validating model: {str(e)}")
        return ValidationResult(
            is_valid=False,
            requirements={},
            errors=[f"Validation error: {str(e)}"],
            warnings=[],
            missing_dependencies=[]
        )

def check_system_package(package: str) -> bool:
    """Check if a system package is installed"""
    # Simple check using which command
    return os.system(f"which {package} > /dev/null 2>&1") == 0

def check_python_package(package: str) -> bool:
    """Check if a Python package is installed"""
    try:
        __import__(package.split('>=')[0].split('==')[0])
        return True
    except ImportError:
        return False

def check_cuda_version(required_version: str) -> bool:
    """Check if CUDA version meets requirements"""
    if not torch.cuda.is_available():
        return False
        
    # Parse required version
    version_match = re.match(r'[>=<]*(\d+\.\d+)', required_version)
    if not version_match:
        return False
    required = float(version_match.group(1))
    
    # Get current version
    current = torch.version.cuda
    if current is None:
        return False
    
    current = float('.'.join(current.split('.')[:2]))
    
    # Compare versions based on requirement
    if '>=' in required_version:
        return current >= required
    elif '>' in required_version:
        return current > required
    elif '<=' in required_version:
        return current <= required
    elif '<' in required_version:
        return current < required
    else:
        return current == required

class InputValidator:
    """Validates inputs for different model types"""
    
    @staticmethod
    def validate_text_input(text: str, max_length: int = 2048) -> bool:
        """Validate text input"""
        if not isinstance(text, str):
            return False
        if len(text.strip()) == 0:
            return False
        if len(text) > max_length:
            return False
        return True
    
    @staticmethod
    def validate_image_input(image_path: str, supported_formats: List[str] = None) -> bool:
        """Validate image input"""
        if supported_formats is None:
            supported_formats = ['jpg', 'jpeg', 'png', 'bmp']
            
        if not os.path.exists(image_path):
            return False
            
        extension = image_path.lower().split('.')[-1]
        if extension not in supported_formats:
            return False
            
        return True
    
    @staticmethod
    def validate_audio_input(audio_path: str, supported_formats: List[str] = None) -> bool:
        """Validate audio input"""
        if supported_formats is None:
            supported_formats = ['wav', 'mp3', 'flac']
            
        if not os.path.exists(audio_path):
            return False
            
        extension = audio_path.lower().split('.')[-1]
        if extension not in supported_formats:
            return False
            
        return True
    
    @staticmethod
    def get_input_requirements(input_type: str) -> Dict[str, Any]:
        """Get input requirements for a given type"""
        requirements = {
            "text": {
                "max_length": 2048,
                "supported_formats": ["txt", "json"],
                "example": "Enter your text prompt here"
            },
            "image": {
                "supported_formats": ["jpg", "jpeg", "png", "bmp"],
                "max_size": "4096x4096",
                "example": "path/to/image.jpg"
            },
            "audio": {
                "supported_formats": ["wav", "mp3", "flac"],
                "max_duration": "30s",
                "sample_rate": "16000Hz",
                "example": "path/to/audio.wav"
            },
            "multimodal": {
                "supported_formats": {
                    "image": ["jpg", "jpeg", "png"],
                    "text": ["txt", "json"]
                },
                "example": {
                    "image": "path/to/image.jpg",
                    "text": "Question about the image"
                }
            }
        }
        return requirements.get(input_type, {})

def format_validation_message(result: ValidationResult) -> str:
    """Format validation result into a user-friendly message"""
    messages = []
    
    if result.is_valid:
        messages.append("✅ Model is compatible with Lilypad")
    else:
        messages.append("❌ Model is not compatible with Lilypad")
    
    if result.errors:
        messages.append("\nErrors:")
        for error in result.errors:
            messages.append(f"  • {error}")
    
    if result.missing_dependencies:
        messages.append("\nMissing Dependencies:")
        for dep in result.missing_dependencies:
            messages.append(f"  • {dep}")
    
    if result.warnings:
        messages.append("\nWarnings:")
        for warning in result.warnings:
            messages.append(f"  • {warning}")
    
    if result.requirements:
        messages.append("\nRequirements:")
        for key, value in result.requirements.items():
            if isinstance(value, (list, dict)):
                messages.append(f"  • {key}:")
                for item in value:
                    messages.append(f"    - {item}")
            else:
                messages.append(f"  • {key}: {value}")
    
    return "\n".join(messages)