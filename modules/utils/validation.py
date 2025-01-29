"""
Validation utilities for checking model compatibility and requirements
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import re
from ..types import ModelType, TaskType

@dataclass
class ValidationResult:
    """Results of model validation"""
    is_valid: bool
    messages: List[str]
    requirements: Dict[str, any]
    missing_dependencies: Optional[List[str]] = None
    warnings: Optional[List[str]] = None

class ModelValidator:
    """Validates model compatibility with Lilypad infrastructure"""
    
    @staticmethod
    def validate_model(model_type: ModelType) -> ValidationResult:
        """
        Validate model compatibility
        
        Args:
            model_type: ModelType object with model information
            
        Returns:
            ValidationResult with validation results
        """
        messages = []
        warnings = []
        requirements = {}
        missing_deps = []
        is_valid = True
        
        # Check GPU requirements
        if model_type.requirements.min_gpu_memory > 0:
            requirements['gpu'] = True
            requirements['min_gpu_memory'] = f"{model_type.requirements.min_gpu_memory}MB"
            messages.append(
                f"Model requires GPU with at least {model_type.requirements.min_gpu_memory}MB memory"
            )
        
        # Check RAM requirements
        requirements['min_ram'] = f"{model_type.requirements.min_ram}MB"
        messages.append(
            f"Model requires at least {model_type.requirements.min_ram}MB RAM"
        )
        
        # Check CUDA version if needed
        if model_type.requirements.cuda_version:
            requirements['cuda_version'] = model_type.requirements.cuda_version
            messages.append(
                f"Model requires CUDA {model_type.requirements.cuda_version}"
            )
        
        # Framework compatibility
        requirements['framework'] = model_type.framework
        messages.append(f"Model uses {model_type.framework} framework")
        
        # Check quantization
        if model_type.quantization:
            requirements['quantization'] = model_type.quantization
            messages.append(f"Model uses {model_type.quantization} quantization")
            
            if model_type.quantization == '8bit' and model_type.framework != 'pytorch':
                warnings.append("8-bit quantization is only fully supported with PyTorch")
        
        # Task-specific validation
        task_validation = TaskValidator.validate_task(model_type.task)
        if not task_validation.is_valid:
            is_valid = False
            messages.extend(task_validation.messages)
            if task_validation.missing_dependencies:
                missing_deps.extend(task_validation.missing_dependencies)
        
        return ValidationResult(
            is_valid=is_valid,
            messages=messages,
            requirements=requirements,
            missing_dependencies=missing_deps if missing_deps else None,
            warnings=warnings if warnings else None
        )

class TaskValidator:
    """Validates task-specific requirements"""
    
    @staticmethod
    def validate_task(task: TaskType) -> ValidationResult:
        """
        Validate task compatibility
        
        Args:
            task: TaskType object with task information
            
        Returns:
            ValidationResult with validation results
        """
        messages = []
        missing_deps = []
        requirements = {}
        is_valid = True
        
        # Check input types
        for input_type in task.requirements.input_types:
            if input_type == 'image':
                try:
                    import PIL
                except ImportError:
                    missing_deps.append('pillow')
                    is_valid = False
            elif input_type == 'audio':
                try:
                    import librosa
                except ImportError:
                    missing_deps.append('librosa')
                    is_valid = False
            elif input_type == 'video':
                try:
                    import decord
                except ImportError:
                    missing_deps.append('decord')
                    is_valid = False
        
        # Add task-specific requirements
        requirements['input_types'] = task.requirements.input_types
        requirements['output_types'] = task.requirements.output_types
        requirements['batch_size'] = task.requirements.default_batch_size
        
        return ValidationResult(
            is_valid=is_valid,
            messages=messages,
            requirements=requirements,
            missing_dependencies=missing_deps if missing_deps else None
        )

class InputValidator:
    """Validates model inputs"""
    
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
            }
        }
        return requirements.get(input_type, {})
    
    @staticmethod
    def validate_input(input_path: str, input_type: str) -> ValidationResult:
        """Validate input data"""
        requirements = InputValidator.get_input_requirements(input_type)
        messages = []
        is_valid = True
        
        # Check file extension
        ext = input_path.split('.')[-1].lower()
        if ext not in requirements.get('formats', []):
            is_valid = False
            messages.append(
                f"Invalid file format: {ext}. Supported formats: {requirements['formats']}"
            )
        
        # Additional type-specific validation
        if input_type == 'image':
            try:
                from PIL import Image
                img = Image.open(input_path)
                if any(dim > requirements['max_size'][0] for dim in img.size):
                    messages.append(
                        f"Image dimensions exceed maximum size of {requirements['max_size']}"
                    )
            except Exception as e:
                is_valid = False
                messages.append(f"Failed to validate image: {str(e)}")
        
        elif input_type == 'audio':
            try:
                import librosa
                duration = librosa.get_duration(filename=input_path)
                if duration > requirements['max_duration']:
                    messages.append(
                        f"Audio duration exceeds maximum of {requirements['max_duration']} seconds"
                    )
            except Exception as e:
                is_valid = False
                messages.append(f"Failed to validate audio: {str(e)}")
        
        elif input_type == 'video':
            try:
                import cv2
                cap = cv2.VideoCapture(input_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps
                if duration > requirements['max_duration']:
                    messages.append(
                        f"Video duration exceeds maximum of {requirements['max_duration']} seconds"
                    )
                cap.release()
            except Exception as e:
                is_valid = False
                messages.append(f"Failed to validate video: {str(e)}")
        
        return ValidationResult(
            is_valid=is_valid,
            messages=messages,
            requirements=requirements
        )