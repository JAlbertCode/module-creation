"""Template validation and verification"""

from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]

class TemplateValidator:
    """Validate configuration templates"""
    
    def __init__(self):
        self.required_fields = {
            'name': str,
            'description': str,
            'model_type': str,
            'config': dict
        }
        
        self.valid_model_types = [
            'text-classification',
            'image-classification',
            'object-detection',
            'question-answering'
        ]
        
        self.resource_limits = {
            'cpu': {'min': 1, 'max': 32},
            'memory': {'min': '1Gi', 'max': '64Gi'},
            'gpu': {'min': 0, 'max': 4}
        }
    
    def validate(self, template: Dict[str, Any]) -> ValidationResult:
        """Validate a template configuration"""
        errors = []
        warnings = []
        
        # Check required fields
        for field, field_type in self.required_fields.items():
            if field not in template:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(template[field], field_type):
                errors.append(f"Invalid type for {field}: expected {field_type.__name__}")
        
        if errors:
            return ValidationResult(False, errors, warnings)
        
        # Validate model type
        if template['model_type'] not in self.valid_model_types:
            errors.append(f"Invalid model_type: {template['model_type']}")
        
        # Validate resources
        resources = template['config'].get('resources', {})
        self._validate_resources(resources, errors, warnings)
        
        # Validate model configuration
        model_config = template['config'].get('model', {})
        self._validate_model_config(model_config, template['model_type'], errors, warnings)
        
        # Validate optimizations
        optimizations = template['config'].get('optimization', {})
        self._validate_optimizations(optimizations, resources, warnings)
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    def _validate_resources(self, resources: Dict[str, Any], errors: List[str], warnings: List[str]):
        """Validate resource configurations"""
        if 'cpu' in resources:
            cpu = resources['cpu']
            if not isinstance(cpu, int) or cpu < self.resource_limits['cpu']['min']:
                errors.append(f"CPU cores must be at least {self.resource_limits['cpu']['min']}")
            elif cpu > self.resource_limits['cpu']['max']:
                warnings.append(f"High CPU allocation: {cpu} cores")
        
        if 'memory' in resources:
            memory = resources['memory']
            if not isinstance(memory, str) or not memory.endswith('Gi'):
                errors.append("Memory must be specified in Gi (e.g., '4Gi')")
            else:
                mem_value = int(memory[:-2])
                if mem_value < int(self.resource_limits['memory']['min'][:-2]):
                    errors.append(f"Memory must be at least {self.resource_limits['memory']['min']}")
                elif mem_value > int(self.resource_limits['memory']['max'][:-2]):
                    warnings.append(f"High memory allocation: {memory}")
        
        if 'gpu' in resources and resources['gpu'] is not None:
            gpu = resources['gpu']
            if not isinstance(gpu, int) or gpu < 0:
                errors.append("GPU count must be a non-negative integer")
            elif gpu > self.resource_limits['gpu']['max']:
                warnings.append(f"High GPU allocation: {gpu} GPUs")
    
    def _validate_model_config(self, config: Dict[str, Any], model_type: str, errors: List[str], warnings: List[str]):
        """Validate model-specific configurations"""
        if 'batch_size' in config:
            batch_size = config['batch_size']
            if not isinstance(batch_size, int) or batch_size < 1:
                errors.append("Batch size must be a positive integer")
            elif batch_size > 64:
                warnings.append(f"Large batch size: {batch_size}")
        
        if model_type == 'text-classification':
            if 'max_length' in config:
                max_length = config['max_length']
                if not isinstance(max_length, int) or max_length < 1:
                    errors.append("max_length must be a positive integer")
                elif max_length > 2048:
                    warnings.append(f"Large max_length: {max_length}")
    
    def _validate_optimizations(self, optimizations: Dict[str, Any], resources: Dict[str, Any], warnings: List[str]):
        """Validate optimization settings"""
        if optimizations.get('enable_tensorrt', False) and resources.get('gpu', 0) == 0:
            warnings.append("TensorRT enabled but no GPU specified")
        
        if optimizations.get('dynamic_batching', False) and resources.get('cpu', 1) < 2:
            warnings.append("Dynamic batching may need more CPU cores")
        
        cache_size = optimizations.get('cache_size', '')
        if cache_size:
            if cache_size.endswith('Gi'):
                cache_gb = int(cache_size[:-2])
                memory_gb = int(resources.get('memory', '0Gi')[:-2])
                if cache_gb > memory_gb / 2:
                    warnings.append(f"Cache size ({cache_size}) is more than 50% of total memory")

def validate_template(template: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """Convenience function to validate a template"""
    validator = TemplateValidator()
    result = validator.validate(template)
    return result.is_valid, result.errors, result.warnings