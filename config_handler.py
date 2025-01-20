from typing import Dict, Any
import yaml

class ConfigHandler:
    """Handle module configuration and customization"""
    
    def __init__(self, model_info, model_type):
        self.model_info = model_info
        self.model_type = model_type
        
    def generate_base_config(self) -> Dict[str, Any]:
        """Generate base configuration"""
        return {
            'model': {
                'id': self.model_info.id,
                'type': self.model_type['task'],
                'batch_size': 1,
                'max_length': 512 if 'text' in self.model_type['task'] else None
            },
            'resources': {
                'cpu': 1,
                'memory': '4Gi',
                'gpu': 1 if self.needs_gpu() else None
            },
            'input': {
                'format': self.model_type['input_type'],
                'preprocessing': self.get_default_preprocessing()
            },
            'output': {
                'format': 'json',
                'path': '/outputs/result.json'
            }
        }
    
    def needs_gpu(self) -> bool:
        """Determine if model needs GPU"""
        model_size = self.model_info.size_in_bytes / (1024 * 1024 * 1024)  # Size in GB
        return model_size > 1.0
    
    def get_default_preprocessing(self) -> Dict[str, Any]:
        """Get default preprocessing config based on model type"""
        if self.model_type['input_type'] == 'image':
            return {
                'resize': [224, 224],
                'normalize': True,
                'color_mode': 'RGB'
            }
        elif self.model_type['input_type'] == 'text':
            return {
                'max_length': 512,
                'truncation': True,
                'padding': 'max_length'
            }
        return {}
    
    def update_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration with user preferences"""
        base_config = self.generate_base_config()
        
        # Deep update of configuration
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d:
                    d[k] = deep_update(d[k], v)
                else:
                    d[k] = v
            return d
            
        return deep_update(base_config, user_config)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration values"""
        try:
            # Validate resource requirements
            if config['resources']['memory'].endswith('Gi'):
                memory = int(config['resources']['memory'][:-2])
                if memory < 2 or memory > 32:
                    raise ValueError("Memory must be between 2Gi and 32Gi")
            
            if config['resources']['cpu'] < 1 or config['resources']['cpu'] > 8:
                raise ValueError("CPU cores must be between 1 and 8")
            
            # Validate batch size
            if config['model']['batch_size'] < 1:
                raise ValueError("Batch size must be positive")
            
            # Validate model-specific settings
            if self.model_type['input_type'] == 'text':
                max_length = config['input']['preprocessing']['max_length']
                if max_length < 1 or max_length > 2048:
                    raise ValueError("max_length must be between 1 and 2048")
            
            elif self.model_type['input_type'] == 'image':
                resize = config['input']['preprocessing']['resize']
                if not all(1 <= dim <= 4096 for dim in resize):
                    raise ValueError("Image dimensions must be between 1 and 4096")
            
            return True
            
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {str(e)}")
    
    def generate_yaml(self, config: Dict[str, Any]) -> str:
        """Generate YAML configuration file"""
        # Convert internal config to Lilypad format
        lilypad_config = {
            'name': self.model_info.id.split('/')[-1],
            'version': '1.0.0',
            'description': f"Hugging Face {self.model_type['task']} model deployment",
            'resources': config['resources'],
            'input': [{
                'name': 'INPUT_PATH',
                'description': f"Path to the input file ({config['input']['format']})",
                'type': 'string',
                'required': True
            }],
            'output': [{
                'name': 'result',
                'description': 'Model output',
                'type': 'file',
                'path': config['output']['path']
            }]
        }
        
        return yaml.dump(lilypad_config, default_flow_style=False)
    
    def generate_env_vars(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Generate environment variables for the container"""
        env_vars = {
            'MODEL_ID': self.model_info.id,
            'BATCH_SIZE': str(config['model']['batch_size']),
        }
        
        # Add model-specific variables
        if 'text' in self.model_type['task']:
            env_vars['MAX_LENGTH'] = str(config['model']['max_length'])
        
        if config['resources']['gpu']:
            env_vars['CUDA_VISIBLE_DEVICES'] = '0'
        
        return env_vars