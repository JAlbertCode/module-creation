"""Model configuration options"""

def get_model_configs(model_type):
    """Get available configurations for a model type"""
    base_configs = {
        'precision': {
            'type': 'select',
            'label': 'Model Precision',
            'options': ['float32', 'float16', 'int8'],
            'default': 'float32',
            'description': 'Lower precision uses less memory but may reduce accuracy'
        },
        'batch_size': {
            'type': 'number',
            'label': 'Batch Size',
            'min': 1,
            'max': 64,
            'default': 1,
            'description': 'Number of inputs to process at once'
        },
        'gpu_memory': {
            'type': 'number',
            'label': 'GPU Memory Limit (GB)',
            'min': 4,
            'max': 24,
            'default': 8,
            'description': 'Maximum GPU memory to use'
        }
    }

    # Add task-specific configurations
    task_configs = {
        'text-generation': {
            'max_length': {
                'type': 'number',
                'label': 'Maximum Length',
                'min': 1,
                'max': 2048,
                'default': 128,
                'description': 'Maximum output length'
            },
            'temperature': {
                'type': 'number',
                'label': 'Temperature',
                'min': 0.1,
                'max': 2.0,
                'default': 1.0,
                'description': 'Randomness in generation'
            }
        },
        'image-to-text': {
            'num_return_sequences': {
                'type': 'number',
                'label': 'Number of Captions',
                'min': 1,
                'max': 10,
                'default': 1,
                'description': 'Number of captions to generate'
            }
        }
    }

    # Get task-specific configs if available
    extra_configs = task_configs.get(model_type['task'], {})
    
    # Combine base configs with task-specific ones
    return {**base_configs, **extra_configs}