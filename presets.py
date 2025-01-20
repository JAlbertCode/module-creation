"""Configuration presets for different use cases"""

PRESETS = {
    'text-classification': {
        'minimal': {
            'resources': {
                'cpu': 1,
                'memory': '2Gi',
                'gpu': None
            },
            'model': {
                'batch_size': 1,
                'max_length': 128
            },
            'description': 'Minimum resource usage, suitable for testing'
        },
        'standard': {
            'resources': {
                'cpu': 2,
                'memory': '4Gi',
                'gpu': None
            },
            'model': {
                'batch_size': 8,
                'max_length': 512
            },
            'description': 'Balanced performance for production use'
        },
        'high-performance': {
            'resources': {
                'cpu': 4,
                'memory': '8Gi',
                'gpu': 1
            },
            'model': {
                'batch_size': 16,
                'max_length': 1024
            },
            'description': 'Maximum performance with GPU acceleration'
        }
    },
    
    'image-classification': {
        'minimal': {
            'resources': {
                'cpu': 1,
                'memory': '2Gi',
                'gpu': None
            },
            'model': {
                'batch_size': 1
            },
            'input': {
                'preprocessing': {
                    'resize': [224, 224],
                    'normalize': True
                }
            },
            'description': 'Minimal setup for basic image classification'
        },
        'standard': {
            'resources': {
                'cpu': 2,
                'memory': '4Gi',
                'gpu': 1
            },
            'model': {
                'batch_size': 4
            },
            'input': {
                'preprocessing': {
                    'resize': [384, 384],
                    'normalize': True
                }
            },
            'description': 'Standard configuration for most use cases'
        },
        'high-quality': {
            'resources': {
                'cpu': 4,
                'memory': '8Gi',
                'gpu': 1
            },
            'model': {
                'batch_size': 8
            },
            'input': {
                'preprocessing': {
                    'resize': [512, 512],
                    'normalize': True
                }
            },
            'description': 'High-quality image processing with GPU'
        }
    },
    
    'object-detection': {
        'minimal': {
            'resources': {
                'cpu': 2,
                'memory': '4Gi',
                'gpu': None
            },
            'model': {
                'batch_size': 1,
                'confidence_threshold': 0.5
            },
            'description': 'Basic object detection setup'
        },
        'standard': {
            'resources': {
                'cpu': 4,
                'memory': '8Gi',
                'gpu': 1
            },
            'model': {
                'batch_size': 2,
                'confidence_threshold': 0.3
            },
            'description': 'Balanced performance for general use'
        },
        'high-precision': {
            'resources': {
                'cpu': 4,
                'memory': '16Gi',
                'gpu': 1
            },
            'model': {
                'batch_size': 4,
                'confidence_threshold': 0.2
            },
            'description': 'High-precision detection with GPU'
        }
    },
    
    'question-answering': {
        'minimal': {
            'resources': {
                'cpu': 1,
                'memory': '2Gi',
                'gpu': None
            },
            'model': {
                'batch_size': 1,
                'max_length': 384,
                'doc_stride': 128
            },
            'description': 'Basic QA setup for short contexts'
        },
        'standard': {
            'resources': {
                'cpu': 2,
                'memory': '4Gi',
                'gpu': None
            },
            'model': {
                'batch_size': 4,
                'max_length': 512,
                'doc_stride': 256
            },
            'description': 'Standard configuration for most use cases'
        },
        'long-context': {
            'resources': {
                'cpu': 4,
                'memory': '8Gi',
                'gpu': 1
            },
            'model': {
                'batch_size': 2,
                'max_length': 1024,
                'doc_stride': 512
            },
            'description': 'Optimized for long context processing'
        }
    }
}

def get_preset(model_type: str, preset_name: str = 'standard'):
    """Get a specific preset configuration"""
    model_presets = PRESETS.get(model_type, PRESETS['text-classification'])
    return model_presets.get(preset_name, model_presets['standard'])

def get_available_presets(model_type: str):
    """Get all available presets for a model type"""
    return PRESETS.get(model_type, PRESETS['text-classification'])