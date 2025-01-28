"""
Configuration management for Lilypad modules
"""

from typing import Dict, List, Optional
import json

def get_model_configs(model_type: Dict) -> Dict:
    """
    Get available configurations for a model type
    
    Args:
        model_type: Dictionary containing model type information
        
    Returns:
        Dictionary of configuration options
    """
    base_configs = {
        'batch_size': {
            'type': 'integer',
            'default': 1,
            'min': 1,
            'max': 32,
            'description': 'Number of inputs to process at once'
        },
        'device': {
            'type': 'string',
            'default': 'cuda',
            'options': ['cuda', 'cpu'],
            'description': 'Device to run inference on'
        }
    }
    
    task_specific_configs = {
        'text-classification': {
            'max_length': {
                'type': 'integer',
                'default': 512,
                'min': 32,
                'max': 2048,
                'description': 'Maximum input text length'
            }
        },
        'text-generation': {
            'max_length': {
                'type': 'integer',
                'default': 100,
                'min': 10,
                'max': 4096,
                'description': 'Maximum generated text length'
            },
            'temperature': {
                'type': 'float',
                'default': 0.7,
                'min': 0.1,
                'max': 2.0,
                'description': 'Sampling temperature'
            },
            'top_p': {
                'type': 'float',
                'default': 0.9,
                'min': 0.0,
                'max': 1.0,
                'description': 'Nucleus sampling parameter'
            }
        },
        'image-classification': {
            'resize_mode': {
                'type': 'string',
                'default': 'pad',
                'options': ['pad', 'crop', 'resize'],
                'description': 'How to handle input image resizing'
            }
        },
        'object-detection': {
            'confidence_threshold': {
                'type': 'float',
                'default': 0.5,
                'min': 0.0,
                'max': 1.0,
                'description': 'Minimum confidence score for detections'
            },
            'nms_threshold': {
                'type': 'float',
                'default': 0.45,
                'min': 0.0,
                'max': 1.0,
                'description': 'Non-maximum suppression threshold'
            }
        },
        'automatic-speech-recognition': {
            'chunk_length_seconds': {
                'type': 'float',
                'default': 30.0,
                'min': 1.0,
                'max': 120.0,
                'description': 'Length of audio chunks to process'
            }
        },
        'video-classification': {
            'frames_per_second': {
                'type': 'integer',
                'default': 1,
                'min': 1,
                'max': 30,
                'description': 'Number of frames to sample per second'
            }
        },
        'visual-question-answering': {
            'max_answer_length': {
                'type': 'integer',
                'default': 64,
                'min': 16,
                'max': 256,
                'description': 'Maximum length of generated answer'
            }
        }
    }
    
    # Get task-specific configs
    task = model_type.get('task', 'text-classification')
    configs = base_configs.copy()
    configs.update(task_specific_configs.get(task, {}))
    
    # Add model-specific memory requirements
    if model_type.get('special_requirements', {}).get('gpu'):
        configs['gpu_memory'] = {
            'type': 'string',
            'default': model_type['special_requirements'].get('min_gpu_memory', '16GB'),
            'description': 'Required GPU memory'
        }
    
    return configs

def generate_module_template(model_id: str, model_type: Dict, configs: Dict) -> str:
    """
    Generate Lilypad module template
    
    Args:
        model_id: Hugging Face model ID
        model_type: Dictionary containing model type information
        configs: Dictionary of configuration options
        
    Returns:
        String containing module template JSON
    """
    template = {
        'machine': {
            'gpu': 1 if model_type.get('special_requirements', {}).get('gpu') else 0,
            'cpu': 1000,
            'ram': 8000
        },
        'job': {
            'APIVersion': 'V1beta1',
            'Spec': {
                'Deal': {
                    'Concurrency': 1
                },
                'Docker': {
                    'Entrypoint': ['python', '/workspace/run_inference.py'],
                    'WorkingDirectory': '/workspace',
                    'EnvironmentVariables': [
                        f'MODEL_ID={model_id}'
                    ],
                    'Image': f'huggingface/{model_id.split("/")[-1]}'
                },
                'Engine': 'Docker',
                'Network': {
                    'Type': 'None'
                },
                'Outputs': [
                    {
                        'Name': 'outputs',
                        'Path': '/outputs'
                    }
                ],
                'PublisherSpec': {
                    'Type': 'ipfs'
                },
                'Resources': {
                    'GPU': str(1 if model_type.get('special_requirements', {}).get('gpu') else 0)
                },
                'Timeout': 1800
            }
        }
    }
    
    # Add config-specific environment variables
    for config_name, config_info in configs.items():
        template['job']['Spec']['Docker']['EnvironmentVariables'].append(
            f'{config_name.upper()}={config_info["default"]}'
        )
    
    return json.dumps(template, indent=2)

def generate_readme(model_id: str, model_type: Dict, validation_result: Dict, configs: Dict) -> str:
    """
    Generate README.md content for the module
    
    Args:
        model_id: Hugging Face model ID
        model_type: Dictionary containing model type information
        validation_result: Results from model validation
        configs: Dictionary of configuration options
        
    Returns:
        String containing README content
    """
    sections = [
        f"# {model_id.split('/')[-1]} Lilypad Module\n\n",
        "## Overview\n\n",
        f"This module runs the [{model_id}](https://huggingface.co/{model_id}) model from Hugging Face on the Lilypad network.\n\n",
        "## Requirements\n\n",
        "".join(f"- {key}: {value}\n" for key, value in validation_result['requirements'].items()),
        "\n## Configuration\n\n",
        "The following configuration options are available:\n\n",
        "".join(f"### {config_name}\n{config_info['description']}\n- Type: {config_info['type']}\n- Default: {config_info['default']}\n" + 
               ("- Options: " + ", ".join(config_info['options']) + "\n" if 'options' in config_info else 
                f"- Range: {config_info.get('min', 'N/A')} - {config_info.get('max', 'N/A')}\n") 
               for config_name, config_info in configs.items()),
        "\n## Usage\n\n",
        "```bash\n",
        f"lilypad run {model_id}:latest \\\n",
        "".join(f"  -i {config_name.upper()}={config_info['default']} \\\n" for config_name, config_info in configs.items()),
        "```\n",
        "\n## Output\n\n",
        "The module will generate output in the following format:\n\n",
        "```json\n{\n  \"results\": [\n    {\n      \"filename\": \"input_file\",\n      \"result\": {} // Model-specific output\n    }\n  ]\n}\n```\n",
        "\n## Development\n\n",
        "To modify this module:\n\n",
        "1. Update the configuration in `lilypad_module.json.tmpl`\n",
        "2. Modify the inference code in `run_inference.py`\n",
        "3. Update dependencies in `requirements.txt`\n\n",
        "## License\n\n",
        f"This module is provided under the same license as the original model: {validation_result['requirements'].get('license', 'Unknown')}\n"
    ]
    
    return ''.join(sections)