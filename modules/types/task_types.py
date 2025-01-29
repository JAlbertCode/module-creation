"""
Task type definitions and detection for Hugging Face models
"""

[Previous content remains the same until detect_task_type function]

def detect_task_type(model_info: Dict[str, Any]) -> TaskType:
    """
    Detect task type from model information
    
    Args:
        model_info: Dictionary containing model information from Hugging Face
        
    Returns:
        TaskType object with complete configuration
    """
    # First check pipeline tag
    if pipeline_tag := model_info.get('pipeline_tag'):
        for category, tasks in TASK_CATEGORIES.items():
            if pipeline_tag in tasks:
                requirements = TASK_REQUIREMENTS.get(pipeline_tag, TASK_REQUIREMENTS['text-classification'])
                return TaskType(
                    name=pipeline_tag,
                    category=category,
                    requirements=requirements,
                    default_parameters=get_default_parameters(pipeline_tag),
                    metrics=get_default_metrics(pipeline_tag),
                    example_models=get_example_models(pipeline_tag)
                )
    
    # Check model tags
    tags = model_info.get('tags', [])
    for category, tasks in TASK_CATEGORIES.items():
        for task in tasks:
            if any(task in tag.lower() for tag in tags):
                requirements = TASK_REQUIREMENTS.get(task, TASK_REQUIREMENTS['text-classification'])
                return TaskType(
                    name=task,
                    category=category,
                    requirements=requirements,
                    default_parameters=get_default_parameters(task),
                    metrics=get_default_metrics(task),
                    example_models=get_example_models(task)
                )
    
    # Check model architecture
    if arch := model_info.get('architectures', [None])[0]:
        arch_lower = arch.lower()
        arch_mappings = {
            'bert': ('text-classification', 'text'),
            'gpt': ('text-generation', 'text'),
            't5': ('text-to-text', 'text'),
            'vit': ('image-classification', 'image'),
            'yolo': ('object-detection', 'image'),
            'wav2vec': ('automatic-speech-recognition', 'audio'),
            'resnet': ('image-classification', 'image'),
            'clip': ('image-text-retrieval', 'multimodal')
        }
        
        for arch_prefix, (task, category) in arch_mappings.items():
            if arch_prefix in arch_lower:
                requirements = TASK_REQUIREMENTS.get(task, TASK_REQUIREMENTS['text-classification'])
                return TaskType(
                    name=task,
                    category=category,
                    requirements=requirements,
                    default_parameters=get_default_parameters(task),
                    metrics=get_default_metrics(task),
                    example_models=get_example_models(task)
                )
    
    # Default to text classification if no task can be determined
    return TaskType(
        name='text-classification',
        category='text',
        requirements=TASK_REQUIREMENTS['text-classification'],
        default_parameters=get_default_parameters('text-classification'),
        metrics=get_default_metrics('text-classification'),
        example_models=get_example_models('text-classification')
    )

def get_default_parameters(task: str) -> Dict[str, Any]:
    """Get default parameters for a task"""
    base_params = {
        'batch_size': TASK_REQUIREMENTS[task].default_batch_size,
        'device': 'cuda' if TASK_REQUIREMENTS[task].requires_gpu else 'cpu'
    }
    
    task_specific_params = {
        'text-generation': {
            'max_length': 128,
            'temperature': 0.7,
            'top_p': 0.9,
            'num_return_sequences': 1
        },
        'text-to-image': {
            'num_inference_steps': 50,
            'guidance_scale': 7.5,
            'negative_prompt': None
        },
        'automatic-speech-recognition': {
            'chunk_length_s': 30,
            'stride_length_s': 5
        },
        'object-detection': {
            'confidence_threshold': 0.5,
            'nms_threshold': 0.45
        }
    }
    
    base_params.update(task_specific_params.get(task, {}))
    return base_params

def get_default_metrics(task: str) -> List[str]:
    """Get default evaluation metrics for a task"""
    metrics_mapping = {
        'text-classification': ['accuracy', 'f1', 'precision', 'recall'],
        'text-generation': ['perplexity', 'bleu', 'rouge'],
        'image-classification': ['accuracy', 'top_k_accuracy'],
        'object-detection': ['map', 'mar', 'precision', 'recall'],
        'automatic-speech-recognition': ['wer', 'cer'],
        'text-to-image': ['fid', 'clip_score', 'inception_score']
    }
    return metrics_mapping.get(task, ['accuracy'])

def get_example_models(task: str) -> List[str]:
    """Get example models for a task"""
    examples_mapping = {
        'text-classification': [
            'bert-base-uncased',
            'roberta-base',
            'distilbert-base-uncased'
        ],
        'text-generation': [
            'gpt2',
            'facebook/opt-350m',
            'EleutherAI/pythia-160m'
        ],
        'image-classification': [
            'google/vit-base-patch16-224',
            'microsoft/resnet-50',
            'facebook/convnext-tiny-224'
        ],
        'object-detection': [
            'facebook/detr-resnet-50',
            'hustvl/yolos-tiny',
            'microsoft/conditional-detr-resnet-50'
        ],
        'automatic-speech-recognition': [
            'facebook/wav2vec2-base-960h',
            'openai/whisper-tiny',
            'facebook/hubert-base-ls960'
        ],
        'text-to-image': [
            'runwayml/stable-diffusion-v1-5',
            'CompVis/stable-diffusion-v1-4',
            'stabilityai/stable-diffusion-2-1'
        ]
    }
    return examples_mapping.get(task, [])