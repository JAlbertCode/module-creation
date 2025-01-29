"""Model type detection and configuration"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from huggingface_hub.hf_api import ModelInfo

@dataclass
class ModelType:
    task: str  # Primary task (classification, generation, etc)
    input_type: str  # Input data type (text, image, audio, etc)
    output_type: str  # Output data type
    framework: str  # PyTorch, TensorFlow, etc
    requires_gpu: bool
    memory_requirements: int  # Minimum RAM in MB
    model_size: int  # Model size in MB
    supported_tasks: List[str]  # All supported tasks

def detect_framework(model_info: ModelInfo) -> str:
    """Detect model framework from files and tags"""
    if any(f.endswith('.pt') or f.endswith('.bin') for f in model_info.siblings):
        return 'pytorch'
    elif any(f.endswith('.h5') or f.endswith('.keras') for f in model_info.siblings):
        return 'tensorflow'
    else:
        # Default to PyTorch as most HF models use it
        return 'pytorch'

def estimate_model_size(model_info: ModelInfo) -> int:
    """Estimate model size in MB from file sizes"""
    total_size = sum(s.size for s in model_info.siblings if s.size is not None)
    return total_size // (1024 * 1024)  # Convert to MB

def estimate_memory_requirements(model_size: int) -> int:
    """Estimate minimum RAM requirements based on model size"""
    # Rough estimate: need 2-3x model size for loading and processing
    return model_size * 3

def detect_model_type(model_info: ModelInfo) -> ModelType:
    """Detect model type and configuration from model info"""
    
    # Get tags and pipeline tag
    tags = model_info.tags if model_info.tags else []
    pipeline_tag = model_info.pipeline_tag if model_info.pipeline_tag else None
    
    # Initialize variables
    task = None
    input_type = None
    output_type = None
    supported_tasks = []
    requires_gpu = False
    
    # Detect task and type from pipeline tag
    if pipeline_tag:
        if pipeline_tag in ['text-classification', 'token-classification']:
            task = pipeline_tag
            input_type = 'text'
            output_type = 'classification'
            supported_tasks = [task]
        elif pipeline_tag in ['text-generation', 'text2text-generation']:
            task = 'text-generation'
            input_type = 'text'
            output_type = 'text'
            supported_tasks = ['text-generation', 'chat']
        elif pipeline_tag in ['image-classification']:
            task = 'image-classification'
            input_type = 'image'
            output_type = 'classification'
            supported_tasks = ['image-classification']
        elif pipeline_tag in ['image-to-text', 'image-captioning']:
            task = 'image-captioning'
            input_type = 'image'
            output_type = 'text'
            supported_tasks = ['image-captioning', 'visual-question-answering']
        elif pipeline_tag in ['text-to-image']:
            task = 'text-to-image'
            input_type = 'text'
            output_type = 'image'
            supported_tasks = ['text-to-image', 'image-variation']
        elif pipeline_tag in ['audio-classification']:
            task = 'audio-classification'
            input_type = 'audio'
            output_type = 'classification'
            supported_tasks = ['audio-classification']
        elif pipeline_tag in ['automatic-speech-recognition']:
            task = 'speech-recognition'
            input_type = 'audio'
            output_type = 'text'
            supported_tasks = ['speech-recognition']
        elif pipeline_tag in ['text-to-speech']:
            task = 'text-to-speech'
            input_type = 'text'
            output_type = 'audio'
            supported_tasks = ['text-to-speech']
        elif pipeline_tag in ['translation']:
            task = 'translation'
            input_type = 'text'
            output_type = 'text'
            supported_tasks = ['translation']
        elif pipeline_tag in ['summarization']:
            task = 'summarization'
            input_type = 'text'
            output_type = 'text'
            supported_tasks = ['summarization']
    
    # If no pipeline tag, try to detect from model tags
    if not task:
        if 'text-generation' in tags:
            task = 'text-generation'
            input_type = 'text'
            output_type = 'text'
            supported_tasks = ['text-generation', 'chat']
        elif 'vision' in tags:
            task = 'image-classification'
            input_type = 'image'
            output_type = 'classification'
            supported_tasks = ['image-classification']
        elif 'audio' in tags:
            task = 'audio-classification'
            input_type = 'audio'
            output_type = 'classification'
            supported_tasks = ['audio-classification']
        elif 'multimodal' in tags:
            task = 'visual-question-answering'
            input_type = 'multimodal'
            output_type = 'text'
            supported_tasks = ['visual-question-answering', 'image-captioning']
        else:
            # Default to text processing if unclear
            task = 'text-generation'
            input_type = 'text'
            output_type = 'text'
            supported_tasks = ['text-generation']

    # Detect framework
    framework = detect_framework(model_info)
    
    # Calculate model size
    model_size = estimate_model_size(model_info)
    
    # Estimate memory requirements
    memory_requirements = estimate_memory_requirements(model_size)
    
    # Determine if GPU is required (based on model size and task)
    requires_gpu = (
        model_size > 500  # Models larger than 500MB likely need GPU
        or task in [
            'text-to-image',
            'image-to-image',
            'text-generation',
            'visual-question-answering'
        ]
        or any(t in tags for t in ['gpu', 'cuda'])
    )

    return ModelType(
        task=task,
        input_type=input_type,
        output_type=output_type,
        framework=framework,
        requires_gpu=requires_gpu,
        memory_requirements=memory_requirements,
        model_size=model_size,
        supported_tasks=supported_tasks
    )

def get_task_requirements(task: str) -> Dict[str, Any]:
    """Get specific requirements for a task"""
    
    base_requirements = {
        "python_version": ">=3.8",
        "cuda_version": ">=11.7",
        "system_packages": []
    }
    
    task_specific = {
        "text-to-image": {
            "min_vram": 8192,  # 8GB VRAM
            "system_packages": ["libgl1-mesa-glx", "libglib2.0-0"],
            "python_packages": ["diffusers", "invisible_watermark"]
        },
        "image-to-text": {
            "min_vram": 4096,  # 4GB VRAM
            "system_packages": ["libgl1-mesa-glx", "libglib2.0-0"],
            "python_packages": ["pillow", "torchvision"]
        },
        "text-to-speech": {
            "system_packages": ["libsndfile1", "ffmpeg"],
            "python_packages": ["soundfile", "librosa"]
        },
        "speech-recognition": {
            "system_packages": ["libsndfile1", "ffmpeg"],
            "python_packages": ["soundfile", "librosa"]
        },
        "visual-question-answering": {
            "min_vram": 8192,  # 8GB VRAM
            "system_packages": ["libgl1-mesa-glx", "libglib2.0-0"],
            "python_packages": ["pillow", "torchvision"]
        },
        "text-generation": {
            "min_vram": 4096,  # Base requirement, may be higher for large models
            "python_packages": ["accelerate", "bitsandbytes", "sentencepiece"]
        }
    }
    
    requirements = base_requirements.copy()
    if task in task_specific:
        task_reqs = task_specific[task]
        requirements["system_packages"].extend(task_reqs.get("system_packages", []))
        if "min_vram" in task_reqs:
            requirements["min_vram"] = task_reqs["min_vram"]
        if "python_packages" in task_reqs:
            requirements["python_packages"] = task_reqs["python_packages"]
            
    return requirements

def get_default_model_configuration(task: str) -> Dict[str, Any]:
    """Get default configuration for a specific task"""
    
    base_config = {
        "use_auth_token": False,
        "revision": "main",
        "trust_remote_code": True,
        "use_safetensors": True,
        "device_map": "auto",
        "torch_dtype": "float16"
    }
    
    task_configs = {
        "text-generation": {
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "num_return_sequences": 1,
            "pad_token_id": None  # Will be set at runtime
        },
        "text-to-image": {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "height": 1024,
            "width": 1024,
            "negative_prompt": "",
            "num_images_per_prompt": 1
        },
        "image-to-text": {
            "max_new_tokens": 100,
            "num_beams": 4,
            "length_penalty": 1.0,
            "do_sample": False
        },
        "visual-question-answering": {
            "max_new_tokens": 100,
            "num_beams": 4,
            "temperature": 0.7,
            "do_sample": True
        },
        "text-to-speech": {
            "vocoder": "default",
            "sample_rate": 24000,
            "voice_preset": "default",
            "audio_format": "wav"
        },
        "speech-recognition": {
            "chunk_length_s": 30,
            "return_timestamps": False,
            "language": None  # Will use model default
        },
        "translation": {
            "max_length": 512,
            "num_beams": 4,
            "length_penalty": 0.6,
            "source_lang": None,  # Will be set at runtime
            "target_lang": None  # Will be set at runtime
        },
        "summarization": {
            "max_length": 130,
            "min_length": 30,
            "do_sample": False,
            "early_stopping": True,
            "num_beams": 4
        },
        "image-classification": {
            "threshold": 0.5,
            "top_k": 5,
            "image_size": 224
        },
        "token-classification": {
            "aggregation_strategy": "simple",
            "ignore_labels": [],
            "padding": True
        }
    }
    
    config = base_config.copy()
    if task in task_configs:
        config.update(task_configs[task])
        
    return config

def get_supported_formats(task: str) -> Dict[str, List[str]]:
    """Get supported input/output formats for a task"""
    
    formats = {
        "text-generation": {
            "input": ["txt", "json"],
            "output": ["txt", "json"]
        },
        "text-to-image": {
            "input": ["txt", "json"],
            "output": ["png", "jpg"]
        },
        "image-to-text": {
            "input": ["png", "jpg", "jpeg", "bmp"],
            "output": ["txt", "json"]
        },
        "visual-question-answering": {
            "input": ["png", "jpg", "jpeg", "bmp", "txt"],
            "output": ["txt", "json"]
        },
        "text-to-speech": {
            "input": ["txt", "json"],
            "output": ["wav", "mp3"]
        },
        "speech-recognition": {
            "input": ["wav", "mp3", "flac"],
            "output": ["txt", "json"]
        },
        "translation": {
            "input": ["txt", "json"],
            "output": ["txt", "json"]
        },
        "image-classification": {
            "input": ["png", "jpg", "jpeg", "bmp"],
            "output": ["json"]
        }
    }
    
    return formats.get(task, {
        "input": ["txt"],
        "output": ["txt", "json"]
    })