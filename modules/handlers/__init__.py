"""Handlers for different types of model inputs"""

from . import audio, video, multimodal

INPUT_TYPE_HANDLERS = {
    'audio': audio,
    'video': video,
    'image-text-pair': multimodal,
    'document-text-pair': multimodal
}

def get_handler(input_type):
    """Get appropriate handler for input type"""
    if input_type in INPUT_TYPE_HANDLERS:
        return INPUT_TYPE_HANDLERS[input_type]
    return None

def get_system_packages(input_type):
    """Get required system packages for input type"""
    handler = get_handler(input_type)
    if handler:
        return handler.get_system_packages()
    return []

def get_requirements(input_type):
    """Get required Python packages for input type"""
    handler = get_handler(input_type)
    if handler:
        return handler.get_requirements()
    return []

def get_inference_code(input_type, model_id, task):
    """Get inference code for input type"""
    handler = get_handler(input_type)
    if handler:
        return handler.get_inference_code(model_id, task)
    return None