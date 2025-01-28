"""Handlers for different types of model inputs"""

from . import audio, video, multimodal, structured, point_cloud, time_series

INPUT_TYPE_HANDLERS = {
    # Basic types
    'text': None,  # Uses default handler
    'image': None, # Uses default handler
    
    # Audio/Video
    'audio': audio,
    'video': video,
    
    # Multimodal
    'image-text-pair': multimodal,
    'document-text-pair': multimodal,
    
    # Structured data
    'tabular': structured,
    'table': structured,
    'json': structured,
    'csv': structured,
    
    # 3D data
    'point-cloud': point_cloud,
    'mesh': point_cloud,
    '3d': point_cloud,
    
    # Time series
    'time-series': time_series,
    'temporal': time_series,
    'sequence': time_series
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