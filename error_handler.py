class ModelGenerationError(Exception):
    """Base exception for model generation errors"""
    pass

def handle_error(error):
    """Convert exceptions to user-friendly messages"""
    error_messages = {
        'InvalidModelId': 'Invalid model ID. Please check the URL and try again.',
        'ModelNotFound': 'Model not found on Hugging Face. Please verify the URL.',
        'InvalidUrl': 'Please enter a valid Hugging Face model URL.',
        'NetworkError': 'Network error. Please check your connection.',
        'GenerationError': 'Error generating module files. Please try again.',
    }
    
    error_type = error.__class__.__name__
    return error_messages.get(error_type, str(error))

def validate_model_url(url: str) -> bool:
    """Validate Hugging Face model URL format"""
    if not url:
        raise ModelGenerationError('InvalidUrl')
        
    if not url.startswith('https://huggingface.co/'):
        raise ModelGenerationError('InvalidUrl')
        
    # Extract model ID
    try:
        model_id = url.split('huggingface.co/')[-1].strip('/')
        if not model_id or '/' not in model_id:
            raise ModelGenerationError('InvalidModelId')
    except:
        raise ModelGenerationError('InvalidModelId')
        
    return True