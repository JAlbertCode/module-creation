def validate_model_url(url):
    """Validate the Hugging Face model URL format"""
    if not url:
        return False
    if not url.startswith(('http://huggingface.co/', 'https://huggingface.co/', 
                          'http://www.huggingface.co/', 'https://www.huggingface.co/')):
        return False
    return True

def extract_model_id(url):
    """Extract model ID from URL"""
    parts = url.split('huggingface.co/')
    if len(parts) != 2:
        raise ValueError("Invalid Hugging Face URL format")
    return parts[1].strip('/')

def generate_module_name(model_id):
    """Generate a valid module name from model ID"""
    # Replace invalid characters and make it lowercase
    name = model_id.split('/')[-1].lower()
    return ''.join(c if c.isalnum() or c == '-' else '-' for c in name)

def estimate_resource_requirements(model_info):
    """Estimate resource requirements based on model info"""
    # Basic estimation based on model size
    size_mb = model_info.size_in_bytes / (1024 * 1024)  # Convert to MB
    
    if size_mb < 100:
        resources = {
            'cpu': '1',
            'memory': '2Gi',
            'gpu': None
        }
    elif size_mb < 500:
        resources = {
            'cpu': '2',
            'memory': '4Gi',
            'gpu': None
        }
    else:
        resources = {
            'cpu': '4',
            'memory': '8Gi',
            'gpu': '1'
        }
    
    return resources