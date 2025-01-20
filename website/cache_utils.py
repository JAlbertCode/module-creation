import redis
import json
from functools import wraps

redis_client = redis.Redis(host='redis', port=6379, db=0)
CACHE_EXPIRATION = 3600  # 1 hour

def cache_model_info(func):
    """Decorator to cache model information"""
    @wraps(func)
    def wrapper(model_url):
        cache_key = f"model_info:{model_url}"
        
        # Try to get from cache
        cached_data = redis_client.get(cache_key)
        if cached_data:
            return json.loads(cached_data)
        
        # Get fresh data
        result = func(model_url)
        
        # Cache the result
        redis_client.setex(
            cache_key,
            CACHE_EXPIRATION,
            json.dumps(result)
        )
        
        return result
    return wrapper

def cache_generated_files(func):
    """Decorator to cache generated files"""
    @wraps(func)
    def wrapper(model_url):
        cache_key = f"generated_files:{model_url}"
        
        # Try to get from cache
        cached_data = redis_client.get(cache_key)
        if cached_data:
            return json.loads(cached_data)
        
        # Generate fresh files
        result = func(model_url)
        
        # Cache the result
        redis_client.setex(
            cache_key,
            CACHE_EXPIRATION,
            json.dumps(result)
        )
        
        return result
    return wrapper

def clear_model_cache(model_url):
    """Clear cached data for a specific model"""
    keys_to_delete = [
        f"model_info:{model_url}",
        f"generated_files:{model_url}"
    ]
    
    for key in keys_to_delete:
        redis_client.delete(key)

def get_cached_models():
    """Get list of currently cached models"""
    cached_models = []
    for key in redis_client.keys("model_info:*"):
        model_url = key.decode('utf-8').split(":", 1)[1]
        cached_models.append(model_url)
    return cached_models