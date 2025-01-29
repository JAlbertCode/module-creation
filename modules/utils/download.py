"""Utilities for downloading and caching Hugging Face models"""

import os
import logging
from typing import List, Dict, Optional, Any
import shutil
import psutil
import fnmatch
from huggingface_hub import snapshot_download
from huggingface_hub.hf_api import ModelInfo

class DownloadManager:
    """Manages model downloading and caching"""
    
    def __init__(self, cache_dir: str = "./model", use_safetensors: bool = True):
        """Initialize download manager
        
        Args:
            cache_dir: Directory to cache downloaded models
            use_safetensors: Whether to prefer safetensors format
        """
        self.cache_dir = cache_dir
        self.use_safetensors = use_safetensors
        self.logger = logging.getLogger(__name__)
        
    def download_model(
        self,
        model_id: str,
        allow_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None
    ) -> str:
        """Download model files to cache directory
        
        Args:
            model_id: Hugging Face model ID
            allow_patterns: List of file patterns to download
            ignore_patterns: List of file patterns to ignore
            
        Returns:
            Path to downloaded model files
        """
        if ignore_patterns is None:
            ignore_patterns = []
            
        if not self.use_safetensors:
            # Don't ignore .bin files if not using safetensors
            ignore_patterns = [p for p in ignore_patterns if not p.endswith('.bin')]
            
        self.logger.info(f"Downloading model {model_id} to {self.cache_dir}")
        
        try:
            model_path = snapshot_download(
                repo_id=model_id,
                cache_dir=self.cache_dir,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                local_files_only=False
            )
            
            self.logger.info(f"Successfully downloaded model to {model_path}")
            return model_path
            
        except Exception as e:
            self.logger.error(f"Error downloading model: {str(e)}")
            raise
            
    def get_model_files(self, model_info: ModelInfo) -> List[str]:
        """Get list of model files to download based on type
        
        Args:
            model_info: ModelInfo from Hugging Face
            
        Returns:
            List of file patterns to download
        """
        files = []
        
        # Config files
        files.extend([
            "config.json",
            "preprocessor_config.json",
            "special_tokens_map.json", 
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
            "merges.txt"
        ])
        
        # Model weights
        if self.use_safetensors:
            files.append("*.safetensors")
        else:
            files.append("pytorch_model.bin")
            
        # Framework-specific files
        if "pytorch" in model_info.tags:
            files.extend([
                "pytorch_model.bin.index.json"
            ])
        elif "tensorflow" in model_info.tags:
            files.extend([
                "tf_model.h5",
                "variables/*"
            ])
            
        return files
        
    def get_ignore_patterns(self) -> List[str]:
        """Get patterns of files to ignore during download"""
        
        patterns = [
            "*.msgpack",
            "*.h5",
            "*.onnx", 
            "*.tflite",
            "*.opt.*",   # Optimized model variants
            "*-sharded/", # Sharded model files
            "*.ckpt",    # Checkpoint files
            "runs/*",    # Training runs
            "events.*",  # TensorBoard event files
            "test*",     # Test files/data
            "eval*"      # Evaluation files/data
        ]
        
        if self.use_safetensors:
            # Ignore PyTorch .bin files if using safetensors
            patterns.append("*.bin")
            
        return patterns
        
    def cleanup(self) -> None:
        """Clean up downloaded files"""
        if os.path.exists(self.cache_dir):
            self.logger.info(f"Cleaning up cache directory {self.cache_dir}")
            # Only remove files, keep directory structure
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to remove {file_path}: {str(e)}")

class ProgressCallback:
    """Callback for download progress tracking"""
    
    def __init__(self, total_files: int):
        """Initialize progress callback
        
        Args:
            total_files: Total number of files to download
        """
        self.total_files = total_files
        self.current_file = 0
        self.current_size = 0
        self.total_size = 0
        self.logger = logging.getLogger(__name__)
        
    def __call__(
        self,
        downloaded_size: int,
        file_size: int,
        filename: str
    ) -> None:
        """Update download progress
        
        Args:
            downloaded_size: Number of bytes downloaded
            file_size: Total file size in bytes 
            filename: Name of current file
        """
        # New file started
        if downloaded_size == 0:
            self.current_file += 1
            self.current_size = 0
            self.total_size = file_size
            self.logger.info(
                f"Downloading file {self.current_file}/{self.total_files}: {filename}"
            )
        
        # Update progress
        self.current_size = downloaded_size
        progress = (downloaded_size / file_size) * 100 if file_size > 0 else 0
        
        # Log progress at 25%, 50%, 75%, 100%
        if progress in [25, 50, 75, 100]:
            self.logger.info(
                f"Download progress for {filename}: {progress:.1f}% "
                f"({downloaded_size/(1024*1024):.1f}MB / {file_size/(1024*1024):.1f}MB)"
            )
            
def calculate_download_size(model_info: ModelInfo, use_safetensors: bool = True) -> int:
    """Calculate total size of files to download
    
    Args:
        model_info: ModelInfo from Hugging Face
        use_safetensors: Whether using safetensors format
        
    Returns:
        Total size in bytes
    """
    total_size = 0
    for sibling in model_info.siblings:
        # Skip files we won't download
        if any(pattern in sibling.rfilename for pattern in [
            ".msgpack", ".h5", ".onnx", ".tflite", ".opt.", 
            "-sharded", ".ckpt", "runs/", "events.", "test", "eval"
        ]):
            continue
            
        # Handle model weight files
        if sibling.rfilename.endswith(".bin"):
            if not use_safetensors:
                total_size += sibling.size
        elif sibling.rfilename.endswith(".safetensors"):
            if use_safetensors:
                total_size += sibling.size
        else:
            # Add size of config and other files
            total_size += sibling.size
            
    return total_size

def verify_downloaded_files(
    model_path: str,
    expected_files: List[str]
) -> bool:
    """Verify all required files were downloaded
    
    Args:
        model_path: Path to downloaded model files
        expected_files: List of expected file patterns
        
    Returns:
        True if all required files present
    """
    # Get list of downloaded files
    downloaded = []
    for root, _, files in os.walk(model_path):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), model_path)
            downloaded.append(rel_path)
    
    # Check each required file pattern
    missing = []
    for pattern in expected_files:
        if not any(fnmatch.fnmatch(f, pattern) for f in downloaded):
            missing.append(pattern)
            
    if missing:
        logging.warning(f"Missing required files: {missing}")
        return False
        
    return True

def get_model_size_requirements(total_size: int) -> Dict[str, int]:
    """Calculate disk and memory requirements for model
    
    Args:
        total_size: Total size of model files in bytes
        
    Returns:
        Dict with disk and memory requirements in bytes
    """
    # Model typically needs 2-3x its size in RAM
    memory_multiplier = 2.5
    
    # Need extra disk space for temporary files
    disk_multiplier = 1.5
    
    return {
        "disk_space": int(total_size * disk_multiplier),
        "memory": int(total_size * memory_multiplier)
    }

def check_system_requirements(requirements: Dict[str, int]) -> bool:
    """Check if system meets model requirements
    
    Args:
        requirements: Dict with disk and memory requirements
        
    Returns:
        True if system meets requirements
    """
    # Check available disk space
    _, _, free = shutil.disk_usage(".")
    if free < requirements["disk_space"]:
        logging.warning(
            f"Insufficient disk space. Need {requirements['disk_space']/(1024**3):.1f}GB, "
            f"have {free/(1024**3):.1f}GB"
        )
        return False
        
    # Check available memory
    memory = psutil.virtual_memory()
    if memory.available < requirements["memory"]:
        logging.warning(
            f"Insufficient memory. Need {requirements['memory']/(1024**3):.1f}GB, "
            f"have {memory.available/(1024**3):.1f}GB"
        )
        return False
        
    return True

def prepare_model_environment(
    model_id: str,
    cache_dir: str = "./model",
    use_safetensors: bool = True
) -> bool:
    """Prepare environment for model download and usage
    
    Args:
        model_id: Hugging Face model ID
        cache_dir: Directory to cache model files
        use_safetensors: Whether to use safetensors format
        
    Returns:
        True if environment is ready
    """
    try:
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize download manager
        downloader = DownloadManager(cache_dir, use_safetensors)
        
        # Get model info
        from huggingface_hub import model_info
        info = model_info(model_id)
        
        # Calculate size requirements
        total_size = calculate_download_size(info, use_safetensors)
        requirements = get_model_size_requirements(total_size)
        
        # Check system requirements
        if not check_system_requirements(requirements):
            return False
            
        # Get required files
        allow_patterns = downloader.get_model_files(info)
        ignore_patterns = downloader.get_ignore_patterns()
        
        # Set up progress tracking
        num_files = len([f for f in info.siblings if any(
            fnmatch.fnmatch(f.rfilename, p) for p in allow_patterns
        )])
        progress = ProgressCallback(num_files)
        
        # Download model files
        model_path = downloader.download_model(
            model_id,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns
        )
        
        # Verify downloaded files
        if not verify_downloaded_files(model_path, allow_patterns):
            return False
            
        return True
        
    except Exception as e:
        logging.error(f"Error preparing model environment: {str(e)}")
        return False