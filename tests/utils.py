"""
Test utilities for Hugging Face to Lilypad converter
"""

import os
import tempfile
import shutil
from typing import Dict, Any, Optional
import torch
from PIL import Image
import numpy as np

class TestResources:
    """Manage test resources and cleanup"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.resources = {}
    
    def cleanup(self):
        """Remove all temporary resources"""
        shutil.rmtree(self.temp_dir)
    
    def create_test_image(self, size: tuple = (64, 64)) -> str:
        """Create a test image file"""
        image = Image.new('RGB', size, color='red')
        path = os.path.join(self.temp_dir, 'test_image.jpg')
        image.save(path)
        self.resources['image'] = path
        return path
    
    def create_test_audio(self, duration: float = 1.0, sample_rate: int = 16000) -> str:
        """Create a test audio file"""
        import soundfile as sf
        samples = np.random.randn(int(duration * sample_rate))
        path = os.path.join(self.temp_dir, 'test_audio.wav')
        sf.write(path, samples, sample_rate)
        self.resources['audio'] = path
        return path
    
    def create_test_video(self, num_frames: int = 10, size: tuple = (64, 64)) -> str:
        """Create a test video file"""
        import cv2
        path = os.path.join(self.temp_dir, 'test_video.mp4')
        
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            30,
            size
        )
        
        for _ in range(num_frames):
            frame = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            writer.write(frame)
        
        writer.release()
        self.resources['video'] = path
        return path

class MockModel:
    """Mock model for testing"""
    
    def __init__(self, task: str):
        self.task = task
        self.config = self._get_config()
    
    def _get_config(self) -> Dict:
        """Get mock configuration based on task"""
        if 'classification' in self.task:
            return {
                'id2label': {0: 'class_0', 1: 'class_1'},
                'label2id': {'class_0': 0, 'class_1': 1}
            }
        elif 'detection' in self.task:
            return {
                'id2label': {0: 'object_0', 1: 'object_1'},
                'label2id': {'object_0': 0, 'object_1': 1}
            }
        return {}
    
    def __call__(self, **kwargs) -> Any:
        """Mock forward pass"""
        batch_size = kwargs.get('input_ids', torch.ones(1, 1)).shape[0]
        
        if 'classification' in self.task:
            logits = torch.randn(batch_size, len(self.config['id2label']))
            return type('Outputs', (), {'logits': logits})()
        
        elif 'detection' in self.task:
            return type('Outputs', (), {
                'boxes': torch.randn(batch_size, 2, 4),
                'scores': torch.rand(batch_size, 2),
                'labels': torch.randint(0, 2, (batch_size, 2))
            })()
        
        return type('Outputs', (), {'logits': torch.randn(batch_size, 10)})()

class MockProcessor:
    """Mock processor for testing"""
    
    def __init__(self, task: str):
        self.task = task
        self.sampling_rate = 16000
    
    def __call__(self, *args, **kwargs) -> Dict:
        """Mock processing"""
        if 'images' in kwargs:
            # Image processing
            return {
                'pixel_values': torch.randn(1, 3, 224, 224),
                'attention_mask': torch.ones(1, 224, 224)
            }
        elif 'audio' in kwargs or 'raw' in kwargs:
            # Audio processing
            return {
                'input_values': torch.randn(1, 16000),
                'attention_mask': torch.ones(1, 100)
            }
        else:
            # Text processing
            return {
                'input_ids': torch.randint(0, 1000, (1, 10)),
                'attention_mask': torch.ones(1, 10)
            }

def assert_valid_output(output: Dict[str, Any], expected_keys: Optional[List[str]] = None):
    """Validate handler output format"""
    assert isinstance(output, dict), "Output should be a dictionary"
    assert "error" not in output, f"Output contains error: {output.get('error')}"
    
    if expected_keys:
        missing_keys = set(expected_keys) - set(output.keys())
        assert not missing_keys, f"Output missing expected keys: {missing_keys}"
        
    # Common metadata checks
    if "metadata" in output:
        assert isinstance(output["metadata"], dict)
        assert "model" in output["metadata"]
        assert "task" in output["metadata"]