"""
Input/Output utilities for handling different data types
"""

import os
import json
import base64
from typing import Any, Dict, Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image

class DataHandler:
    """Handler for different data types and formats"""
    
    @staticmethod
    def _detect_file_type(file_path: str) -> str:
        """Detect file type from extension"""
        ext = file_path.lower().split('.')[-1]
        
        image_exts = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
        audio_exts = {'wav', 'mp3', 'ogg', 'flac'}
        video_exts = {'mp4', 'avi', 'mov', 'webm', 'mkv'}
        text_exts = {'txt', 'md', 'json', 'yml', 'yaml'}
        
        if ext in image_exts:
            return 'image'
        elif ext in audio_exts:
            return 'audio'
        elif ext in video_exts:
            return 'video'
        elif ext in text_exts:
            return 'text'
        else:
            return 'binary'

    @staticmethod
    def _detect_data_type(data: Any) -> str:
        """Detect type of data"""
        if isinstance(data, Image.Image):
            return 'image'
        elif isinstance(data, np.ndarray):
            if data.ndim == 3 and data.shape[-1] in {3, 4}:
                return 'image'
            elif data.ndim >= 3:
                return 'video'
            else:
                return 'array'
        elif isinstance(data, dict):
            return 'json'
        elif isinstance(data, (str, bytes)):
            return 'text'
        else:
            return 'unknown'

    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, Any]:
        """Get detailed information about a file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        info = {
            'path': file_path,
            'size': os.path.getsize(file_path),
            'modified': os.path.getmtime(file_path),
            'type': DataHandler._detect_file_type(file_path)
        }
        
        try:
            if info['type'] == 'image':
                image = Image.open(file_path)
                info.update({
                    'width': image.width,
                    'height': image.height,
                    'mode': image.mode,
                    'format': image.format
                })
            elif info['type'] == 'audio':
                import librosa
                y, sr = librosa.load(file_path)
                info.update({
                    'duration': librosa.get_duration(y=y, sr=sr),
                    'sample_rate': sr,
                    'samples': len(y)
                })
            elif info['type'] == 'video':
                import cv2
                cap = cv2.VideoCapture(file_path)
                info.update({
                    'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    'fps': cap.get(cv2.CAP_PROP_FPS),
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
                })
                cap.release()
        except Exception as e:
            info['error'] = str(e)
        
        return info

class BatchProcessor:
    """Process multiple files in batches"""
    
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.data_handler = DataHandler()

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        process_func: callable,
        file_pattern: str = '*'
    ) -> Dict[str, Any]:
        """
        Process all files in a directory
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            process_func: Function to process each batch
            file_pattern: Pattern to match files
            
        Returns:
            Dictionary with processing results
        """
        input_path = Path(input_dir)
        files = list(input_path.glob(file_pattern))
        results = {
            'processed': 0,
            'failed': 0,
            'files': []
        }
        
        # Process in batches
        for i in range(0, len(files), self.batch_size):
            batch_files = files[i:i + self.batch_size]
            batch_data = []
            
            # Load batch
            for file_path in batch_files:
                try:
                    data = self.data_handler.load_file(str(file_path))
                    batch_data.append((file_path.name, data))
                except Exception as e:
                    results['failed'] += 1
                    results['files'].append({
                        'file': file_path.name,
                        'status': 'failed',
                        'error': str(e)
                    })
                    continue
            
            if not batch_data:
                continue
            
            # Process batch
            try:
                processed = process_func([d[1] for d in batch_data])
                
                # Save results
                for (filename, _), result in zip(batch_data, processed):
                    output_path = os.path.join(
                        output_dir,
                        os.path.splitext(filename)[0] + '.json'
                    )
                    self.data_handler.save_json(result, output_path)
                    
                    results['processed'] += 1
                    results['files'].append({
                        'file': filename,
                        'status': 'success',
                        'output': output_path
                    })
                    
            except Exception as e:
                # Mark whole batch as failed
                results['failed'] += len(batch_data)
                for filename, _ in batch_data:
                    results['files'].append({
                        'file': filename,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        return results

class CacheManager:
    """Manage cached files and data"""
    
    def __init__(self, cache_dir: str = '.cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cached_file(self, key: str) -> Optional[str]:
        """Get path to cached file if it exists"""
        cache_path = self.cache_dir / key
        return str(cache_path) if cache_path.exists() else None
        
    def cache_file(self, key: str, file_path: str) -> str:
        """Cache a file"""
        cache_path = self.cache_dir / key
        os.makedirs(cache_path.parent, exist_ok=True)
        
        if os.path.exists(file_path):
            import shutil
            shutil.copy2(file_path, cache_path)
        return str(cache_path)
        
    def cache_data(self, key: str, data: Any) -> str:
        """Cache arbitrary data"""
        cache_path = self.cache_dir / key
        os.makedirs(cache_path.parent, exist_ok=True)
        
        self.data_handler.save_file(data, str(cache_path))
        return str(cache_path)
        
    def clear_cache(self, older_than: Optional[float] = None):
        """Clear cached files"""
        for path in self.cache_dir.glob('**/*'):
            if path.is_file():
                if older_than is None or \
                   (time.time() - path.stat().st_mtime) > older_than:
                    path.unlink()
                    
    def get_cache_size(self) -> int:
        """Get total size of cached files in bytes"""
        return sum(f.stat().st_size for f in self.cache_dir.glob('**/*') if f.is_file())