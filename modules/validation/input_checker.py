"""Validation utilities for input types and formats"""

from typing import Dict, List, Tuple
import os
from pathlib import Path
import mimetypes
import magic  # python-magic library for better file type detection

class InputValidator:
    """Validator for different input types"""
    
    SUPPORTED_FORMATS = {
        'image': {
            'extensions': ['.jpg', '.jpeg', '.png', '.webp', '.bmp'],
            'mime_types': ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'],
            'min_size': 1024,  # 1KB
            'max_size': 50 * 1024 * 1024  # 50MB
        },
        'audio': {
            'extensions': ['.wav', '.mp3', '.flac', '.ogg', '.m4a'],
            'mime_types': ['audio/wav', 'audio/mpeg', 'audio/flac', 'audio/ogg', 'audio/mp4'],
            'min_size': 1024,  # 1KB
            'max_size': 500 * 1024 * 1024  # 500MB
        },
        'video': {
            'extensions': ['.mp4', '.avi', '.mov', '.webm'],
            'mime_types': ['video/mp4', 'video/x-msvideo', 'video/quicktime', 'video/webm'],
            'min_size': 1024,  # 1KB
            'max_size': 2 * 1024 * 1024 * 1024  # 2GB
        },
        'point-cloud': {
            'extensions': ['.ply', '.pcd', '.obj', '.stl'],
            'mime_types': ['application/octet-stream', 'model/obj', 'model/stl'],
            'min_size': 1024,  # 1KB
            'max_size': 1024 * 1024 * 1024  # 1GB
        },
        'text': {
            'extensions': ['.txt', '.md', '.json', '.csv'],
            'mime_types': ['text/plain', 'text/markdown', 'application/json', 'text/csv'],
            'min_size': 1,  # 1B
            'max_size': 10 * 1024 * 1024  # 10MB
        },
        'tabular': {
            'extensions': ['.csv', '.xlsx', '.parquet', '.json'],
            'mime_types': ['text/csv', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/parquet', 'application/json'],
            'min_size': 1,  # 1B
            'max_size': 1024 * 1024 * 1024  # 1GB
        }
    }

    @staticmethod
    def validate_file(file_path: str, input_type: str) -> Tuple[bool, List[str]]:
        """Validate a file for a specific input type"""
        errors = []
        
        if not os.path.exists(file_path):
            return False, ["File does not exist"]
            
        if input_type not in InputValidator.SUPPORTED_FORMATS:
            return False, [f"Unsupported input type: {input_type}"]
            
        format_info = InputValidator.SUPPORTED_FORMATS[input_type]
        
        # Check file extension
        ext = Path(file_path).suffix.lower()
        if ext not in format_info['extensions']:
            errors.append(f"Invalid file extension. Supported: {', '.join(format_info['extensions'])}")
            
        # Check file size
        size = os.path.getsize(file_path)
        if size < format_info['min_size']:
            errors.append(f"File too small. Minimum size: {format_info['min_size']} bytes")
        if size > format_info['max_size']:
            errors.append(f"File too large. Maximum size: {format_info['max_size']} bytes")
            
        # Check mime type
        try:
            mime = magic.Magic(mime=True)
            file_type = mime.from_file(file_path)
            if file_type not in format_info['mime_types']:
                errors.append(f"Invalid file type. Supported MIME types: {', '.join(format_info['mime_types'])}")
        except Exception as e:
            errors.append(f"Error checking file type: {str(e)}")
            
        return len(errors) == 0, errors

    @staticmethod
    def validate_text_input(text: str, min_length: int = 1, max_length: int = 1000000) -> Tuple[bool, List[str]]:
        """Validate text input"""
        errors = []
        
        if not text:
            errors.append("Text input is empty")
            return False, errors
            
        if len(text) < min_length:
            errors.append(f"Text too short. Minimum length: {min_length} characters")
            
        if len(text) > max_length:
            errors.append(f"Text too long. Maximum length: {max_length} characters")
            
        return len(errors) == 0, errors

    @staticmethod
    def get_input_requirements(input_type: str) -> Dict[str, any]:
        """Get requirements for a specific input type"""
        if input_type not in InputValidator.SUPPORTED_FORMATS:
            return {}
            
        format_info = InputValidator.SUPPORTED_FORMATS[input_type]
        return {
            'supported_formats': format_info['extensions'],
            'mime_types': format_info['mime_types'],
            'min_size': format_info['min_size'],
            'max_size': format_info['max_size']
        }

    @staticmethod
    def format_error_message(errors: List[str]) -> str:
        """Format validation errors into a user-friendly message"""
        if not errors:
            return "✅ Input validation passed"
            
        message = ["❌ Input validation failed:"]
        for error in errors:
            message.append(f"  • {error}")
        return "\n".join(message)
