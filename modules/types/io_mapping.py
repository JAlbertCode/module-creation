"""
Comprehensive mapping of input/output types and their handling requirements
"""

from typing import Dict, Any, List, Set
from dataclasses import dataclass

@dataclass
class IORequirements:
    """Requirements for input/output handling"""
    formats: Set[str]  # Supported file formats
    processors: List[str]  # Required preprocessors
    validators: List[str]  # Required validation functions
    max_size: Dict[str, int]  # Size limits for different dimensions
    batch_support: bool  # Whether batching is supported

# Input Types
INPUT_TYPES = {
    'text': IORequirements(
        formats={'txt', 'json', 'csv', 'jsonl'},
        processors=['tokenizer', 'cleaner'],
        validators=['length', 'encoding'],
        max_size={'length': 32768},  # Max tokens
        batch_support=True
    ),
    
    'image': IORequirements(
        formats={'jpg', 'jpeg', 'png', 'bmp', 'webp'},
        processors=['image_processor', 'scaler'],
        validators=['dimensions', 'channels', 'format'],
        max_size={'width': 8192, 'height': 8192},
        batch_support=True
    ),
    
    'audio': IORequirements(
        formats={'wav', 'mp3', 'flac', 'ogg'},
        processors=['audio_processor', 'resampler'],
        validators=['duration', 'sample_rate', 'channels'],
        max_size={'duration': 3600, 'sample_rate': 48000},  # 1 hour max
        batch_support=False
    ),
    
    'video': IORequirements(
        formats={'mp4', 'avi', 'mov', 'webm'},
        processors=['video_processor', 'frame_extractor'],
        validators=['duration', 'fps', 'resolution'],
        max_size={'duration': 3600, 'width': 3840, 'height': 2160},  # 1 hour max, 4K
        batch_support=False
    ),
    
    'point_cloud': IORequirements(
        formats={'ply', 'pcd', 'xyz'},
        processors=['point_processor', 'normalizer'],
        validators=['num_points', 'dimensions'],
        max_size={'points': 1000000},  # 1M points max
        batch_support=True
    ),
    
    'graph': IORequirements(
        formats={'json', 'graphml', 'gml'},
        processors=['graph_processor'],
        validators=['num_nodes', 'num_edges'],
        max_size={'nodes': 100000, 'edges': 500000},
        batch_support=True
    ),
    
    'time_series': IORequirements(
        formats={'csv', 'parquet', 'hdf5'},
        processors=['time_processor', 'scaler'],
        validators=['length', 'frequency'],
        max_size={'length': 1000000},  # 1M timepoints max
        batch_support=True
    ),
    
    'mesh': IORequirements(
        formats={'obj', 'stl', 'fbx'},
        processors=['mesh_processor'],
        validators=['num_vertices', 'num_faces'],
        max_size={'vertices': 1000000, 'faces': 2000000},
        batch_support=True
    ),
    
    'structured': IORequirements(
        formats={'csv', 'json', 'parquet', 'arrow'},
        processors=['tabular_processor'],
        validators=['num_rows', 'schema'],
        max_size={'rows': 1000000, 'columns': 1000},
        batch_support=True
    )
}

# Output Types
OUTPUT_TYPES = {
    'classification': {
        'format': 'json',
        'schema': {
            'predictions': [{'label': str, 'confidence': float}],
            'metadata': dict
        }
    },
    
    'generation': {
        'format': 'varies',  # Based on input type
        'schema': {
            'generated_content': Any,  # Type matches input
            'parameters': dict,
            'metadata': dict
        }
    },
    
    'detection': {
        'format': 'json',
        'schema': {
            'detections': [{
                'label': str,
                'confidence': float,
                'bbox': {'x1': float, 'y1': float, 'x2': float, 'y2': float}
            }],
            'metadata': dict
        }
    },
    
    'segmentation': {
        'format': 'json+binary',
        'schema': {
            'segments': [{
                'label': str,
                'mask': bytes,  # RLE encoded
                'area': int,
                'bbox': dict
            }],
            'metadata': dict
        }
    },
    
    'transcription': {
        'format': 'json',
        'schema': {
            'text': str,
            'segments': [{
                'text': str,
                'start': float,
                'end': float,
                'confidence': float
            }],
            'metadata': dict
        }
    },
    
    'embedding': {
        'format': 'numpy',
        'schema': {
            'embeddings': 'ndarray',
            'dimensions': int,
            'metadata': dict
        }
    }
}

# Special Input Combinations
MULTIMODAL_COMBINATIONS = {
    'vision-language': {
        'inputs': ['image', 'text'],
        'processors': ['clip_processor'],
        'tasks': ['visual_qa', 'image_captioning', 'visual_entailment']
    },
    
    'audio-visual': {
        'inputs': ['video', 'audio'],
        'processors': ['av_processor'],
        'tasks': ['av_sync', 'sound_separation', 'action_recognition']
    },
    
    'document-understanding': {
        'inputs': ['image', 'text', 'layout'],
        'processors': ['layoutlm_processor'],
        'tasks': ['document_qa', 'form_understanding', 'table_extraction']
    },
    
    'sensor-fusion': {
        'inputs': ['time_series', 'structured', 'point_cloud'],
        'processors': ['fusion_processor'],
        'tasks': ['multimodal_prediction', 'anomaly_detection']
    }
}

def get_input_requirements(input_type: str) -> IORequirements:
    """Get requirements for an input type"""
    if input_type not in INPUT_TYPES:
        raise ValueError(f"Unsupported input type: {input_type}")
    return INPUT_TYPES[input_type]

def get_output_schema(output_type: str) -> Dict:
    """Get schema for an output type"""
    if output_type not in OUTPUT_TYPES:
        raise ValueError(f"Unsupported output type: {output_type}")
    return OUTPUT_TYPES[output_type]

def validate_multimodal_combination(input_types: List[str]) -> Dict:
    """Validate and get requirements for a combination of inputs"""
    for combo_name, combo_info in MULTIMODAL_COMBINATIONS.items():
        if set(input_types) == set(combo_info['inputs']):
            return combo_info
    raise ValueError(f"Unsupported input combination: {input_types}")