"""Comprehensive mapping of all Hugging Face model types and tasks"""

# Full list of task types and their configurations
TASK_MAP = {
    # Text Generation Tasks
    'text-generation': {
        'input_types': ['text'],
        'output_types': ['text'],
        'parameters': {
            'max_length': {'type': 'int', 'default': 128},
            'temperature': {'type': 'float', 'default': 1.0},
            'top_p': {'type': 'float', 'default': 0.9},
            'repetition_penalty': {'type': 'float', 'default': 1.0},
            'do_sample': {'type': 'bool', 'default': True}
        }
    },
    'text2text-generation': {
        'input_types': ['text'],
        'output_types': ['text'],
        'parameters': {
            'max_length': {'type': 'int', 'default': 128},
            'min_length': {'type': 'int', 'default': 10},
            'length_penalty': {'type': 'float', 'default': 1.0},
            'num_beams': {'type': 'int', 'default': 4}
        }
    },
    'summarization': {
        'input_types': ['text'],
        'output_types': ['text'],
        'parameters': {
            'max_length': {'type': 'int', 'default': 130},
            'min_length': {'type': 'int', 'default': 30},
            'length_penalty': {'type': 'float', 'default': 2.0},
            'num_beams': {'type': 'int', 'default': 4}
        }
    },
    'translation': {
        'input_types': ['text'],
        'output_types': ['text'],
        'parameters': {
            'source_lang': {'type': 'string'},
            'target_lang': {'type': 'string'},
            'max_length': {'type': 'int', 'default': 128}
        }
    },
    
    # Text Classification Tasks
    'text-classification': {
        'input_types': ['text'],
        'output_types': ['label', 'scores'],
        'parameters': {
            'return_all_scores': {'type': 'bool', 'default': True},
            'function_to_apply': {'type': 'string', 'default': 'sigmoid'}
        }
    },
    'token-classification': {
        'input_types': ['text'],
        'output_types': ['tokens', 'labels'],
        'parameters': {
            'aggregation_strategy': {'type': 'string', 'default': 'simple'}
        }
    },
    'question-answering': {
        'input_types': ['question', 'context'],
        'output_types': ['answer', 'score', 'start', 'end'],
        'parameters': {
            'max_answer_len': {'type': 'int', 'default': 30},
            'max_question_len': {'type': 'int', 'default': 64},
            'max_seq_len': {'type': 'int', 'default': 384},
            'doc_stride': {'type': 'int', 'default': 128}
        }
    },
    
    # Image Tasks
    'image-classification': {
        'input_types': ['image'],
        'output_types': ['label', 'scores'],
        'parameters': {
            'threshold': {'type': 'float', 'default': 0.1}
        }
    },
    'image-segmentation': {
        'input_types': ['image'],
        'output_types': ['mask', 'labels', 'scores'],
        'parameters': {
            'threshold': {'type': 'float', 'default': 0.5},
            'mask_threshold': {'type': 'float', 'default': 0.5}
        }
    },
    'object-detection': {
        'input_types': ['image'],
        'output_types': ['boxes', 'labels', 'scores'],
        'parameters': {
            'threshold': {'type': 'float', 'default': 0.9},
            'overlap_threshold': {'type': 'float', 'default': 0.3}
        }
    },
    'image-to-text': {
        'input_types': ['image'],
        'output_types': ['text'],
        'parameters': {
            'max_new_tokens': {'type': 'int', 'default': 128},
            'num_beams': {'type': 'int', 'default': 4}
        }
    },
    'text-to-image': {
        'input_types': ['text'],
        'output_types': ['image'],
        'parameters': {
            'num_inference_steps': {'type': 'int', 'default': 50},
            'guidance_scale': {'type': 'float', 'default': 7.5},
            'negative_prompt': {'type': 'string', 'default': ''},
            'height': {'type': 'int', 'default': 512},
            'width': {'type': 'int', 'default': 512}
        }
    },
    'image-to-image': {
        'input_types': ['image', 'prompt'],
        'output_types': ['image'],
        'parameters': {
            'strength': {'type': 'float', 'default': 0.8},
            'guidance_scale': {'type': 'float', 'default': 7.5}
        }
    },
    
    # Audio Tasks
    'automatic-speech-recognition': {
        'input_types': ['audio'],
        'output_types': ['text'],
        'parameters': {
            'chunk_length_s': {'type': 'float', 'default': 30},
            'stride_length_s': {'type': 'float', 'default': 5}
        }
    },
    'audio-classification': {
        'input_types': ['audio'],
        'output_types': ['label', 'scores'],
        'parameters': {
            'threshold': {'type': 'float', 'default': 0.5}
        }
    },
    'text-to-speech': {
        'input_types': ['text'],
        'output_types': ['audio'],
        'parameters': {
            'voice_preset': {'type': 'string'},
            'speaking_rate': {'type': 'float', 'default': 1.0},
            'pitch': {'type': 'float', 'default': 0.0}
        }
    },
    
    # Video Tasks
    'video-classification': {
        'input_types': ['video'],
        'output_types': ['label', 'scores'],
        'parameters': {
            'frames_per_second': {'type': 'int', 'default': 1},
            'chunk_size': {'type': 'int', 'default': 8}
        }
    },
    'text-to-video': {
        'input_types': ['text'],
        'output_types': ['video'],
        'parameters': {
            'num_inference_steps': {'type': 'int', 'default': 50},
            'num_frames': {'type': 'int', 'default': 16},
            'height': {'type': 'int', 'default': 256},
            'width': {'type': 'int', 'default': 256}
        }
    },
    
    # Multi-modal Tasks
    'visual-question-answering': {
        'input_types': ['image', 'text'],
        'output_types': ['text', 'score'],
        'parameters': {
            'max_length': {'type': 'int', 'default': 64}
        }
    },
    'document-question-answering': {
        'input_types': ['image', 'text'],
        'output_types': ['text', 'score'],
        'parameters': {
            'max_length': {'type': 'int', 'default': 64}
        }
    },
    
    # Specialized Tasks
    'feature-extraction': {
        'input_types': ['text', 'image', 'audio'],
        'output_types': ['embeddings'],
        'parameters': {
            'pooling': {'type': 'string', 'default': 'mean'},
            'normalize': {'type': 'bool', 'default': False}
        }
    },
    'sentence-similarity': {
        'input_types': ['text_pair'],
        'output_types': ['similarity_score'],
        'parameters': {
            'scoring_function': {'type': 'string', 'default': 'cosine'}
        }
    },
    'zero-shot-classification': {
        'input_types': ['text', 'candidate_labels'],
        'output_types': ['labels', 'scores'],
        'parameters': {
            'hypothesis_template': {'type': 'string', 'default': 'This example is {}.'}
        }
    },

    # Structured Data Tasks
    'tabular-classification': {
        'input_types': ['table'],
        'output_types': ['label', 'scores'],
        'parameters': {
            'threshold': {'type': 'float', 'default': 0.5}
        }
    },
    'tabular-regression': {
        'input_types': ['table'],
        'output_types': ['value'],
        'parameters': {}
    },
    
    # Time Series Tasks
    'time-series-forecasting': {
        'input_types': ['time_series'],
        'output_types': ['forecast'],
        'parameters': {
            'prediction_length': {'type': 'int'},
            'quantiles': {'type': 'list', 'default': [0.1, 0.5, 0.9]}
        }
    },
    
    # 3D Tasks
    'point-cloud-segmentation': {
        'input_types': ['point_cloud'],
        'output_types': ['labels', 'scores'],
        'parameters': {
            'threshold': {'type': 'float', 'default': 0.5}
        }
    },
    'mesh-reconstruction': {
        'input_types': ['point_cloud', 'image'],
        'output_types': ['mesh'],
        'parameters': {
            'resolution': {'type': 'int', 'default': 256}
        }
    }
}

# Input type specifications
INPUT_SPECS = {
    'text': {
        'formats': ['.txt', '.json'],
        'max_length': 2048,
        'preprocessing': ['tokenization', 'padding']
    },
    'image': {
        'formats': ['.jpg', '.jpeg', '.png', '.bmp'],
        'max_size': 1024,  # pixels
        'preprocessing': ['resize', 'normalize']
    },
    'audio': {
        'formats': ['.wav', '.mp3', '.flac'],
        'max_length': 30,  # seconds
        'preprocessing': ['resample', 'normalize']
    },
    'video': {
        'formats': ['.mp4', '.avi', '.mov'],
        'max_length': 60,  # seconds
        'preprocessing': ['frame_extraction', 'resize']
    },
    'point_cloud': {
        'formats': ['.ply', '.pcd', '.obj'],
        'max_points': 1000000,
        'preprocessing': ['sampling', 'normalize']
    },
    'table': {
        'formats': ['.csv', '.json', '.parquet'],
        'max_rows': 1000000,
        'preprocessing': ['type_conversion', 'scaling']
    },
    'time_series': {
        'formats': ['.csv', '.json'],
        'max_length': 10000,
        'preprocessing': ['resampling', 'scaling']
    }
}

def get_task_info(task_name):
    """Get information about a specific task"""
    return TASK_MAP.get(task_name)

def get_input_spec(input_type):
    """Get specifications for an input type"""
    return INPUT_SPECS.get(input_type)

def get_required_preprocessing(task_name):
    """Get required preprocessing steps for a task"""
    task_info = get_task_info(task_name)
    if not task_info:
        return []
    
    preprocessing_steps = []
    for input_type in task_info['input_types']:
        input_spec = get_input_spec(input_type)
        if input_spec and 'preprocessing' in input_spec:
            preprocessing_steps.extend(input_spec['preprocessing'])
    
    return list(set(preprocessing_steps))  # Remove duplicates