"""Model type detection and mapping"""

# Comprehensive mapping of all supported model types and inputs
PIPELINE_MAPPING = {
    # Text Tasks
    'text-classification': {'task': 'text-classification', 'input': 'text'},
    'text-generation': {'task': 'text-generation', 'input': 'text'},
    'question-answering': {'task': 'question-answering', 'input': 'text'},
    'summarization': {'task': 'summarization', 'input': 'text'},
    'translation': {'task': 'translation', 'input': 'text'},
    'text2text-generation': {'task': 'text2text-generation', 'input': 'text'},
    'sentence-similarity': {'task': 'sentence-similarity', 'input': 'text-pair'},
    'token-classification': {'task': 'token-classification', 'input': 'text'},
    'fill-mask': {'task': 'fill-mask', 'input': 'text'},
    'zero-shot-classification': {'task': 'zero-shot-classification', 'input': 'text'},
    'text-to-sql': {'task': 'text-to-sql', 'input': 'text'},
    'table-question-answering': {'task': 'table-question-answering', 'input': 'tabular'},
    'named-entity-recognition': {'task': 'ner', 'input': 'text'},
    
    # Image Tasks
    'image-classification': {'task': 'image-classification', 'input': 'image'},
    'object-detection': {'task': 'object-detection', 'input': 'image'},
    'image-segmentation': {'task': 'image-segmentation', 'input': 'image'},
    'image-to-text': {'task': 'image-to-text', 'input': 'image'},
    'text-to-image': {'task': 'text-to-image', 'input': 'text'},
    'image-to-image': {'task': 'image-to-image', 'input': 'image'},
    'depth-estimation': {'task': 'depth-estimation', 'input': 'image'},
    'image-to-3d': {'task': 'image-to-3d', 'input': 'image'},
    'super-resolution': {'task': 'super-resolution', 'input': 'image'},
    'inpainting': {'task': 'inpainting', 'input': 'image'},
    'background-removal': {'task': 'background-removal', 'input': 'image'},
    'pose-detection': {'task': 'pose-detection', 'input': 'image'},
    'face-detection': {'task': 'face-detection', 'input': 'image'},
    'optical-character-recognition': {'task': 'ocr', 'input': 'image'},
    
    # Audio Tasks
    'automatic-speech-recognition': {'task': 'speech-recognition', 'input': 'audio'},
    'audio-classification': {'task': 'audio-classification', 'input': 'audio'},
    'text-to-speech': {'task': 'text-to-speech', 'input': 'text'},
    'sound-event-detection': {'task': 'sound-event-detection', 'input': 'audio'},
    'music-generation': {'task': 'music-generation', 'input': 'audio'},
    'voice-conversion': {'task': 'voice-conversion', 'input': 'audio'},
    'speech-enhancement': {'task': 'speech-enhancement', 'input': 'audio'},
    'audio-super-resolution': {'task': 'audio-super-resolution', 'input': 'audio'},
    'audio-separation': {'task': 'audio-separation', 'input': 'audio'},
    'voice-activity-detection': {'task': 'voice-activity-detection', 'input': 'audio'},
    
    # Video Tasks
    'video-classification': {'task': 'video-classification', 'input': 'video'},
    'text-to-video': {'task': 'text-to-video', 'input': 'text'},
    'video-inpainting': {'task': 'video-inpainting', 'input': 'video'},
    'video-motion-transfer': {'task': 'video-motion-transfer', 'input': 'video'},
    'video-super-resolution': {'task': 'video-super-resolution', 'input': 'video'},
    'video-stabilization': {'task': 'video-stabilization', 'input': 'video'},
    'video-frame-interpolation': {'task': 'video-frame-interpolation', 'input': 'video'},
    'video-object-tracking': {'task': 'video-object-tracking', 'input': 'video'},
    'action-recognition': {'task': 'action-recognition', 'input': 'video'},
    'scene-detection': {'task': 'scene-detection', 'input': 'video'},
    
    # 3D/Point Cloud Tasks
    'point-cloud-classification': {'task': 'point-cloud-classification', 'input': 'point-cloud'},
    'point-cloud-segmentation': {'task': 'point-cloud-segmentation', 'input': 'point-cloud'},
    '3d-object-detection': {'task': '3d-object-detection', 'input': 'point-cloud'},
    '3d-scene-understanding': {'task': '3d-scene-understanding', 'input': 'point-cloud'},
    '3d-mesh-reconstruction': {'task': '3d-mesh-reconstruction', 'input': 'point-cloud'},
    
    # Time Series Tasks
    'time-series-forecasting': {'task': 'forecasting', 'input': 'time-series'},
    'time-series-classification': {'task': 'time-series-classification', 'input': 'time-series'},
    'anomaly-detection': {'task': 'anomaly-detection', 'input': 'time-series'},
    'event-detection': {'task': 'event-detection', 'input': 'time-series'},
    'trend-prediction': {'task': 'trend-prediction', 'input': 'time-series'},
    
    # Structured Data Tasks
    'tabular-classification': {'task': 'tabular-classification', 'input': 'tabular'},
    'tabular-regression': {'task': 'tabular-regression', 'input': 'tabular'},
    'graph-classification': {'task': 'graph-classification', 'input': 'graph'},
    'node-classification': {'task': 'node-classification', 'input': 'graph'},
    'link-prediction': {'task': 'link-prediction', 'input': 'graph'},
    'graph-generation': {'task': 'graph-generation', 'input': 'graph'},
    'molecular-property-prediction': {'task': 'molecular-property-prediction', 'input': 'graph'},
    'protein-structure-prediction': {'task': 'protein-structure-prediction', 'input': 'sequence'},
    
    # Multimodal Tasks
    'visual-question-answering': {'task': 'vqa', 'input': 'image-text-pair'},
    'document-question-answering': {'task': 'document-qa', 'input': 'document-text-pair'},
    'audio-to-video-sync': {'task': 'audio-video-sync', 'input': 'audio-video-pair'},
    'text-audio-video-alignment': {'task': 'multimodal-alignment', 'input': 'multimodal'},
    'speech-to-sign': {'task': 'speech-to-sign', 'input': 'audio'},
    'sign-language-recognition': {'task': 'sign-language-recognition', 'input': 'video'},
    'cross-modal-retrieval': {'task': 'cross-modal-retrieval', 'input': 'multimodal'},
    'multi-document-qa': {'task': 'multi-document-qa', 'input': 'document-text-pair'},
    
    # Reinforcement Learning Tasks
    'policy-model': {'task': 'policy-modeling', 'input': 'state'},
    'value-function': {'task': 'value-function', 'input': 'state'},
    'world-model': {'task': 'world-modeling', 'input': 'state-action-pair'}
}

# Keywords that help identify model types from model card descriptions
TASK_KEYWORDS = {
    'text': ['nlp', 'language', 'text', 'sentence', 'word', 'token', 'document'],
    'image': ['vision', 'image', 'photo', 'picture', 'visual', 'pixel'],
    'audio': ['speech', 'sound', 'audio', 'voice', 'acoustic'],
    'video': ['video', 'motion', 'temporal', 'frame'],
    'point-cloud': ['3d', 'point cloud', 'mesh', 'pointcloud', 'depth'],
    'time-series': ['time series', 'temporal', 'sequence', 'sequential'],
    'tabular': ['table', 'tabular', 'structured', 'database'],
    'graph': ['graph', 'network', 'node', 'edge', 'molecular'],
    'multimodal': ['multimodal', 'multi-modal', 'cross-modal']
}

def detect_model_type_from_card(model_info):
    """Detect model type from model card description"""
    if not model_info.cardData:
        return None
        
    card_text = str(model_info.cardData).lower()
    
    # Check for task-specific keywords
    matched_types = []
    for input_type, keywords in TASK_KEYWORDS.items():
        if any(keyword in card_text for keyword in keywords):
            matched_types.append(input_type)
            
    if len(matched_types) == 1:
        # If we found exactly one match, use it
        matched_type = matched_types[0]
        # Return a default task for this input type
        for task, info in PIPELINE_MAPPING.items():
            if info['input'] == matched_type:
                return info
    elif len(matched_types) > 1:
        # If multiple matches, prefer the most specific one
        priority = ['multimodal', 'point-cloud', 'time-series', 'video', 'audio', 'image', 'text']
        for type_priority in priority:
            if type_priority in matched_types:
                for task, info in PIPELINE_MAPPING.items():
                    if info['input'] == type_priority:
                        return info
                        
    return None

def detect_model_type(model_info):
    """Detect the model type from model info and tags"""
    # First check explicit tags
    for tag in model_info.tags:
        if tag in PIPELINE_MAPPING:
            return PIPELINE_MAPPING[tag]
    
    # If no matching tag, try to infer from model card
    card_type = detect_model_type_from_card(model_info)
    if card_type:
        return card_type
    
    # Default to text classification if we can't determine the type
    return {'task': 'text-classification', 'input': 'text'}