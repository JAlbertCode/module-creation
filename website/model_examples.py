from typing import Dict, Any
import base64
import json

class ModelExampleGenerator:
    """Generates example inputs and usage patterns for different model types"""
    
    def __init__(self, model_info, pipeline_type):
        self.model_info = model_info
        self.pipeline_type = pipeline_type
    
    def get_examples(self) -> Dict[str, Any]:
        """Get complete example set for the model type"""
        examples = {
            'input_format': self._get_input_format(),
            'example_input': self._get_example_input(),
            'example_output': self._get_example_output(),
            'curl_example': self._get_curl_example(),
            'python_example': self._get_python_example(),
            'lilypad_example': self._get_lilypad_example()
        }
        
        # Add any model-specific parameters
        if specific_params := self._get_specific_params():
            examples['model_parameters'] = specific_params
            
        return examples
    
    def _get_input_format(self) -> Dict[str, str]:
        """Get input format description"""
        formats = {
            'text-classification': {
                'type': 'text',
                'format': 'plain text file or string',
                'description': 'Text content to be classified'
            },
            'image-classification': {
                'type': 'image',
                'format': 'JPG, PNG, or WebP',
                'description': 'Image file to be classified'
            },
            'object-detection': {
                'type': 'image',
                'format': 'JPG, PNG, or WebP',
                'description': 'Image file for object detection'
            },
            'text-generation': {
                'type': 'text',
                'format': 'plain text file or string',
                'description': 'Prompt text for generation'
            },
            'question-answering': {
                'type': 'json',
                'format': 'JSON with question and context',
                'description': 'Question and context for answering'
            }
        }
        return formats.get(self.pipeline_type, {
            'type': 'text',
            'format': 'plain text or file',
            'description': 'Input data for the model'
        })
    
    def _get_example_input(self) -> Dict[str, Any]:
        """Get example input for the model type"""
        examples = {
            'text-classification': {
                'text': 'This movie was fantastic! The acting and direction were superb.',
                'file_content': 'This movie was fantastic! The acting and direction were superb.'
            },
            'image-classification': {
                'description': 'An image file (e.g., cat.jpg)',
                'file_type': 'image/jpeg',
                'example_command': 'cp your_image.jpg input.jpg'
            },
            'object-detection': {
                'description': 'An image file (e.g., street_scene.jpg)',
                'file_type': 'image/jpeg',
                'example_command': 'cp your_image.jpg input.jpg'
            },
            'text-generation': {
                'text': 'Once upon a time in a galaxy',
                'file_content': 'Once upon a time in a galaxy'
            },
            'question-answering': {
                'json': {
                    'question': 'What is the capital of France?',
                    'context': 'Paris is the capital and largest city of France.'
                },
                'file_content': '{"question": "What is the capital of France?", "context": "Paris is the capital and largest city of France."}'
            }
        }
        return examples.get(self.pipeline_type, {
            'text': 'Example input text',
            'file_content': 'Example input text'
        })
    
    def _get_example_output(self) -> Dict[str, Any]:
        """Get example output format"""
        outputs = {
            'text-classification': {
                'label': 'POSITIVE',
                'score': 0.9853,
                'status': 'success'
            },
            'image-classification': {
                'predictions': [
                    {'label': 'cat', 'score': 0.985},
                    {'label': 'dog', 'score': 0.012}
                ],
                'status': 'success'
            },
            'object-detection': {
                'objects': [
                    {
                        'label': 'car',
                        'score': 0.982,
                        'box': {'x1': 100, 'y1': 200, 'x2': 300, 'y2': 400}
                    }
                ],
                'status': 'success'
            },
            'text-generation': {
                'generated_text': 'Once upon a time in a galaxy far, far away...',
                'status': 'success'
            },
            'question-answering': {
                'answer': 'Paris',
                'score': 0.989,
                'start': 0,
                'end': 5,
                'status': 'success'
            }
        }
        return outputs.get(self.pipeline_type, {
            'result': 'Model-specific output',
            'status': 'success'
        })
    
    def _get_curl_example(self) -> str:
        """Get curl command example"""
        return f'''curl -X POST \\
    -H "Content-Type: application/json" \\
    -d '{json.dumps(self._get_example_input())}' \\
    http://localhost:8080/predict'''
    
    def _get_python_example(self) -> str:
        """Get Python code example"""
        return f'''import requests
import json

url = "http://localhost:8080/predict"
data = {json.dumps(self._get_example_input(), indent=4)}

response = requests.post(url, json=data)
result = response.json()
print(json.dumps(result, indent=2))'''
    
    def _get_lilypad_example(self) -> str:
        """Get Lilypad command example"""
        return f'''lilypad run {self.model_info.id.split('/')[-1]} \\
    --input "INPUT_PATH=/path/to/input/{self._get_input_filename()}"'''
    
    def _get_specific_params(self) -> Dict[str, Any]:
        """Get model-specific parameters if any"""
        params = {
            'text-generation': {
                'max_length': {'type': 'integer', 'default': 100, 'description': 'Maximum length of generated text'},
                'temperature': {'type': 'float', 'default': 0.7, 'description': 'Sampling temperature'},
                'top_p': {'type': 'float', 'default': 0.9, 'description': 'Nucleus sampling parameter'}
            },
            'image-classification': {
                'top_k': {'type': 'integer', 'default': 5, 'description': 'Number of top predictions to return'}
            }
        }
        return params.get(self.pipeline_type, {})
    
    def _get_input_filename(self) -> str:
        """Get default input filename based on pipeline type"""
        extensions = {
            'text-classification': 'txt',
            'image-classification': 'jpg',
            'object-detection': 'jpg',
            'question-answering': 'json',
            'text-generation': 'txt'
        }
        return f"input.{extensions.get(self.pipeline_type, 'txt')}"