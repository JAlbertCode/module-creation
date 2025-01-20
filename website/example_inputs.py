class ExampleGenerator:
    def __init__(self, model_info):
        self.model_info = model_info
        self.task_type = self._detect_task_type()
        
    def _detect_task_type(self):
        """Detect the task type from model tags"""
        pipeline_tasks = {
            'image-classification': 'image',
            'object-detection': 'image',
            'text-classification': 'text',
            'sentiment-analysis': 'text',
            'question-answering': 'qa',
            'text-generation': 'text',
            'translation': 'text',
            'summarization': 'text',
            'feature-extraction': 'text'
        }
        
        for tag in self.model_info.tags:
            if tag in pipeline_tasks:
                return pipeline_tasks[tag]
        return 'text'  # default
        
    def get_example_input(self):
        """Get example input based on task type"""
        if self.task_type == 'image':
            return self._get_image_example()
        elif self.task_type == 'qa':
            return self._get_qa_example()
        else:
            return self._get_text_example()
            
    def _get_image_example(self):
        return {
            'format': 'base64',
            'description': 'A base64-encoded image file',
            'example': 'IMAGE_DATA=$(base64 -w 0 your_image.jpg)',
            'curl_example': '''curl -X POST \\
    -H "Content-Type: application/json" \\
    -d '{"input_data": "$IMAGE_DATA"}' \\
    http://localhost:8080/predict'''
        }
        
    def _get_qa_example(self):
        return {
            'format': 'json',
            'description': 'A question and context pair',
            'example': {
                'question': 'What is the capital of France?',
                'context': 'Paris is the capital and largest city of France.'
            },
            'curl_example': '''curl -X POST \\
    -H "Content-Type: application/json" \\
    -d '{"question": "What is the capital of France?", "context": "Paris is the capital and largest city of France."}' \\
    http://localhost:8080/predict'''
        }
        
    def _get_text_example(self):
        return {
            'format': 'text',
            'description': 'Plain text input',
            'example': 'This is an example text for analysis.',
            'curl_example': '''curl -X POST \\
    -H "Content-Type: application/json" \\
    -d '{"input_text": "This is an example text for analysis."}' \\
    http://localhost:8080/predict'''
        }
        
    def get_example_output(self):
        """Get example output format based on task type"""
        if self.task_type == 'image':
            return {
                'predictions': [
                    {'label': 'cat', 'score': 0.95},
                    {'label': 'dog', 'score': 0.03},
                    {'label': 'bird', 'score': 0.02}
                ],
                'status': 'success'
            }
        elif self.task_type == 'qa':
            return {
                'answer': 'Paris',
                'score': 0.98,
                'start': 0,
                'end': 5,
                'status': 'success'
            }
        else:
            return {
                'label': 'positive',
                'score': 0.89,
                'status': 'success'
            }