"""Handler for multimodal models (vision-language, document understanding)"""

import os
from typing import List, Any, Dict, Optional
import torch
from PIL import Image
from .base import BaseHandler

class MultimodalHandler(BaseHandler):
    """Handler for multimodal models"""
    
    TASK_TO_MODEL_CLASS = {
        "visual-question-answering": "AutoModelForVisualQuestionAnswering",
        "document-question-answering": "AutoModelForDocumentQuestionAnswering",
        "document-visual-qa": "AutoModelForDocumentVisualQuestionAnswering",
        "image-text-retrieval": "AutoModelForImageTextRetrieval",
        "visual-reasoning": "AutoModelForVisualReasoning",
        "document-layout-analysis": "AutoModelForDocumentLayoutAnalysis"
    }
    
    def __init__(self, model_id: str, task: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_id, task, config)
        self.system_dependencies = ["libgl1-mesa-glx", "libglib2.0-0", "tesseract-ocr"]
        
    def generate_imports(self) -> str:
        """Generate necessary imports"""
        model_class = self.TASK_TO_MODEL_CLASS.get(self.task)
        imports = [
            "import os",
            "import json",
            "import torch",
            "import numpy as np",
            "from PIL import Image",
            "import pytesseract",
            "from typing import Dict, Any, List"
        ]
        
        imports.extend([
            f"from transformers import AutoProcessor, {model_class}",
            "from transformers.image_utils import load_image"
        ])
        
        return "\n".join(imports)
        
    def generate_inference(self) -> str:
        """Generate task-specific inference code"""
        if "document" in self.task:
            return self._generate_document_qa_code()
        elif "visual-question-answering" in self.task:
            return self._generate_vqa_code()
        elif "image-text-retrieval" in self.task:
            return self._generate_retrieval_code()
        else:
            raise ValueError(f"Unsupported task: {self.task}")
            
    def _generate_vqa_code(self) -> str:
        """Generate VQA code"""
        return '''
def load_model():
    """Load VQA model and processor"""
    processor = AutoProcessor.from_pretrained("./model")
    model = AutoModelForVisualQuestionAnswering.from_pretrained(
        "./model",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, processor

def run_vqa(
    image_path: str,
    question: str,
    model,
    processor
) -> Dict[str, Any]:
    """Run visual question answering"""
    # Load image
    image = load_image(image_path)
    
    # Process inputs
    inputs = processor(
        images=image,
        text=question,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate answer
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode answer
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    
    return {
        "input_image": image_path,
        "question": question,
        "answer": answer,
        "metadata": {
            "model_type": model.config.model_type,
            "supported_tasks": model.config.task_specific_params.keys()
        }
    }

def main():
    """Main inference function"""
    image_path = os.getenv("IMAGE_PATH", "/inputs/image.jpg")
    question = os.getenv("MODEL_INPUT", "What is shown in this image?")
    
    model, processor = load_model()
    results = run_vqa(image_path, question, model, processor)
    
    output_file = os.path.join("/outputs", "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
'''

    def _generate_document_qa_code(self) -> str:
        """Generate document QA code"""
        return '''
def load_model():
    """Load document QA model and processor"""
    processor = AutoProcessor.from_pretrained("./model")
    model = {{ model_type.model_class }}.from_pretrained(
        "./model",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, processor

def extract_text(image: Image.Image) -> str:
    """Extract text from document image"""
    return pytesseract.image_to_string(image)

def run_document_qa(
    image_path: str,
    question: str,
    model,
    processor
) -> Dict[str, Any]:
    """Run document question answering"""
    # Load document image
    image = load_image(image_path)
    
    # Extract text
    extracted_text = extract_text(image)
    
    # Process inputs
    inputs = processor(
        images=image,
        text=question,
        return_tensors="pt"
    ).to(model.device)
    
    # Add extracted text if model supports it
    if hasattr(model.config, "use_ocr_text"):
        inputs["ocr_text"] = [extracted_text]
    
    # Generate answer
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode answer
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    
    return {
        "input_document": image_path,
        "question": question,
        "answer": answer,
        "extracted_text": extracted_text,
        "metadata": {
            "model_type": model.config.model_type,
            "uses_ocr": hasattr(model.config, "use_ocr_text")
        }
    }

def main():
    """Main inference function"""
    image_path = os.getenv("IMAGE_PATH", "/inputs/document.jpg")
    question = os.getenv("MODEL_INPUT", "What is this document about?")
    
    model, processor = load_model()
    results = run_document_qa(image_path, question, model, processor)
    
    output_file = os.path.join("/outputs", "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
'''

    def _generate_retrieval_code(self) -> str:
        """Generate image-text retrieval code"""
        return '''
def load_model():
    """Load retrieval model and processor"""
    processor = AutoProcessor.from_pretrained("./model")
    model = AutoModelForImageTextRetrieval.from_pretrained(
        "./model",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, processor

def run_retrieval(
    image_path: str,
    text: str,
    model,
    processor
) -> Dict[str, Any]:
    """Run image-text retrieval"""
    # Load image
    image = load_image(image_path)
    
    # Process inputs
    inputs = processor(
        images=image,
        text=text,
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    # Get similarity score
    with torch.no_grad():
        outputs = model(**inputs)
        
    similarity = torch.nn.functional.softmax(outputs.logits, dim=-1)
    score = float(similarity[0][1])  # Positive pair score
    
    return {
        "input_image": image_path,
        "input_text": text,
        "similarity_score": score,
        "metadata": {
            "model_type": model.config.model_type,
            "score_threshold": getattr(model.config, "score_threshold", 0.5)
        }
    }

def main():
    """Main inference function"""
    image_path = os.getenv("IMAGE_PATH", "/inputs/image.jpg")
    text = os.getenv("MODEL_INPUT", "A description of the image")
    
    model, processor = load_model()
    results = run_retrieval(image_path, text, model, processor)
    
    output_file = os.path.join("/outputs", "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
'''

    def get_requirements(self) -> List[str]:
        """Get required packages"""
        base_requirements = [
            "torch>=2.0.0",
            "transformers>=4.36.0",
            "pillow>=10.0.0",
            "numpy>=1.24.0",
            "pytesseract>=0.3.10"
        ]
        
        if "document" in self.task:
            base_requirements.extend([
                "pdf2image>=1.16.3",
                "python-poppler"
            ])
            
        return base_requirements
        
    def requires_gpu(self) -> bool:
        """Check if model requires GPU"""
        return True
        
    def validate_input(self, input_data: Any) -> bool:
        """Validate input"""
        if not isinstance(input_data, dict):
            return False
            
        required_keys = ["image", "text"]
        if not all(k in input_data for k in required_keys):
            return False
            
        if not os.path.exists(input_data["image"]):
            return False
            
        try:
            Image.open(input_data["image"])
            return True
        except:
            return False