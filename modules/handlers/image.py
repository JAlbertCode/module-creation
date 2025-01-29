"""Handler for image-based models"""

import os
from typing import List, Any, Dict, Optional
import torch
from PIL import Image
import base64
from io import BytesIO

from .base import BaseHandler

class ImageHandler(BaseHandler):
    """Handler for image models (classification, generation, vision-language)"""
    
    TASK_TO_MODEL_CLASS = {
        "image-classification": "AutoModelForImageClassification",
        "image-to-text": "AutoModelForCausalLM",
        "text-to-image": "StableDiffusionPipeline",  # Special case
        "visual-question-answering": "AutoModelForVisualQuestionAnswering",
        "image-segmentation": "AutoModelForImageSegmentation",
        "object-detection": "AutoModelForObjectDetection"
    }
    
    def __init__(self, model_id: str, task: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_id, task, config)
        self.system_dependencies = ["libgl1-mesa-glx", "libglib2.0-0"]
        
    def generate_imports(self) -> str:
        model_class = self.TASK_TO_MODEL_CLASS.get(self.task)
        imports = [
            "import os",
            "import json",
            "import torch",
            "from PIL import Image",
            "import base64",
            "from io import BytesIO"
        ]
        
        if self.task == "text-to-image":
            imports.extend([
                "from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler",
                "import numpy as np"
            ])
        else:
            imports.extend([
                f"from transformers import AutoProcessor, {model_class}",
                "from transformers.image_utils import load_image"
            ])
            
        return "\n".join(imports)
    
    def generate_inference(self) -> str:
        """Generate task-specific inference code"""
        
        if self.task == "text-to-image":
            return self._generate_text_to_image_code()
        elif self.task == "image-classification":
            return self._generate_classification_code()  
        elif self.task == "image-to-text":
            return self._generate_image_to_text_code()
        elif self.task == "visual-question-answering":
            return self._generate_vqa_code()
        else:
            raise ValueError(f"Unsupported task: {self.task}")
            
    def _generate_text_to_image_code(self) -> str:
        return '''
def load_model():
    """Load text-to-image model"""
    model = StableDiffusionPipeline.from_pretrained(
        "./model",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config)
    model.to("cuda")
    return model

def run_inference(prompt: str, model) -> Dict[str, Any]:
    """Generate image from text"""
    image = model(
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        height=1024,
        width=1024
    ).images[0]
    
    # Convert to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Save image file
    image_path = os.path.join("/outputs", "generated_image.png")
    image.save(image_path)
    
    return {
        "prompt": prompt,
        "image_base64": img_str,
        "image_path": image_path
    }

def main():
    """Main inference function"""
    prompt = os.getenv("MODEL_INPUT", "A beautiful painting of a landscape")
    
    model = load_model()
    results = run_inference(prompt, model)
    save_output(results)
'''
    
    def _generate_classification_code(self) -> str:
        return '''
def load_model():
    """Load image classification model"""
    processor = AutoProcessor.from_pretrained("./model")
    model = AutoModelForImageClassification.from_pretrained(
        "./model",
        torch_dtype=torch.float16
    ).to("cuda")
    return model, processor

def run_inference(image_path: str, model, processor) -> Dict[str, Any]:
    """Classify image"""
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    predicted_class_id = logits.argmax(-1).item()
    probabilities = torch.softmax(logits, dim=-1)[0]
    
    return {
        "predicted_class": model.config.id2label[predicted_class_id],
        "confidence": float(probabilities[predicted_class_id]),
        "probabilities": {
            model.config.id2label[i]: float(prob)
            for i, prob in enumerate(probabilities)
        }
    }

def main():
    """Main inference function"""
    image_path = os.getenv("MODEL_INPUT", "/inputs/image.jpg")
    
    model, processor = load_model()
    results = run_inference(image_path, model, processor)
    save_output(results)
'''

    def _generate_image_to_text_code(self) -> str:
        return '''
def load_model():
    """Load image captioning model"""
    processor = AutoProcessor.from_pretrained("./model")
    model = AutoModelForCausalLM.from_pretrained(
        "./model",
        torch_dtype=torch.float16
    ).to("cuda")
    return model, processor

def run_inference(image_path: str, model, processor) -> Dict[str, Any]:
    """Generate caption for image"""
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=4,
            length_penalty=1.0
        )
        
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    return {
        "caption": caption,
        "image_path": image_path
    }

def main():
    """Main inference function"""
    image_path = os.getenv("MODEL_INPUT", "/inputs/image.jpg")
    
    model, processor = load_model()
    results = run_inference(image_path, model, processor)
    save_output(results)
'''

    def _generate_vqa_code(self) -> str:
        return '''
def load_model():
    """Load visual question answering model"""
    processor = AutoProcessor.from_pretrained("./model")
    model = AutoModelForVisualQuestionAnswering.from_pretrained(
        "./model",
        torch_dtype=torch.float16
    ).to("cuda")
    return model, processor

def run_inference(image_path: str, question: str, model, processor) -> Dict[str, Any]:
    """Answer question about image"""
    image = load_image(image_path)
    inputs = processor(images=image, text=question, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=4,
            early_stopping=True
        )
        
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    
    return {
        "question": question,
        "answer": answer,
        "image_path": image_path
    }

def main():
    """Main inference function"""
    image_path = os.getenv("IMAGE_PATH", "/inputs/image.jpg")
    question = os.getenv("MODEL_INPUT", "What is shown in this image?")
    
    model, processor = load_model()
    results = run_inference(image_path, question, model, processor)
    save_output(results)
'''

    def get_requirements(self) -> List[str]:
        """Get required packages"""
        base_requirements = [
            "torch>=2.0.0",
            "transformers>=4.36.0",
            "pillow>=10.0.0",
            "numpy>=1.24.0"
        ]
        
        if self.task == "text-to-image":
            base_requirements.extend([
                "diffusers>=0.25.0",
                "invisible-watermark>=0.2.0",
                "accelerate>=0.25.0"
            ])
            
        return base_requirements
        
    def requires_gpu(self) -> bool:
        """Check if model requires GPU"""
        # Image models generally need GPU
        return True
        
    def validate_input(self, input_data: Any) -> bool:
        """Validate input based on task"""
        if self.task == "text-to-image":
            return isinstance(input_data, str) and len(input_data.strip()) > 0
        elif self.task == "visual-question-answering":
            if not isinstance(input_data, dict):
                return False
            image_path = input_data.get("image")
            question = input_data.get("question")
            return (isinstance(image_path, str) and os.path.exists(image_path) and
                    isinstance(question, str) and len(question.strip()) > 0)
        else:
            # For image input tasks
            return isinstance(input_data, str) and os.path.exists(input_data)
            
    def format_output(self, output: Any) -> Dict[str, Any]:
        """Format output based on task"""
        if self.task == "image-classification":
            return {
                "label": output.label,
                "confidence": float(output.score),
                "all_labels": [
                    {"label": label, "score": float(score)}
                    for label, score in output.all_scores.items()
                ]
            }
        elif self.task == "text-to-image":
            return {
                "image_path": output.image_path,
                "prompt": output.prompt,
                "base64_image": output.image_base64
            }
        else:
            return output