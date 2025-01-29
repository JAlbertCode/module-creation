"""Handler for text-based models"""

from typing import List, Any, Dict
from .base import BaseHandler

class TextHandler(BaseHandler):
    """Handler for text models (classification, generation, translation, etc.)"""
    
    TASK_TO_MODEL_CLASS = {
        "text-classification": "AutoModelForSequenceClassification",
        "text-generation": "AutoModelForCausalLM",
        "translation": "AutoModelForSeq2SeqLM",
        "summarization": "AutoModelForSeq2SeqLM",
        "question-answering": "AutoModelForQuestionAnswering",
        "token-classification": "AutoModelForTokenClassification"
    }
    
    def generate_imports(self) -> str:
        model_class = self.TASK_TO_MODEL_CLASS.get(self.task, "AutoModel")
        
        return f"""from transformers import AutoTokenizer, {model_class}
import torch
import os
from typing import Dict, Any, List, Union
"""
    
    def generate_inference(self) -> str:
        """Generate task-specific inference code"""
        model_class = self.TASK_TO_MODEL_CLASS.get(self.task, "AutoModel")
        
        # Common setup code
        setup_code = '''
def load_model():
    """Load model and tokenizer"""
    model_path = "./model"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = {model_class}.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

'''.format(model_class=model_class)
        
        # Task-specific inference code
        if self.task == "text-classification":
            inference_code = '''
def run_inference(text: str, model, tokenizer) -> Dict[str, Any]:
    """Run classification inference"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1)
    
    return {
        "input": text,
        "prediction": prediction.item(),
        "probabilities": probs[0].tolist()
    }
'''
        elif self.task == "text-generation":
            inference_code = '''
def run_inference(prompt: str, model, tokenizer) -> Dict[str, Any]:
    """Run text generation inference"""
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)
    
    generation_config = {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    outputs = model.generate(
        **inputs,
        **generation_config
    )
    
    generated_text = tokenizer.batch_decode(
        outputs, 
        skip_special_tokens=True
    )[0]
    
    return {
        "prompt": prompt,
        "generated_text": generated_text
    }
'''
        else:
            # Add other task types as needed
            inference_code = '''
def run_inference(input_text: str, model, tokenizer) -> Dict[str, Any]:
    """Run default inference"""
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    return {
        "input": input_text,
        "outputs": outputs.logits[0].tolist()
    }
'''
        
        # Main function to tie everything together
        main_function = '''
def main():
    """Main inference function"""
    # Get input from environment variable
    input_text = os.getenv("MODEL_INPUT", "Default input text")
    
    # Load model
    model, tokenizer = load_model()
    
    # Run inference
    results = run_inference(input_text, model, tokenizer)
    
    # Save results
    save_output(results)

'''
        
        return setup_code + inference_code + main_function
    
    def get_requirements(self) -> List[str]:
        """Get required packages"""
        return [
            "torch>=2.0.0",
            "transformers>=4.36.0",
            "accelerate>=0.25.0",
            "safetensors>=0.4.0"
        ]
    
    def requires_gpu(self) -> bool:
        """Check if model requires GPU based on size and task"""
        # TODO: Implement proper size checking
        return True
        
    def validate_input(self, input_data: Any) -> bool:
        """Validate text input"""
        if not isinstance(input_data, str):
            return False
        if len(input_data.strip()) == 0:
            return False
        return True
        
    def format_output(self, output: Any) -> Dict[str, Any]:
        """Format output based on task"""
        if self.task == "text-classification":
            return {
                "label": output.label,
                "score": float(output.score)
            }
        elif self.task == "text-generation":
            return {
                "generated_text": output[0]["generated_text"]
            }
        else:
            return {"output": output}