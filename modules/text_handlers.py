"""
Advanced text processing handlers for Hugging Face models
"""

from typing import Dict, Any, Optional
from .base_handler import BaseHandler

class TextProcessingHandler(BaseHandler):
    """Advanced handler for text-based models with extended capabilities"""
    
    def generate_imports(self) -> str:
        imports = super().generate_imports()
        task_specific_imports = {
            'text-to-text': 'from transformers import AutoModelForSeq2SeqLM',
            'text-generation': 'from transformers import AutoModelForCausalLM',
            'text-classification': 'from transformers import AutoModelForSequenceClassification',
            'token-classification': 'from transformers import AutoModelForTokenClassification',
            'question-answering': 'from transformers import AutoModelForQuestionAnswering',
            'text-summarization': 'from transformers import AutoModelForSeq2SeqLM',
            'zero-shot-classification': 'from transformers import pipeline',
            'sentence-similarity': 'from transformers import AutoModel',
            'fill-mask': 'from transformers import AutoModelForMaskedLM'
        }
        
        return imports + "\n" + task_specific_imports.get(self.task, '')
    
    def generate_inference(self) -> str:
        """Generate inference code based on task"""
        if self.task == 'text-to-text':
            return self._generate_text_to_text_inference()
        elif self.task == 'token-classification':
            return self._generate_token_classification_inference()
        elif self.task == 'question-answering':
            return self._generate_question_answering_inference()
        elif self.task == 'zero-shot-classification':
            return self._generate_zero_shot_inference()
        elif self.task == 'sentence-similarity':
            return self._generate_sentence_similarity_inference()
        elif self.task == 'fill-mask':
            return self._generate_fill_mask_inference()
        else:
            return super().generate_inference()

    def _generate_text_to_text_inference(self) -> str:
        return '''
def process_input(text: str, model, tokenizer):
    """Process text-to-text generation tasks like translation, summarization"""
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=int(os.getenv("MAX_LENGTH", 128)),
        num_beams=int(os.getenv("NUM_BEAMS", 4)),
        length_penalty=float(os.getenv("LENGTH_PENALTY", 1.0)),
        early_stopping=True
    )
    return {
        "generated_text": tokenizer.decode(outputs[0], skip_special_tokens=True),
        "input_text": text
    }
'''

    def _generate_token_classification_inference(self) -> str:
        return '''
def process_input(text: str, model, tokenizer):
    """Process token classification tasks like NER"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    
    # Get predictions
    predictions = outputs.logits.argmax(-1)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    # Convert predictions to labels
    label_list = model.config.id2label
    labeled_tokens = []
    for token, pred in zip(tokens, predictions):
        if token.startswith("##"):
            labeled_tokens[-1]["token"] += token[2:]
        else:
            labeled_tokens.append({
                "token": token,
                "label": label_list[pred]
            })
    
    return {
        "tokens": labeled_tokens,
        "input_text": text
    }
'''

    def _generate_question_answering_inference(self) -> str:
        return '''
def process_input(question: str, context: str, model, tokenizer):
    """Process question answering tasks"""
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    )
    
    outputs = model(**inputs)
    
    answer_start = outputs.start_logits.argmax()
    answer_end = outputs.end_logits.argmax()
    
    answer = tokenizer.decode(inputs.input_ids[0][answer_start:answer_end+1])
    
    return {
        "answer": answer,
        "confidence": float(outputs.start_logits.softmax(-1).max()),
        "question": question,
        "context": context
    }
'''

    def _generate_zero_shot_inference(self) -> str:
        return '''
def process_input(text: str, labels: List[str], model, tokenizer):
    """Process zero-shot classification"""
    classifier = pipeline(
        "zero-shot-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    result = classifier(
        text,
        labels,
        multi_label=bool(os.getenv("MULTI_LABEL", False))
    )
    
    return {
        "labels": result["labels"],
        "scores": result["scores"],
        "input_text": text
    }
'''

    def _generate_sentence_similarity_inference(self) -> str:
        return '''
def process_input(sentence1: str, sentence2: str, model, tokenizer):
    """Process sentence similarity tasks"""
    inputs = tokenizer(
        sentence1,
        sentence2,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    similarity = torch.nn.functional.cosine_similarity(
        embeddings[0].unsqueeze(0),
        embeddings[1].unsqueeze(0)
    )
    
    return {
        "similarity_score": float(similarity),
        "sentence1": sentence1,
        "sentence2": sentence2
    }
'''

    def _generate_fill_mask_inference(self) -> str:
        return '''
def process_input(text: str, model, tokenizer):
    """Process mask filling tasks"""
    # Ensure text contains the mask token
    mask_token = tokenizer.mask_token
    if mask_token not in text:
        text = f"The {mask_token} is bright today."
    
    inputs = tokenizer(text, return_tensors="pt")
    mask_positions = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    
    outputs = model(**inputs)
    predictions = []
    
    for pos in mask_positions:
        logits = outputs.logits[0, pos, :]
        probs = logits.softmax(dim=-1)
        values, indices = probs.topk(5)
        
        predictions.append({
            "position": int(pos),
            "options": [
                {
                    "token": tokenizer.decode([idx]),
                    "score": float(score)
                }
                for idx, score in zip(indices, values)
            ]
        })
    
    return {
        "predictions": predictions,
        "input_text": text
    }
'''