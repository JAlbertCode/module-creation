"""
Text processing handler for Hugging Face models
"""

from typing import Dict, Any, List
from .base import BaseHandler

class TextHandler(BaseHandler):
    """Handler for text-based models"""
    
    def generate_imports(self) -> str:
        imports = super().generate_imports()
        return imports + """
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    pipeline
)
"""

    def generate_inference(self) -> str:
        if self.task == 'text-classification':
            return self._generate_classification_inference()
        elif self.task == 'text-generation':
            return self._generate_generation_inference()
        elif self.task in {'translation', 'summarization'}:
            return self._generate_seq2seq_inference()
        elif self.task == 'question-answering':
            return self._generate_qa_inference()
        elif self.task == 'token-classification':
            return self._generate_token_classification_inference()
        else:
            return self._generate_default_inference()

    def _generate_classification_inference(self) -> str:
        return '''
def process_input(text: str, model, tokenizer) -> Dict[str, Any]:
    """Classify text into predefined categories"""
    try:
        # Get parameters
        max_length = int(os.getenv("MAX_LENGTH", "512"))
        
        # Process text
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        with torch.inference_mode():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
        
        # Get predictions
        predictions = []
        values, indices = probabilities[0].topk(
            min(len(model.config.id2label), int(os.getenv("TOP_K", "5")))
        )
        
        for value, index in zip(values, indices):
            predictions.append({
                "label": model.config.id2label[index.item()],
                "confidence": float(value)
            })
        
        return {
            "predictions": predictions,
            "text": text,
            "metadata": {
                "model": self.model_id,
                "task": "text-classification",
                "input_length": len(text.split())
            }
        }
        
    except Exception as e:
        return {"error": str(e)}
'''

    def _generate_generation_inference(self) -> str:
        return '''
def process_input(text: str, model, tokenizer) -> Dict[str, Any]:
    """Generate text continuation"""
    try:
        # Get generation parameters
        max_length = int(os.getenv("MAX_LENGTH", "100"))
        temperature = float(os.getenv("TEMPERATURE", "0.7"))
        top_p = float(os.getenv("TOP_P", "0.9"))
        num_return_sequences = int(os.getenv("NUM_SEQUENCES", "1"))
        
        # Process prompt
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=temperature > 0
            )
        
        # Decode generated sequences
        sequences = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return {
            "generated_texts": sequences,
            "prompt": text,
            "parameters": {
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p
            },
            "metadata": {
                "model": self.model_id,
                "task": "text-generation"
            }
        }
        
    except Exception as e:
        return {"error": str(e)}
'''

    def _generate_seq2seq_inference(self) -> str:
        return '''
def process_input(text: str, model, tokenizer) -> Dict[str, Any]:
    """Process sequence-to-sequence tasks (translation, summarization)"""
    try:
        # Get parameters
        max_length = int(os.getenv("MAX_LENGTH", "512"))
        num_beams = int(os.getenv("NUM_BEAMS", "4"))
        
        # Additional parameters for translation
        if self.task == "translation":
            src_lang = os.getenv("SOURCE_LANG")
            tgt_lang = os.getenv("TARGET_LANG")
            if src_lang and tgt_lang:
                text = f">>{src_lang}<< {text}"
        
        # Process input
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=1.0,
                early_stopping=True
            )
        
        # Decode output
        result = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return {
            "output_text": result,
            "input_text": text,
            "parameters": {
                "max_length": max_length,
                "num_beams": num_beams,
                "src_lang": src_lang if self.task == "translation" else None,
                "tgt_lang": tgt_lang if self.task == "translation" else None
            },
            "metadata": {
                "model": self.model_id,
                "task": self.task
            }
        }
        
    except Exception as e:
        return {"error": str(e)}
'''

    def _generate_qa_inference(self) -> str:
        return '''
def process_input(question: str, context: str, model, tokenizer) -> Dict[str, Any]:
    """Answer questions based on context"""
    try:
        # Get parameters
        max_length = int(os.getenv("MAX_LENGTH", "512"))
        
        # Process input
        inputs = tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=max_length,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Handle long contexts with sliding window
        all_answers = []
        for i in range(0, len(inputs["input_ids"])):
            with torch.inference_mode():
                outputs = model(
                    input_ids=inputs["input_ids"][i:i+1],
                    attention_mask=inputs["attention_mask"][i:i+1]
                )
                
                answer_start = outputs.start_logits.argmax()
                answer_end = outputs.end_logits.argmax() + 1
                
                # Convert to original text span
                offset_mapping = inputs["offset_mapping"][i]
                answer = context[
                    offset_mapping[answer_start][0]:
                    offset_mapping[answer_end-1][1]
                ]
                
                confidence = float(
                    outputs.start_logits.softmax(-1).max() *
                    outputs.end_logits.softmax(-1).max()
                )
                
                all_answers.append({
                    "text": answer,
                    "confidence": confidence,
                    "start": int(offset_mapping[answer_start][0]),
                    "end": int(offset_mapping[answer_end-1][1])
                })
        
        # Select best answer
        best_answer = max(all_answers, key=lambda x: x["confidence"])
        
        return {
            "answer": best_answer["text"],
            "confidence": best_answer["confidence"],
            "question": question,
            "context": context,
            "span": {
                "start": best_answer["start"],
                "end": best_answer["end"]
            },
            "metadata": {
                "model": self.model_id,
                "task": "question-answering"
            }
        }
        
    except Exception as e:
        return {"error": str(e)}
'''

    def _generate_token_classification_inference(self) -> str:
        return '''
def process_input(text: str, model, tokenizer) -> Dict[str, Any]:
    """Classify individual tokens (NER, POS tagging)"""
    try:
        # Get parameters
        max_length = int(os.getenv("MAX_LENGTH", "512"))
        
        # Process input
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_offsets_mapping=True
        )
        
        with torch.inference_mode():
            outputs = model(**{
                k: v for k, v in inputs.items()
                if k != "offset_mapping"
            })
            predictions = outputs.logits.argmax(dim=-1)[0]
        
        # Convert predictions to labels with positions
        tokens = []
        offset_mapping = inputs["offset_mapping"][0]
        
        current_entity = None
        for idx, (pred, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            token = text[start:end]
            if not token.strip():
                continue
                
            label = model.config.id2label[pred.item()]
            
            if label.startswith("B-"):  # Beginning of entity
                if current_entity:
                    tokens.append(current_entity)
                current_entity = {
                    "entity": label[2:],
                    "text": token,
                    "start": int(start),
                    "end": int(end)
                }
            elif label.startswith("I-") and current_entity:  # Inside entity
                if current_entity["entity"] == label[2:]:
                    current_entity["text"] += " " + token
                    current_entity["end"] = int(end)
            else:  # Outside any entity
                if current_entity:
                    tokens.append(current_entity)
                    current_entity = None
                tokens.append({
                    "text": token,
                    "start": int(start),
                    "end": int(end),
                    "entity": label
                })
        
        if current_entity:
            tokens.append(current_entity)
        
        return {
            "tokens": tokens,
            "text": text,
            "metadata": {
                "model": self.model_id,
                "task": "token-classification",
                "label_scheme": model.config.label2id
            }
        }
        
    except Exception as e:
        return {"error": str(e)}
'''

    def _generate_default_inference(self) -> str:
        return '''
def process_input(text: str, model, tokenizer) -> Dict[str, Any]:
    """Default text processing pipeline"""
    try:
        # Process input
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=int(os.getenv("MAX_LENGTH", "512")),
            return_tensors="pt"
        )
        
        with torch.inference_mode():
            outputs = model(**inputs)
        
        return {
            "outputs": outputs.logits.tolist() if hasattr(outputs, "logits") else None,
            "text": text,
            "metadata": {
                "model": self.model_id,
                "task": self.task
            }
        }
        
    except Exception as e:
        return {"error": str(e)}
'''

    def get_requirements(self) -> List[str]:
        reqs = super().get_requirements()
        reqs.extend([
            "sentencepiece",
            "protobuf",
            "sacrebleu",
            "rouge-score"
        ])
        return reqs

    def requires_gpu(self) -> bool:
        return self.task in {
            'text-generation',
            'translation'
        } or 'large' in self.model_id.lower()