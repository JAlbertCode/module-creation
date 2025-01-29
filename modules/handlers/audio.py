"""Handler for audio-based models"""

import os
from typing import List, Any, Dict, Optional
import torch
from .base import BaseHandler

class AudioHandler(BaseHandler):
    """Handler for audio models (ASR, TTS, audio classification)"""
    
    TASK_TO_MODEL_CLASS = {
        "automatic-speech-recognition": "AutoModelForSpeechSeq2Seq",
        "text-to-speech": "AutoModelForTextToSpeech",
        "audio-classification": "AutoModelForAudioClassification",
        "audio-to-audio": "AutoModelForAudioToAudio",
        "audio-frame-classification": "AutoModelForAudioFrameClassification"
    }
    
    def __init__(self, model_id: str, task: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_id, task, config)
        self.system_dependencies = ["libsndfile1", "ffmpeg"]
        
    def generate_imports(self) -> str:
        """Generate necessary imports"""
        model_class = self.TASK_TO_MODEL_CLASS.get(self.task)
        
        imports = [
            "import os",
            "import json",
            "import torch",
            "import numpy as np",
            "import librosa",
            "import soundfile as sf"
        ]
        
        if self.task == "text-to-speech":
            imports.extend([
                "from transformers import VitsTokenizer, VitsModel, AutoProcessor"
            ])
        elif self.task == "automatic-speech-recognition":
            imports.extend([
                f"from transformers import AutoProcessor, {model_class}",
                "from datasets import Audio"
            ])
        else:
            imports.extend([
                f"from transformers import AutoProcessor, {model_class}"
            ])
            
        return "\n".join(imports)
        
    def generate_inference(self) -> str:
        """Generate inference code based on task"""
        if self.task == "automatic-speech-recognition":
            return self._generate_asr_code()
        elif self.task == "text-to-speech":
            return self._generate_tts_code()
        elif self.task == "audio-classification":
            return self._generate_classification_code()
        else:
            raise ValueError(f"Unsupported task: {self.task}")
            
    def _generate_asr_code(self) -> str:
        """Generate code for speech recognition"""
        return '''
def load_model():
    """Load ASR model and processor"""
    processor = AutoProcessor.from_pretrained("./model")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "./model",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, processor

def load_audio(audio_path: str, processor) -> Dict[str, Any]:
    """Load and preprocess audio file"""
    # Load audio
    speech, sr = librosa.load(audio_path, sr=16000)
    
    # Convert to mono if stereo
    if len(speech.shape) > 1:
        speech = speech.mean(axis=0)
    
    # Prepare feature extractor inputs
    inputs = processor(
        speech,
        sampling_rate=16000,
        return_tensors="pt"
    ).to("cuda")
    
    return inputs

def transcribe_audio(
    audio_path: str,
    model,
    processor
) -> Dict[str, Any]:
    """Transcribe audio file"""
    # Load audio
    inputs = load_audio(audio_path, processor)
    
    # Generate transcription
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens={{ model_config.max_new_tokens or 256 }},
            num_beams={{ model_config.num_beams or 4 }}
        )
    
    # Decode output    
    transcription = processor.batch_decode(
        outputs,
        skip_special_tokens=True
    )[0]
    
    return {
        "audio_path": audio_path,
        "transcription": transcription,
        "parameters": {
            "max_new_tokens": {{ model_config.max_new_tokens or 256 }},
            "num_beams": {{ model_config.num_beams or 4 }}
        }
    }

def main():
    """Main inference function"""
    # Get input from environment
    audio_path = os.getenv("MODEL_INPUT", "/inputs/audio.wav")
    
    # Load model
    model, processor = load_model()
    
    # Run inference
    results = transcribe_audio(audio_path, model, processor)
    
    # Save results
    output_path = os.path.join("/outputs", "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
'''

    def _generate_tts_code(self) -> str:
        """Generate code for text-to-speech"""
        return '''
def load_model():
    """Load TTS model, tokenizer and vocoder"""
    tokenizer = VitsTokenizer.from_pretrained("./model")
    model = VitsModel.from_pretrained(
        "./model",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("./model")
    return model, tokenizer, processor

def generate_speech(
    text: str,
    model,
    tokenizer,
    processor
) -> Dict[str, Any]:
    """Generate speech from text"""
    # Tokenize text
    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)
    
    # Get speaker embedding if model supports it
    if hasattr(processor, "get_speaker_embeddings"):
        speaker_embeddings = processor.get_speaker_embeddings()
        inputs["speaker_embeddings"] = speaker_embeddings.to(model.device)
    
    # Generate audio
    with torch.no_grad():
        output = model.generate_speech(**inputs)
    
    # Convert to numpy and save
    audio = output.cpu().numpy()
    audio_path = os.path.join("/outputs", "generated_speech.wav")
    
    # Save audio
    sr = model.config.sampling_rate
    sf.write(audio_path, audio, sr)
    
    return {
        "text": text,
        "audio_path": audio_path,
        "parameters": {
            "sampling_rate": sr,
            "speaker": getattr(model.config, "speaker_name", "default")
        }
    }

def main():
    """Main inference function"""
    # Get input from environment
    text = os.getenv("MODEL_INPUT", "Hello world!")
    
    # Load model
    model, tokenizer, processor = load_model()
    
    # Generate speech
    results = generate_speech(text, model, tokenizer, processor)
    
    # Save results
    output_path = os.path.join("/outputs", "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
'''

    def _generate_classification_code(self) -> str:
        """Generate code for audio classification"""
        return '''
def load_model():
    """Load audio classification model"""
    processor = AutoProcessor.from_pretrained("./model")
    model = AutoModelForAudioClassification.from_pretrained(
        "./model",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, processor

def classify_audio(
    audio_path: str,
    model,
    processor
) -> Dict[str, Any]:
    """Classify audio file"""
    # Get parameters from environment or use defaults
    top_k = int(os.getenv("TOP_K", {{ model_config.top_k or 5 }}))
    threshold = float(os.getenv("THRESHOLD", {{ model_config.threshold or 0.5 }}))
    
    # Load and preprocess audio
    waveform, sr = librosa.load(audio_path, sr=None)
    inputs = processor(
        waveform,
        sampling_rate=sr,
        return_tensors="pt"
    ).to(model.device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities
    probs = torch.nn.functional.softmax(outputs.logits[0], dim=-1)
    
    # Get top predictions
    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))
    
    # Format predictions
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        # Skip if below threshold
        if prob < threshold:
            continue
            
        predictions.append({
            "label": model.config.id2label[idx.item()],
            "confidence": float(prob)
        })
    
    return {
        "audio_path": audio_path,
        "predictions": predictions,
        "parameters": {
            "top_k": top_k,
            "threshold": threshold
        },
        "audio_info": {
            "duration": len(waveform) / sr,
            "sample_rate": sr
        }
    }

def main():
    """Main inference function"""
    # Get input from environment
    audio_path = os.getenv("MODEL_INPUT", "/inputs/audio.wav")
    
    # Load model
    model, processor = load_model()
    
    # Run classification
    results = classify_audio(audio_path, model, processor)
    
    # Save results
    output_path = os.path.join("/outputs", "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
'''

    def get_requirements(self) -> List[str]:
        """Get required packages"""
        base_requirements = [
            "torch>=2.0.0",
            "transformers>=4.36.0",
            "librosa>=0.10.0",
            "soundfile>=0.12.0",
            "numpy>=1.24.0"
        ]
        
        if self.task == "automatic-speech-recognition":
            base_requirements.extend([
                "datasets>=2.14.0"
            ])
            
        return base_requirements
        
    def requires_gpu(self) -> bool:
        """Check if model requires GPU"""
        # Audio models generally need GPU for efficient processing
        return True
        
    def validate_input(self, input_data: Any) -> bool:
        """Validate input based on task"""
        if self.task == "text-to-speech":
            return isinstance(input_data, str) and len(input_data.strip()) > 0
        else:
            # For audio input tasks
            if not isinstance(input_data, str):
                return False
            if not os.path.exists(input_data):
                return False
            # Check if file is audio
            try:
                librosa.load(input_data)
                return True
            except:
                return False
            
    def format_output(self, output: Any) -> Dict[str, Any]:
        """Format output based on task"""
        if self.task == "automatic-speech-recognition":
            return {
                "text": output.transcription,
                "metadata": {
                    "duration": output.get("duration"),
                    "sample_rate": output.get("sample_rate")
                }
            }
        elif self.task == "text-to-speech":
            return {
                "audio_path": output.audio_path,
                "parameters": output.parameters
            }
        else:
            return output  # Already formatted