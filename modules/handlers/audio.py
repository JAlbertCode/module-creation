"""
Audio processing handler for Hugging Face models
"""

from typing import Dict, Any, List
from .base import BaseHandler

class AudioHandler(BaseHandler):
    """Handler for audio-based models"""
    
    def generate_imports(self) -> str:
        imports = super().generate_imports()
        return imports + """
import librosa
import soundfile as sf
import numpy as np
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    AutoProcessor
)
"""

    def generate_inference(self) -> str:
        if self.task == 'automatic-speech-recognition':
            return self._generate_asr_inference()
        elif self.task == 'text-to-speech':
            return self._generate_tts_inference()
        elif self.task == 'speaker-diarization':
            return self._generate_diarization_inference()
        elif self.task == 'audio-classification':
            return self._generate_classification_inference()
        else:
            return self._generate_default_inference()

    def _generate_asr_inference(self) -> str:
        return '''
def process_input(audio_path: str, model, processor) -> Dict[str, Any]:
    """Transcribe speech to text"""
    try:
        # Load audio
        print("Loading audio file...")
        waveform, sample_rate = librosa.load(audio_path)
        
        # Resample if needed
        if sample_rate != processor.sampling_rate:
            waveform = librosa.resample(
                waveform,
                orig_sr=sample_rate,
                target_sr=processor.sampling_rate
            )
        
        # Process audio in chunks for long files
        max_length = int(os.getenv("MAX_LENGTH", "30")) * processor.sampling_rate
        results = []
        
        for i in range(0, len(waveform), max_length):
            chunk = waveform[i:i + max_length]
            inputs = processor(
                chunk,
                sampling_rate=processor.sampling_rate,
                return_tensors="pt"
            )
            
            with torch.inference_mode():
                outputs = model(**inputs)
                predictions = outputs.logits.argmax(dim=-1)
                transcription = processor.batch_decode(predictions)[0]
                results.append(transcription)
        
        return {
            "transcription": " ".join(results),
            "chunks": results,
            "metadata": {
                "duration": len(waveform) / processor.sampling_rate,
                "sample_rate": processor.sampling_rate
            }
        }
        
    except Exception as e:
        return {"error": str(e)}
'''

    def _generate_tts_inference(self) -> str:
        return '''
def process_input(text: str, model, processor) -> Dict[str, Any]:
    """Generate speech from text"""
    try:
        # Process text
        inputs = processor(text=text, return_tensors="pt")
        
        # Get voice ID if specified
        voice_id = os.getenv("VOICE_ID")
        if voice_id:
            inputs["speaker_embeddings"] = model.encode_speaker(voice_id)
        
        # Generate speech
        with torch.inference_mode():
            speech = model.generate_speech(
                inputs["input_ids"],
                processor,
                speaker_embeddings=inputs.get("speaker_embeddings")
            )
        
        # Save audio
        output_path = os.path.join(os.getenv("OUTPUT_DIR", "/outputs"), "generated_speech.wav")
        sf.write(output_path, speech.numpy(), processor.sampling_rate)
        
        return {
            "output_path": output_path,
            "text": text,
            "metadata": {
                "duration": len(speech) / processor.sampling_rate,
                "sample_rate": processor.sampling_rate
            }
        }
        
    except Exception as e:
        return {"error": str(e)}
'''

    def _generate_diarization_inference(self) -> str:
        return '''
def process_input(audio_path: str, model, processor) -> Dict[str, Any]:
    """Identify different speakers in audio"""
    try:
        # Load audio
        waveform, sample_rate = librosa.load(audio_path)
        
        if sample_rate != processor.sampling_rate:
            waveform = librosa.resample(
                waveform,
                orig_sr=sample_rate,
                target_sr=processor.sampling_rate
            )
        
        # Process in chunks
        chunk_length = int(os.getenv("CHUNK_LENGTH", "30")) * processor.sampling_rate
        speakers = []
        
        for i in range(0, len(waveform), chunk_length):
            chunk = waveform[i:i + chunk_length]
            inputs = processor(
                chunk,
                sampling_rate=processor.sampling_rate,
                return_tensors="pt"
            )
            
            with torch.inference_mode():
                outputs = model(**inputs)
                predictions = outputs.logits.argmax(dim=-1)
                
                # Convert predictions to time segments
                for j, speaker_id in enumerate(predictions[0].unique()):
                    speakers.append({
                        "speaker_id": int(speaker_id),
                        "start_time": i / processor.sampling_rate,
                        "end_time": min(
                            (i + chunk_length) / processor.sampling_rate,
                            len(waveform) / processor.sampling_rate
                        )
                    })
        
        return {
            "speakers": speakers,
            "num_speakers": len(set(s["speaker_id"] for s in speakers)),
            "metadata": {
                "duration": len(waveform) / processor.sampling_rate,
                "sample_rate": processor.sampling_rate
            }
        }
        
    except Exception as e:
        return {"error": str(e)}
'''

    def _generate_classification_inference(self) -> str:
        return '''
def process_input(audio_path: str, model, processor) -> Dict[str, Any]:
    """Classify audio content"""
    try:
        # Load audio
        waveform, sample_rate = librosa.load(audio_path)
        
        if sample_rate != processor.sampling_rate:
            waveform = librosa.resample(
                waveform,
                orig_sr=sample_rate,
                target_sr=processor.sampling_rate
            )
        
        # Extract features if needed
        features = None
        if os.getenv("EXTRACT_FEATURES", "false").lower() == "true":
            features = {
                "mfcc": librosa.feature.mfcc(y=waveform, sr=processor.sampling_rate).tolist(),
                "spectral_centroid": librosa.feature.spectral_centroid(
                    y=waveform,
                    sr=processor.sampling_rate
                ).tolist(),
                "zero_crossing_rate": librosa.feature.zero_crossing_rate(
                    y=waveform
                ).tolist()
            }
        
        # Process audio
        inputs = processor(
            waveform,
            sampling_rate=processor.sampling_rate,
            return_tensors="pt"
        )
        
        with torch.inference_mode():
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
        
        # Get top predictions
        values, indices = predictions[0].topk(
            min(len(model.config.id2label), int(os.getenv("TOP_K", "5")))
        )
        
        classifications = [
            {
                "label": model.config.id2label[idx.item()],
                "confidence": float(val)
            }
            for val, idx in zip(values, indices)
        ]
        
        return {
            "classifications": classifications,
            "features": features,
            "metadata": {
                "duration": len(waveform) / processor.sampling_rate,
                "sample_rate": processor.sampling_rate
            }
        }
        
    except Exception as e:
        return {"error": str(e)}
'''

    def _generate_default_inference(self) -> str:
        return '''
def process_input(audio_path: str, model, processor) -> Dict[str, Any]:
    """Default audio processing"""
    try:
        # Load audio
        waveform, sample_rate = librosa.load(audio_path)
        
        if sample_rate != processor.sampling_rate:
            waveform = librosa.resample(
                waveform,
                orig_sr=sample_rate,
                target_sr=processor.sampling_rate
            )
            
        inputs = processor(
            waveform,
            sampling_rate=processor.sampling_rate,
            return_tensors="pt"
        )
        
        with torch.inference_mode():
            outputs = model(**inputs)
        
        return {
            "outputs": outputs.logits.tolist() if hasattr(outputs, "logits") else None,
            "metadata": {
                "duration": len(waveform) / processor.sampling_rate,
                "sample_rate": processor.sampling_rate
            }
        }
        
    except Exception as e:
        return {"error": str(e)}
'''

    def get_requirements(self) -> List[str]:
        reqs = super().get_requirements()
        reqs.extend([
            "librosa>=0.10.1",
            "soundfile>=0.12.1",
            "numpy<2.0.0"
        ])
        return reqs

    def get_system_packages(self) -> List[str]:
        return [
            "libsndfile1",
            "libsndfile1-dev"
        ]

    def requires_gpu(self) -> bool:
        return self.task in {
            'text-to-speech',
            'speaker-diarization'
        }