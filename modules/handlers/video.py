"""Handler for video-based models"""

import os
from typing import List, Any, Dict, Optional
import torch
from .base import BaseHandler

class VideoHandler(BaseHandler):
    """Handler for video models (classification, generation, feature extraction)"""
    
    TASK_TO_MODEL_CLASS = {
        "video-classification": "AutoModelForVideoClassification",
        "video-to-text": "AutoModelForVideoToText",
        "text-to-video": "AutoModelForTextToVideo",
        "video-feature-extraction": "AutoModelForVideoFeatureExtraction",
        "video-frame-classification": "AutoModelForVideoFrameClassification"
    }
    
    def __init__(self, model_id: str, task: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_id, task, config)
        # Add video-specific system dependencies
        self.system_dependencies = [
            "ffmpeg",
            "libgl1-mesa-glx",
            "libglib2.0-0",
            "libsm6",
            "libxext6"
        ]
        
    def generate_imports(self) -> str:
        """Generate necessary imports"""
        model_class = self.TASK_TO_MODEL_CLASS.get(self.task)
        
        imports = [
            "import os",
            "import json",
            "import torch",
            "import numpy as np",
            "import decord",
            "import cv2",
            "from PIL import Image",
            "from pathlib import Path"
        ]
        
        if self.task == "text-to-video":
            imports.extend([
                "from diffusers import AutoPipelineForText2Video",
                "import imageio"
            ])
        else:
            imports.extend([
                f"from transformers import AutoProcessor, {model_class}",
                "from torchvision import transforms"
            ])
            
        return "\n".join(imports)
        
    def generate_inference(self) -> str:
        """Generate task-specific inference code"""
        if self.task == "video-classification":
            return self._generate_classification_code()
        elif self.task == "text-to-video":
            return self._generate_generation_code()
        elif self.task == "video-to-text":
            return self._generate_captioning_code()
        else:
            raise ValueError(f"Unsupported task: {self.task}")
            
    def _generate_classification_code(self) -> str:
        """Generate video classification code"""
        return '''
def load_model():
    """Load video classification model and processor"""
    processor = AutoProcessor.from_pretrained("./model")
    model = AutoModelForVideoClassification.from_pretrained(
        "./model",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, processor

def load_video(video_path: str, processor) -> Dict[str, torch.Tensor]:
    """Load and preprocess video
    
    Args:
        video_path: Path to video file
        processor: Video processor
        
    Returns:
        Dict containing processed video frames
    """
    # Load video with decord
    video_reader = decord.VideoReader(video_path)
    video_len = len(video_reader)
    
    # Get frame sample rate from env or config
    fps = float(os.getenv("FPS", {{ model_config.fps or 4 }}))
    sample_rate = int(video_reader.get_avg_fps() / fps)
    
    # Sample frames
    frame_idxs = list(range(0, video_len, sample_rate))
    frames = video_reader.get_batch(frame_idxs).asnumpy()
    
    # Process frames
    inputs = processor(
        images=list(frames), 
        return_tensors="pt"
    )
    
    return inputs

def classify_video(
    video_path: str,
    model,
    processor
) -> Dict[str, Any]:
    """Classify video content
    
    Args:
        video_path: Path to video file
        model: Classification model
        processor: Video processor
        
    Returns:
        Dict containing predictions and metadata
    """
    # Get parameters from environment
    top_k = int(os.getenv("TOP_K", {{ model_config.top_k or 5 }}))
    threshold = float(os.getenv("THRESHOLD", {{ model_config.threshold or 0.5 }}))
    
    # Load and process video
    inputs = load_video(video_path, processor)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
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
        if prob < threshold:
            continue
            
        predictions.append({
            "label": model.config.id2label[idx.item()],
            "confidence": float(prob)
        })
    
    # Get video metadata
    reader = decord.VideoReader(video_path)
    
    return {
        "video_path": video_path,
        "predictions": predictions,
        "parameters": {
            "top_k": top_k,
            "threshold": threshold,
            "fps": float(os.getenv("FPS", {{ model_config.fps or 4 }}))
        },
        "video_metadata": {
            "duration": len(reader) / reader.get_avg_fps(),
            "fps": reader.get_avg_fps(),
            "frame_count": len(reader),
            "width": reader.width,
            "height": reader.height
        }
    }

def main():
    """Main inference function"""
    # Get input path
    video_path = os.getenv("MODEL_INPUT", "/inputs/video.mp4")
    
    # Load model
    model, processor = load_model()
    
    # Run classification
    results = classify_video(video_path, model, processor)
    
    # Save results
    output_file = os.path.join("/outputs", "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
'''

    def _generate_generation_code(self) -> str:
        """Generate text-to-video code"""
        return '''
def load_model():
    """Load text-to-video model"""
    pipe = AutoPipelineForText2Video.from_pretrained(
        "./model",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    return pipe

def generate_video(
    prompt: str,
    model
) -> Dict[str, Any]:
    """Generate video from text prompt
    
    Args:
        prompt: Text prompt
        model: Text-to-video pipeline
        
    Returns:
        Dict containing generated video and metadata
    """
    # Get generation parameters from environment
    num_frames = int(os.getenv("NUM_FRAMES", {{ model_config.num_frames or 16 }}))
    num_inference_steps = int(os.getenv("NUM_STEPS", {{ model_config.num_inference_steps or 50 }}))
    height = int(os.getenv("HEIGHT", {{ model_config.height or 256 }}))
    width = int(os.getenv("WIDTH", {{ model_config.width or 256 }}))
    fps = int(os.getenv("FPS", {{ model_config.fps or 8 }}))
    
    # Generate video frames
    output = model(
        prompt,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width
    )
    
    video_frames = output.frames[0]
    
    # Save as mp4
    output_path = os.path.join("/outputs", "generated_video.mp4")
    imageio.mimsave(
        output_path,
        video_frames,
        fps=fps,
        quality=8,
        macro_block_size=1
    )
    
    return {
        "prompt": prompt,
        "video_path": output_path,
        "parameters": {
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "height": height,
            "width": width,
            "fps": fps
        }
    }

def main():
    """Main inference function"""
    # Get input prompt
    prompt = os.getenv("MODEL_INPUT", "A beautiful sunset over mountains")
    
    # Load model
    model = load_model()
    
    # Generate video
    results = generate_video(prompt, model)
    
    # Save results
    output_file = os.path.join("/outputs", "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
'''

    def _generate_captioning_code(self) -> str:
        """Generate video-to-text code"""
        return '''
def load_model():
    """Load video captioning model and processor"""
    processor = AutoProcessor.from_pretrained("./model")
    model = AutoModelForVideoToText.from_pretrained(
        "./model",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, processor

def generate_caption(
    video_path: str,
    model,
    processor
) -> Dict[str, Any]:
    """Generate caption for video
    
    Args:
        video_path: Path to video file
        model: Captioning model
        processor: Video processor
        
    Returns:
        Dict containing caption and metadata
    """
    # Get parameters from environment
    max_length = int(os.getenv("MAX_LENGTH", {{ model_config.max_length or 50 }}))
    num_beams = int(os.getenv("NUM_BEAMS", {{ model_config.num_beams or 4 }}))
    
    # Load and process video
    inputs = load_video(video_path, processor)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate caption
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams
        )
    
    # Decode caption
    caption = processor.batch_decode(
        outputs,
        skip_special_tokens=True
    )[0]
    
    # Get video metadata
    reader = decord.VideoReader(video_path)
    
    return {
        "video_path": video_path,
        "caption": caption,
        "parameters": {
            "max_length": max_length,
            "num_beams": num_beams
        },
        "video_metadata": {
            "duration": len(reader) / reader.get_avg_fps(),
            "fps": reader.get_avg_fps(),
            "frame_count": len(reader)
        }
    }

def main():
    """Main inference function"""
    # Get input path
    video_path = os.getenv("MODEL_INPUT", "/inputs/video.mp4")
    
    # Load model
    model, processor = load_model()
    
    # Generate caption
    results = generate_caption(video_path, model, processor)
    
    # Save results
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
            "decord>=0.6.0",
            "opencv-python>=4.8.0",
            "imageio>=2.31.0",
            "imageio-ffmpeg>=0.4.9",
            "pillow>=10.0.0",
            "numpy>=1.24.0"
        ]
        
        if self.task == "text-to-video":
            base_requirements.extend([
                "diffusers>=0.21.0",
                "accelerate>=0.25.0"
            ])
            
        return base_requirements
        
    def requires_gpu(self) -> bool:
        """Check if model requires GPU"""
        # Video models require GPU for reasonable performance
        return True
        
    def validate_input(self, input_data: Any) -> bool:
        """Validate input based on task"""
        if self.task == "text-to-video":
            return isinstance(input_data, str) and len(input_data.strip()) > 0
        else:
            # For video input tasks
            if not isinstance(input_data, str):
                return False
            if not os.path.exists(input_data):
                return False
            # Check if file is video
            try:
                reader = decord.VideoReader(input_data)
                return True
            except:
                return False
            
    def format_output(self, output: Any) -> Dict[str, Any]:
        """Format output based on task"""
        if self.task == "video-to-text":
            return {
                "caption": output.caption,
                "metadata": output.video_metadata
            }
        elif self.task == "text-to-video":
            return {
                "video_path": output.video_path,
                "parameters": output.parameters
            }
        else:
            return output  # Already formatted