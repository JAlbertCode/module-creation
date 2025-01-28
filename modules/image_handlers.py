"""
Advanced image processing handlers for Hugging Face models
"""

from typing import Dict, Any, Optional, List
import os
import base64
from io import BytesIO
from .base_handler import BaseHandler

class ImageProcessingHandler(BaseHandler):
    """Advanced handler for image-based models with extended capabilities"""
    
    def generate_imports(self) -> str:
        imports = super().generate_imports() + """
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F
from diffusers import StableDiffusionPipeline, AutoencoderKL
"""
        
        task_specific_imports = {
            'image-to-image': """from diffusers import StableDiffusionImg2ImgPipeline""",
            'image-inpainting': """from diffusers import StableDiffusionInpaintPipeline""",
            'image-segmentation': """from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation""",
            'depth-estimation': """from transformers import DPTImageProcessor, DPTForDepthEstimation""",
            'image-super-resolution': """from diffusers import StableDiffusionUpscalePipeline""",
            'image-style-transfer': """from diffusers import StableDiffusionControlNetPipeline, ControlNetModel""",
            'image-captioning': """from transformers import BlipProcessor, BlipForConditionalGeneration""",
            'facial-recognition': """from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer""",
            'pose-estimation': """from transformers import DetrImageProcessor, DetrForObjectDetection"""
        }
        
        return imports + "\n" + task_specific_imports.get(self.task, '')

    def _generate_common_image_utils(self) -> str:
        """Generate common utility functions for image processing"""
        return '''
def load_image(image_path: str) -> Image.Image:
    """Load and validate image file"""
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {str(e)}")

def save_image(image: Image.Image, output_path: str, format: str = 'PNG') -> str:
    """Save image and return base64 representation"""
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path, format=format)
    
    # Create base64 representation
    buffered = BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode()

def resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """Resize image maintaining aspect ratio"""
    ratio = max_size / max(image.size)
    if ratio < 1:
        new_size = tuple(int(dim * ratio) for dim in image.size)
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image

def apply_image_transforms(image: Image.Image, processor) -> torch.Tensor:
    """Apply model-specific transforms to image"""
    if processor is not None:
        return processor(images=image, return_tensors="pt")["pixel_values"]
    else:
        return F.to_tensor(image).unsqueeze(0)
'''

    def _generate_facial_recognition_inference(self) -> str:
        return '''
def process_input(image_path: str, model, processor) -> Dict[str, Any]:
    """Detect and analyze faces in image"""
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.inference_mode():
        outputs = model(**inputs)
    
    faces = []
    for box, score, landmark in zip(outputs.pred_boxes[0], outputs.scores[0], outputs.pred_keypoints[0]):
        if score > float(os.getenv("CONFIDENCE_THRESHOLD", 0.7)):
            face_data = {
                "bbox": box.tolist(),
                "confidence": float(score),
                "landmarks": {
                    "eyes": landmark[0:4].tolist(),  # Left and right eye
                    "nose": landmark[4:6].tolist(),
                    "mouth": landmark[6:10].tolist(),  # Corners and center
                }
            }
            faces.append(face_data)
    
    # Draw results on image if requested
    if os.getenv("DRAW_RESULTS", "true").lower() == "true":
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)
        for face in faces:
            box = face["bbox"]
            # Draw bounding box
            draw.rectangle(box, outline="red", width=2)
            # Draw landmarks
            for point_group in face["landmarks"].values():
                for x, y in point_group:
                    draw.ellipse([x-2, y-2, x+2, y+2], fill="blue")
        
        output_path = os.path.join(os.getenv("OUTPUT_DIR", "/outputs"), "faces_detected.png")
        base64_image = save_image(draw_image, output_path)
    else:
        output_path = None
        base64_image = None
    
    return {
        "faces": faces,
        "num_faces": len(faces),
        "output_path": output_path,
        "base64_image": base64_image
    }
'''

    def _generate_pose_estimation_inference(self) -> str:
        return '''
def process_input(image_path: str, model, processor) -> Dict[str, Any]:
    """Estimate human poses in image"""
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.inference_mode():
        outputs = model(**inputs)
        
    # Process keypoints
    keypoints = outputs.pred_keypoints[0]  # Shape: [num_instances, num_keypoints, 3]
    scores = outputs.scores[0]
    
    poses = []
    for instance_keypoints, score in zip(keypoints, scores):
        if score > float(os.getenv("CONFIDENCE_THRESHOLD", 0.7)):
            keypoint_data = {
                "confidence": float(score),
                "keypoints": {
                    "head": instance_keypoints[0:5].tolist(),
                    "shoulders": instance_keypoints[5:7].tolist(),
                    "elbows": instance_keypoints[7:9].tolist(),
                    "wrists": instance_keypoints[9:11].tolist(),
                    "hips": instance_keypoints[11:13].tolist(),
                    "knees": instance_keypoints[13:15].tolist(),
                    "ankles": instance_keypoints[15:17].tolist()
                }
            }
            poses.append(keypoint_data)
    
    # Draw results if requested
    if os.getenv("DRAW_RESULTS", "true").lower() == "true":
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)
        
        # Define connections for skeleton visualization
        skeleton_connections = [
            ("head", "shoulders"),
            ("shoulders", "elbows"),
            ("elbows", "wrists"),
            ("shoulders", "hips"),
            ("hips", "knees"),
            ("knees", "ankles")
        ]
        
        for pose in poses:
            # Draw keypoints
            for part_name, points in pose["keypoints"].items():
                for x, y, conf in points:
                    if conf > 0.5:
                        draw.ellipse([x-3, y-3, x+3, y+3], fill="red")
            
            # Draw skeleton
            for conn in skeleton_connections:
                pt1 = pose["keypoints"][conn[0]][0][:2]
                pt2 = pose["keypoints"][conn[1]][0][:2]
                draw.line([pt1[0], pt1[1], pt2[0], pt2[1]], fill="blue", width=2)
        
        output_path = os.path.join(os.getenv("OUTPUT_DIR", "/outputs"), "poses_detected.png")
        base64_image = save_image(draw_image, output_path)
    else:
        output_path = None
        base64_image = None
    
    return {
        "poses": poses,
        "num_people": len(poses),
        "output_path": output_path,
        "base64_image": base64_image
    }
'''

    def _generate_default_inference(self) -> str:
        return '''
def process_input(image_path: str, model, processor) -> Dict[str, Any]:
    """Default image processing pipeline"""
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.inference_mode():
        outputs = model(**inputs)
    
    return {
        "raw_outputs": outputs.logits.tolist(),
        "shape": list(outputs.logits.shape)
    }
'''