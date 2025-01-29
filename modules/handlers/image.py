"""
Image processing handler for Hugging Face models
"""

from typing import Dict, Any, List
from .base import BaseHandler

class ImageHandler(BaseHandler):
    """Handler for image-based models"""
    
    def generate_imports(self) -> str:
        imports = super().generate_imports()
        return imports + """
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F
from transformers import (
    AutoImageProcessor,
    ViTImageProcessor,
    DeiTImageProcessor,
    AutoProcessor,
    pipeline
)
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    AutoencoderKL
)
"""

    def generate_inference(self) -> str:
        if self.task == 'image-classification':
            return self._generate_classification_inference()
        elif self.task == 'object-detection':
            return self._generate_detection_inference()
        elif self.task == 'image-segmentation':
            return self._generate_segmentation_inference()
        elif self.task == 'text-to-image':
            return self._generate_text_to_image_inference()
        elif self.task == 'image-to-image':
            return self._generate_image_to_image_inference()
        elif self.task == 'image-inpainting':
            return self._generate_inpainting_inference()
        else:
            return self._generate_default_inference()

    def _generate_classification_inference(self) -> str:
        return '''
def process_input(image_path: str, model, processor) -> Dict[str, Any]:
    """Classify image content"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.inference_mode():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
        
        # Get top predictions
        values, indices = probabilities[0].topk(
            min(len(model.config.id2label), int(os.getenv("TOP_K", "5")))
        )
        
        predictions = []
        for value, index in zip(values, indices):
            predictions.append({
                "label": model.config.id2label[index.item()],
                "confidence": float(value)
            })
        
        return {
            "predictions": predictions,
            "metadata": {
                "image_size": image.size,
                "model": self.model_id,
                "task": "image-classification"
            }
        }
        
    except Exception as e:
        return {"error": str(e)}
'''

    def _generate_detection_inference(self) -> str:
        return '''
def process_input(image_path: str, model, processor) -> Dict[str, Any]:
    """Detect objects in image"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        
        # Get parameters
        confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
        
        with torch.inference_mode():
            outputs = model(**inputs)
        
        # Process detections
        detections = []
        scores = outputs.scores[0]
        boxes = outputs.boxes[0]
        labels = outputs.labels[0]
        
        for score, box, label in zip(scores, boxes, labels):
            if score > confidence_threshold:
                detections.append({
                    "label": model.config.id2label[label.item()],
                    "confidence": float(score),
                    "box": {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3])
                    }
                })
        
        # Optionally draw detections
        if os.getenv("DRAW_DETECTIONS", "true").lower() == "true":
            import cv2
            import numpy as np
            
            # Convert PIL to OpenCV format
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            for det in detections:
                box = det["box"]
                cv2.rectangle(
                    image_cv,
                    (int(box["x1"]), int(box["y1"])),
                    (int(box["x2"]), int(box["y2"])),
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    image_cv,
                    f"{det['label']}: {det['confidence']:.2f}",
                    (int(box["x1"]), int(box["y1"] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
            
            # Save annotated image
            output_path = os.path.join(
                os.getenv("OUTPUT_DIR", "/outputs"),
                "detections.jpg"
            )
            cv2.imwrite(output_path, image_cv)
        
        return {
            "detections": detections,
            "num_objects": len(detections),
            "metadata": {
                "image_size": image.size,
                "confidence_threshold": confidence_threshold,
                "model": self.model_id,
                "task": "object-detection"
            }
        }
        
    except Exception as e:
        return {"error": str(e)}
'''

    def _generate_segmentation_inference(self) -> str:
        return '''
def process_input(image_path: str, model, processor) -> Dict[str, Any]:
    """Perform semantic segmentation"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits
            
        # Get the predicted segmentation map
        seg_map = logits[0].argmax(dim=0).cpu().numpy()
        
        # Get unique segments and their areas
        segments = []
        for segment_id in np.unique(seg_map):
            mask = seg_map == segment_id
            segments.append({
                "label": model.config.id2label[segment_id],
                "area": int(mask.sum()),
                "percentage": float(mask.sum() / mask.size * 100)
            })
        
        # Optionally create visualization
        if os.getenv("CREATE_VISUALIZATION", "true").lower() == "true":
            # Create color map for segments
            num_classes = len(model.config.id2label)
            color_map = np.random.randint(0, 255, size=(num_classes, 3))
            
            # Create colored segmentation map
            colored_seg = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
            for segment_id in np.unique(seg_map):
                colored_seg[seg_map == segment_id] = color_map[segment_id]
            
            # Save visualization
            vis_path = os.path.join(
                os.getenv("OUTPUT_DIR", "/outputs"),
                "segmentation.png"
            )
            Image.fromarray(colored_seg).save(vis_path)
        
        return {
            "segments": segments,
            "unique_segments": len(segments),
            "metadata": {
                "image_size": image.size,
                "model": self.model_id,
                "task": "image-segmentation",
                "label_map": model.config.id2label
            }
        }
        
    except Exception as e:
        return {"error": str(e)}
'''

    def _generate_text_to_image_inference(self) -> str:
        return '''
def process_input(prompt: str, model, processor) -> Dict[str, Any]:
    """Generate image from text prompt"""
    try:
        # Get generation parameters
        num_inference_steps = int(os.getenv("NUM_INFERENCE_STEPS", "50"))
        guidance_scale = float(os.getenv("GUIDANCE_SCALE", "7.5"))
        negative_prompt = os.getenv("NEGATIVE_PROMPT", None)
        height = int(os.getenv("HEIGHT", "512"))
        width = int(os.getenv("WIDTH", "512"))
        num_images = int(os.getenv("NUM_IMAGES", "1"))
        
        with torch.inference_mode():
            # Generate images
            outputs = model(
                prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images
            )
        
        # Save generated images
        image_paths = []
        for i, image in enumerate(outputs.images):
            output_path = os.path.join(
                os.getenv("OUTPUT_DIR", "/outputs"),
                f"generated_{i+1}.png"
            )
            image.save(output_path)
            image_paths.append(output_path)
        
        return {
            "image_paths": image_paths,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "parameters": {
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "height": height,
                "width": width
            },
            "metadata": {
                "model": self.model_id,
                "task": "text-to-image"
            }
        }
        
    except Exception as e:
        return {"error": str(e)}
'''

    def _generate_image_to_image_inference(self) -> str:
        return '''
def process_input(image_path: str, model, processor) -> Dict[str, Any]:
    """Transform input image based on prompt"""
    try:
        # Load input image
        init_image = Image.open(image_path).convert('RGB')
        
        # Get parameters
        prompt = os.getenv("PROMPT", "High quality, detailed image")
        strength = float(os.getenv("STRENGTH", "0.75"))
        num_inference_steps = int(os.getenv("NUM_INFERENCE_STEPS", "50"))
        guidance_scale = float(os.getenv("GUIDANCE_SCALE", "7.5"))
        negative_prompt = os.getenv("NEGATIVE_PROMPT", None)
        
        with torch.inference_mode():
            # Generate transformed image
            output = model(
                prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt
            ).images[0]
        
        # Save result
        output_path = os.path.join(
            os.getenv("OUTPUT_DIR", "/outputs"),
            "transformed.png"
        )
        output.save(output_path)
        
        return {
            "output_path": output_path,
            "prompt": prompt,
            "parameters": {
                "strength": strength,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale
            },
            "metadata": {
                "original_size": init_image.size,
                "model": self.model_id,
                "task": "image-to-image"
            }
        }
        
    except Exception as e:
        return {"error": str(e)}
'''

    def _generate_inpainting_inference(self) -> str:
        return '''
def process_input(image_path: str, mask_path: str, model, processor) -> Dict[str, Any]:
    """Inpaint masked region of image"""
    try:
        # Load input image and mask
        init_image = Image.open(image_path).convert('RGB')
        mask_image = Image.open(mask_path).convert('RGB')
        
        # Get parameters
        prompt = os.getenv("PROMPT", "Fill in the masked area naturally")
        num_inference_steps = int(os.getenv("NUM_INFERENCE_STEPS", "50"))
        guidance_scale = float(os.getenv("GUIDANCE_SCALE", "7.5"))
        
        with torch.inference_mode():
            # Generate inpainted image
            output = model(
                prompt=prompt,
                image=init_image,
                mask_image=mask_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
        
        # Save result
        output_path = os.path.join(
            os.getenv("OUTPUT_DIR", "/outputs"),
            "inpainted.png"
        )
        output.save(output_path)
        
        return {
            "output_path": output_path,
            "prompt": prompt,
            "parameters": {
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale
            },
            "metadata": {
                "image_size": init_image.size,
                "model": self.model_id,
                "task": "image-inpainting"
            }
        }
        
    except Exception as e:
        return {"error": str(e)}
'''

    def _generate_default_inference(self) -> str:
        return '''
def process_input(image_path: str, model, processor) -> Dict[str, Any]:
    """Default image processing pipeline"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.inference_mode():
            outputs = model(**inputs)
        
        return {
            "outputs": outputs.logits.tolist() if hasattr(outputs, "logits") else None,
            "metadata": {
                "image_size": image.size,
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
            "pillow>=10.0.0",
            "torchvision>=0.16.0",
            "opencv-python>=4.8.0",
            "diffusers>=0.24.0"
        ])
        return reqs

    def requires_gpu(self) -> bool:
        return self.task in {
            'text-to-image',
            'image-to-image',
            'image-inpainting'
        } or any(x in self.model_id.lower() for x in ['diffusion', 'vit-large'])