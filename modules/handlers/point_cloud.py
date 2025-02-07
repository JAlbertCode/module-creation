"""Handler for point cloud models (classification, segmentation, detection)"""

import os
from typing import List, Any, Dict, Optional
import torch
import numpy as np
from .base import BaseHandler

class PointCloudHandler(BaseHandler):
    """Handler for point cloud processing models"""
    
    TASK_TO_MODEL_CLASS = {
        "point-cloud-classification": "AutoModelForPointCloudClassification",
        "point-cloud-segmentation": "AutoModelForPointCloudSegmentation",
        "point-cloud-detection": "AutoModelForPointCloudDetection",
        "point-cloud-completion": "AutoModelForPointCloudCompletion"
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
            "import numpy as np",
            "import open3d as o3d",
            "from typing import Dict, Any, List"
        ]
        
        imports.extend([
            f"from transformers import AutoProcessor, {model_class}"
        ])
        
        return "\n".join(imports)

    def _generate_classification_code(self) -> str:
        return '''
def load_model():
    """Load point cloud classification model"""
    processor = AutoProcessor.from_pretrained("./model")
    model = AutoModelForPointCloudClassification.from_pretrained(
        "./model",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, processor

def load_point_cloud(file_path: str) -> np.ndarray:
    """Load and preprocess point cloud"""
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return points

def classify_point_cloud(
    file_path: str,
    model,
    processor
) -> Dict[str, Any]:
    """Run point cloud classification"""
    # Load point cloud
    points = load_point_cloud(file_path)
    
    # Process inputs
    inputs = processor(
        point_clouds=points,
        return_tensors="pt"
    ).to(model.device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predictions
    probs = torch.nn.functional.softmax(outputs.logits[0], dim=-1)
    pred_idx = torch.argmax(probs).item()
    
    # Format results
    result = {
        "input_file": file_path,
        "prediction": {
            "label": model.config.id2label[pred_idx],
            "confidence": float(probs[pred_idx])
        },
        "point_cloud_info": {
            "num_points": len(points),
            "bounds": {
                "min": points.min(axis=0).tolist(),
                "max": points.max(axis=0).tolist()
            }
        }
    }
    
    # Save visualization
    vis_path = os.path.join("/outputs", "visualization.ply")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(vis_path, pcd)
    result["visualization_path"] = vis_path
    
    return result

def main():
    """Main inference function"""
    file_path = os.getenv("MODEL_INPUT", "/inputs/pointcloud.ply")
    
    model, processor = load_model()
    results = classify_point_cloud(file_path, model, processor)
    
    output_file = os.path.join("/outputs", "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
'''

    def _generate_segmentation_code(self) -> str:
        return '''
def load_model():
    """Load point cloud segmentation model"""
    processor = AutoProcessor.from_pretrained("./model")
    model = AutoModelForPointCloudSegmentation.from_pretrained(
        "./model",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, processor

def process_point_cloud(
    file_path: str,
    model,
    processor
) -> Dict[str, Any]:
    """Run point cloud segmentation"""
    # Load point cloud
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    
    # Process inputs
    inputs = processor(
        point_clouds=points,
        return_tensors="pt"
    ).to(model.device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predictions
    logits = outputs.logits[0]
    segment_ids = torch.argmax(logits, dim=-1).cpu().numpy()
    
    # Create colored visualization
    colors = np.zeros((len(points), 3))
    num_segments = len(model.config.id2label)
    
    for segment_id in range(num_segments):
        mask = segment_ids == segment_id
        hue = segment_id / num_segments
        rgb = np.array(colorsys.hsv_to_rgb(hue, 0.7, 0.9))
        colors[mask] = rgb
    
    # Save visualization
    vis_path = os.path.join("/outputs", "segmentation.ply")
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(vis_path, pcd)
    
    # Save segment IDs
    segments_path = os.path.join("/outputs", "segments.npy")
    np.save(segments_path, segment_ids)
    
    # Get segment statistics
    unique, counts = np.unique(segment_ids, return_counts=True)
    segment_stats = {
        model.config.id2label[int(idx)]: int(count)
        for idx, count in zip(unique, counts)
    }
    
    return {
        "input_file": file_path,
        "segmentation": {
            "segments_file": segments_path,
            "visualization": vis_path,
            "segment_counts": segment_stats
        },
        "point_cloud_info": {
            "num_points": len(points),
            "bounds": {
                "min": points.min(axis=0).tolist(),
                "max": points.max(axis=0).tolist()
            }
        }
    }

def main():
    """Main inference function"""
    file_path = os.getenv("MODEL_INPUT", "/inputs/pointcloud.ply")
    
    model, processor = load_model()
    results = process_point_cloud(file_path, model, processor)
    
    output_file = os.path.join("/outputs", "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
'''

    def get_requirements(self) -> List[str]:
        base_requirements = [
            "torch>=2.0.0",
            "transformers>=4.36.0",
            "numpy>=1.24.0",
            "open3d>=0.17.0"
        ]
        return base_requirements
        
    def requires_gpu(self) -> bool:
        return True
        
    def validate_input(self, input_data: Any) -> bool:
        if not isinstance(input_data, str):
            return False
        if not os.path.exists(input_data):
            return False
        try:
            import open3d as o3d
            o3d.io.read_point_cloud(input_data)
            return True
        except:
            return False