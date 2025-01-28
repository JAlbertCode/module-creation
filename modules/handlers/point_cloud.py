"""Handler for 3D point cloud and mesh data"""

def get_inference_code(model_id, task):
    return f'''import os
import json
import numpy as np
import trimesh
import open3d as o3d
from transformers import pipeline
import torch
import argparse

def get_output_dir():
    """Get output directory based on environment"""
    return '/outputs' if os.path.exists('/.dockerenv') else './outputs'

def load_point_cloud(input_path):
    """Load point cloud or mesh data"""
    ext = os.path.splitext(input_path)[1].lower()
    
    if ext in ['.ply', '.pcd']:
        # Load as point cloud
        pcd = o3d.io.read_point_cloud(input_path)
        return np.asarray(pcd.points)
    elif ext in ['.obj', '.stl', '.glb', '.gltf']:
        # Load as mesh and convert to point cloud
        mesh = trimesh.load(input_path)
        points = mesh.sample(10000)  # Sample 10k points
        return points
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def main():
    parser = argparse.ArgumentParser(description='Run inference on {task} model')
    parser.add_argument('--input_path', help='Path to input 3D file (PLY/OBJ/STL/etc.)')
    args = parser.parse_args()

    input_path = args.input_path or os.environ.get('INPUT_PATH')
    if not input_path:
        raise ValueError("Please provide input path via --input_path or INPUT_PATH environment variable")
    
    try:
        print("Loading model...")
        device = 0 if torch.cuda.is_available() else -1
        print(f"Device set to use {{'gpu' if torch.cuda.is_available() else 'cpu'}}")
        
        pipe = pipeline(
            task="{task}", 
            model="{model_id}",
            device=device
        )
        
        print("Loading point cloud data...")
        points = load_point_cloud(input_path)
        
        print("Running inference...")
        result = pipe(points)
        output = {{
            "result": result,
            "status": "success"
        }}
        print("Inference complete.")
        
    except Exception as e:
        output = {{"error": str(e), "status": "error"}}
        print(f"Error: {{e}}")
    
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'result.json')
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {{output_path}}")

if __name__ == "__main__":
    main()'''

def get_system_packages():
    return [
        "libeigen3-dev",    # Required for Open3D
        "libgl1-mesa-dev",  # OpenGL support
        "xvfb"              # Virtual framebuffer for headless rendering
    ]

def get_requirements():
    return [
        "open3d==0.17.0",
        "trimesh==4.0.5",
        "numpy<2.0.0",
        "scipy>=1.0.0"
    ]