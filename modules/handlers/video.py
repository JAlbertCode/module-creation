"""Video input handler for Hugging Face models"""

def get_inference_code(model_id, task):
    return f'''import os
import json
from transformers import pipeline
import torch
import decord
import argparse

def get_output_dir():
    """Get output directory based on environment"""
    return '/outputs' if os.path.exists('/.dockerenv') else './outputs'

def main():
    parser = argparse.ArgumentParser(description='Run inference on {task} model')
    parser.add_argument('--video_path', help='Path to input video file')
    parser.add_argument('--num_frames', type=int, help='Number of frames to process', default=32)
    args = parser.parse_args()

    input_path = args.video_path or os.environ.get('INPUT_PATH')
    if not input_path:
        raise ValueError("Please provide video path via --video_path or INPUT_PATH environment variable")
    
    try:
        print("Loading model...")
        device = 0 if torch.cuda.is_available() else -1
        print(f"Device set to use {{'gpu' if torch.cuda.is_available() else 'cpu'}}")
        
        pipe = pipeline(
            task="{task}", 
            model="{model_id}",
            device=device
        )
        
        print("Running inference...")
        vr = decord.VideoReader(input_path)
        total_frames = len(vr)
        frame_indices = list(range(0, total_frames, total_frames // args.num_frames))[:args.num_frames]
        frames = vr.get_batch(frame_indices).asnumpy()
        result = pipe(frames)
        output = {{"result": result, "status": "success"}}
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
        "ffmpeg",
        "libavcodec-dev",
        "libavformat-dev",
        "libswscale-dev"
    ]

def get_requirements():
    return [
        "decord==0.6.0",
        "numpy<2.0.0"
    ]