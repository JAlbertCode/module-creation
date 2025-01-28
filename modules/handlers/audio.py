"""Audio input handler for Hugging Face models"""

def get_inference_code(model_id, task):
    return f'''import os
import json
from transformers import pipeline
import torch
import librosa
import soundfile as sf
import argparse

def get_output_dir():
    """Get output directory based on environment"""
    return '/outputs' if os.path.exists('/.dockerenv') else './outputs'

def main():
    parser = argparse.ArgumentParser(description='Run inference on {task} model')
    parser.add_argument('--audio_path', help='Path to input audio file')
    args = parser.parse_args()

    input_path = args.audio_path or os.environ.get('INPUT_PATH')
    if not input_path:
        raise ValueError("Please provide audio path via --audio_path or INPUT_PATH environment variable")
    
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
        waveform, sample_rate = librosa.load(input_path)
        result = pipe({{"raw": waveform, "sampling_rate": sample_rate}})
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
        "libsndfile1",
        "libsndfile1-dev"
    ]

def get_requirements():
    return [
        "librosa==0.10.1",
        "soundfile==0.12.1",
        "numpy<2.0.0"
    ]