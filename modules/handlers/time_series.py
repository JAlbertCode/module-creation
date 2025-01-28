"""Handler for time series data"""

def get_inference_code(model_id, task):
    return f'''import os
import json
import pandas as pd
import numpy as np
from transformers import pipeline
import torch
import argparse

def get_output_dir():
    """Get output directory based on environment"""
    return '/outputs' if os.path.exists('/.dockerenv') else './outputs'

def load_time_series(input_path):
    """Load time series data from various formats"""
    ext = os.path.splitext(input_path)[1].lower()
    
    if ext == '.csv':
        df = pd.read_csv(input_path)
    elif ext == '.json':
        df = pd.read_json(input_path)
    elif ext == '.parquet':
        df = pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    # Try to parse date/time columns
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                continue
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Run inference on {task} model')
    parser.add_argument('--input_path', help='Path to input time series file')
    parser.add_argument('--time_column', help='Name of time/date column')
    parser.add_argument('--value_column', help='Name of value column')
    parser.add_argument('--window_size', type=int, help='Size of sliding window', default=100)
    args = parser.parse_args()

    input_path = args.input_path or os.environ.get('INPUT_PATH')
    time_col = args.time_column or os.environ.get('TIME_COLUMN')
    value_col = args.value_column or os.environ.get('VALUE_COLUMN')
    window_size = args.window_size or int(os.environ.get('WINDOW_SIZE', 100))

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
        
        print("Loading time series data...")
        df = load_time_series(input_path)
        
        if not time_col:
            time_col = df.select_dtypes(include=['datetime64']).columns[0]
        if not value_col:
            value_col = df.select_dtypes(include=['float64', 'int64']).columns[0]
        
        print("Running inference...")
        # Create sliding windows
        values = df[value_col].values
        windows = np.lib.stride_tricks.sliding_window_view(values, window_size)
        
        results = []
        for window in windows:
            result = pipe(window)
            results.append(result)
        
        output = {{
            "result": results,
            "timestamps": df[time_col].tolist(),
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
        "libhdf5-dev",  # For HDF5 format support
        "liblz4-dev"    # For compression support
    ]

def get_requirements():
    return [
        "pandas==2.0.3",
        "pyarrow==14.0.1",  # For parquet support
        "fastparquet==2023.10.1",
        "scipy>=1.0.0",
        "statsmodels>=0.14.0"  # For time series analysis
    ]