"""Handler for structured data inputs (CSV, JSON, etc.)"""

def get_inference_code(model_id, task):
    return f'''import os
import json
import pandas as pd
from transformers import pipeline
import torch
import argparse

def get_output_dir():
    """Get output directory based on environment"""
    return '/outputs' if os.path.exists('/.dockerenv') else './outputs'

def load_input(input_path):
    """Load input data based on file extension"""
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(input_path)
    elif ext == '.json':
        return pd.read_json(input_path)
    elif ext == '.xlsx' or ext == '.xls':
        return pd.read_excel(input_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def main():
    parser = argparse.ArgumentParser(description='Run inference on {task} model')
    parser.add_argument('--input_path', help='Path to input file (CSV/JSON/Excel)')
    parser.add_argument('--column', help='Column name for input data')
    args = parser.parse_args()

    input_path = args.input_path or os.environ.get('INPUT_PATH')
    column = args.column or os.environ.get('COLUMN')

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
        
        print("Loading input data...")
        df = load_input(input_path)
        
        print("Running inference...")
        if column:
            results = [pipe(text) for text in df[column]]
        else:
            results = [pipe(row.to_dict()) for _, row in df.iterrows()]
        
        output = {{
            "result": results,
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
        "liblzma-dev",  # For pandas Excel support
        "libxml2-dev"   # For pandas XML support
    ]

def get_requirements():
    return [
        "pandas==2.0.3",
        "openpyxl==3.1.2",  # For Excel support
        "pyarrow==14.0.1",  # For efficient DataFrame operations
        "xlrd==2.0.1"       # For old Excel format support
    ]