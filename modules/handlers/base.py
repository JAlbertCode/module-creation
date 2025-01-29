"""
Base handler class defining the interface for all model type handlers
"""

from typing import Dict, Any, Optional, List
import os
import json
from abc import ABC, abstractmethod

class BaseHandler(ABC):
    """
    Abstract base class for all model handlers.
    Defines the interface that all handlers must implement.
    """
    
    def __init__(self, model_id: str, task: str):
        """
        Initialize handler with model information
        
        Args:
            model_id: Hugging Face model identifier
            task: Model task type
        """
        self.model_id = model_id
        self.task = task
        self._validate_init_params()

    def _validate_init_params(self) -> None:
        """Validate initialization parameters"""
        if not self.model_id or '/' not in self.model_id:
            raise ValueError("Invalid model_id. Must be in format 'owner/name'")
        if not self.task:
            raise ValueError("Task must be specified")

    @abstractmethod
    def generate_imports(self) -> str:
        """
        Generate required import statements
        
        Returns:
            String containing import statements
        """
        return """
import os
import json
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
"""

    @abstractmethod
    def generate_inference(self) -> str:
        """
        Generate model-specific inference code
        
        Returns:
            String containing inference code
        """
        pass

    def generate_setup(self) -> str:
        """
        Generate model and environment setup code
        
        Returns:
            String containing setup code
        """
        return f"""
def setup_model():
    \"\"\"Initialize model and processor\"\"\"
    model_id = "{self.model_id}"
    
    try:
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load processor/tokenizer
        if os.path.exists(os.path.join("model", "tokenizer_config.json")):
            processor = AutoTokenizer.from_pretrained("model")
        else:
            processor = AutoModel.from_pretrained("model")
        
        # Load model
        model = AutoModel.from_pretrained(
            "model",
            device_map=device,
            trust_remote_code=True
        )
        
        return model, processor
        
    except Exception as e:
        raise RuntimeError(f"Failed to setup model: {str(e)}")
"""

    def generate_output_handling(self) -> str:
        """
        Generate output processing and saving code
        
        Returns:
            String containing output handling code
        """
        return """
def save_output(output: Any, output_path: str) -> None:
    \"\"\"Save output to file\"\"\"
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if isinstance(output, (str, int, float, bool, list, dict)):
            # Save JSON-serializable output
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
        elif hasattr(output, 'save'):
            # Handle objects with save method (e.g., torch tensors)
            output.save(output_path)
        else:
            # Convert to string representation
            with open(output_path, 'w') as f:
                f.write(str(output))
                
    except Exception as e:
        raise RuntimeError(f"Failed to save output: {str(e)}")
"""

    def generate_main(self) -> str:
        """
        Generate main execution block
        
        Returns:
            String containing main execution code
        """
        return """
if __name__ == "__main__":
    # Setup model and processor
    model, processor = setup_model()
    
    # Get input/output paths from environment
    input_path = os.getenv("INPUT_PATH", "/workspace/input")
    output_path = os.getenv("OUTPUT_PATH", "/workspace/output")
    
    try:
        # Process input
        result = process_input(input_path, model, processor)
        
        # Save output
        save_output(result, os.path.join(output_path, "result.json"))
        print("Processing completed successfully")
        
    except Exception as e:
        error_output = {
            "error": str(e),
            "type": e.__class__.__name__,
            "details": getattr(e, 'details', None)
        }
        save_output(error_output, os.path.join(output_path, "error.json"))
        print(f"Error during processing: {str(e)}")
        raise
"""

    def get_requirements(self) -> List[str]:
        """
        Get list of required Python packages
        
        Returns:
            List of package requirements
        """
        return [
            "torch>=2.1.0",
            "transformers>=4.36.0",
            "safetensors>=0.4.0"
        ]

    def get_system_packages(self) -> List[str]:
        """
        Get list of required system packages
        
        Returns:
            List of system package requirements
        """
        return []

    def get_module_config(self) -> Dict[str, Any]:
        """
        Get module configuration
        
        Returns:
            Dictionary containing module configuration
        """
        return {
            "machine": {
                "gpu": int(self.requires_gpu()),
                "cpu": 1000,
                "ram": 8000
            },
            "job": {
                "APIVersion": "V1beta1",
                "Spec": {
                    "Deal": {"Concurrency": 1},
                    "Docker": {
                        "Entrypoint": ["python", "/workspace/run_inference.py"],
                        "Image": f"huggingface/{self.model_id.split('/')[-1]}"
                    },
                    "Engine": "Docker",
                    "Network": {"Type": "None"},
                    "Timeouts": {"Running": 1800}
                }
            }
        }

    def requires_gpu(self) -> bool:
        """
        Check if model requires GPU
        
        Returns:
            Boolean indicating GPU requirement
        """
        # Override in subclasses for more specific logic
        return True

    def validate_input(self, input_path: str) -> None:
        """
        Validate input data
        
        Args:
            input_path: Path to input data
        """
        if not os.path.exists(input_path):
            raise ValueError(f"Input path does not exist: {input_path}")

    def generate_code(self) -> str:
        """
        Generate complete module code
        
        Returns:
            String containing complete module code
        """
        return "\n\n".join([
            '"""Generated Lilypad module for {}"""'.format(self.model_id),
            self.generate_imports(),
            self.generate_setup(),
            self.generate_inference(),
            self.generate_output_handling(),
            self.generate_main()
        ])

    def create_module(self, output_dir: str) -> None:
        """
        Create complete Lilypad module
        
        Args:
            output_dir: Directory to create module in
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Write inference code
        with open(os.path.join(output_dir, "run_inference.py"), "w") as f:
            f.write(self.generate_code())
        
        # Write requirements
        with open(os.path.join(output_dir, "requirements.txt"), "w") as f:
            f.write("\n".join(self.get_requirements()))
        
        # Write module config
        with open(os.path.join(output_dir, "lilypad_module.json"), "w") as f:
            json.dump(self.get_module_config(), f, indent=2)
        
        # Write README
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(self._generate_readme())
    
    def _generate_readme(self) -> str:
        """Generate README content"""
        return f"""# {self.model_id} Lilypad Module

This module runs the [{self.model_id}](https://huggingface.co/{self.model_id}) model from Hugging Face
on the Lilypad network.

## Task

This model performs {self.task} using the Hugging Face Transformers library.

## Requirements

- GPU: {self.requires_gpu()}
- System packages: {', '.join(self.get_system_packages()) or 'None'}

## Usage

```bash
lilypad run {self.model_id}
```

## Input Format

See the example inputs in the `examples/` directory.

## Output Format

The module will output a JSON file with the following structure:
```json
{
    "result": {},  # Model-specific output
    "metadata": {}  # Additional information
}
```
"""