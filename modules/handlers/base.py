"""Base handler for all model types"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import os
import json

class BaseHandler(ABC):
    def __init__(self, model_id: str, task: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base handler
        
        Args:
            model_id: Hugging Face model ID
            task: Model task type
            config: Optional model configuration
        """
        self.model_id = model_id
        self.task = task
        self.config = config or {}
        
    @abstractmethod
    def generate_imports(self) -> str:
        """Generate import statements for the model"""
        pass
        
    @abstractmethod
    def generate_inference(self) -> str:
        """Generate inference code for the model"""
        pass
        
    @abstractmethod
    def get_requirements(self) -> List[str]:
        """Get required packages for the model"""
        pass
        
    @abstractmethod
    def requires_gpu(self) -> bool:
        """Check if model requires GPU"""
        pass
        
    def generate_full_code(self) -> str:
        """Generate complete inference script"""
        imports = self.generate_imports()
        inference = self.generate_inference()
        
        return f"""{imports}

import os
import json
from typing import Dict, Any

def save_output(output: Dict[str, Any], output_file: str = "results.json") -> None:
    """Save output to JSON file"""
    output_dir = "/outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")

{inference}

if __name__ == "__main__":
    main()
"""

    def get_docker_instructions(self) -> List[str]:
        """Get Dockerfile instructions specific to this model type"""
        instructions = []
        
        # Add model-specific system dependencies
        if hasattr(self, 'system_dependencies'):
            deps = " ".join(self.system_dependencies)
            instructions.append(f"RUN apt-get update && apt-get install -y {deps}")
            
        # Add cache directory for offline use
        instructions.extend([
            "ENV HF_HOME=/cache/huggingface",
            "RUN mkdir -p /cache/huggingface"
        ])
            
        return instructions
        
    def get_env_vars(self) -> Dict[str, str]:
        """Get required environment variables"""
        return {
            "HF_HUB_OFFLINE": "1",  # Run in offline mode
            "TRANSFORMERS_OFFLINE": "1"
        }
        
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data format"""
        raise NotImplementedError("Subclasses must implement validate_input")
        
    def format_output(self, output: Any) -> Dict[str, Any]:
        """Format model output for saving"""
        raise NotImplementedError("Subclasses must implement format_output")
        
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for lilypad_module.json.tmpl"""
        gpu_count = 1 if self.requires_gpu() else 0
        
        return {
            "machine": {
                "gpu": gpu_count,
                "cpu": 1000,
                "ram": 8000
            },
            "job": {
                "APIVersion": "V1beta1",
                "Spec": {
                    "Deal": {
                        "Concurrency": 1
                    },
                    "Docker": {
                        "Entrypoint": ["python", "/app/run_inference.py"],
                        "WorkingDirectory": "/app",
                        "EnvironmentVariables": [],
                        "Image": "" # To be filled by module generator
                    },
                    "Engine": "Docker",
                    "Network": {
                        "Type": "None"
                    },
                    "Outputs": [
                        {
                            "Name": "outputs",
                            "Path": "/outputs"
                        }
                    ],
                    "PublisherSpec": {
                        "Type": "ipfs"
                    },
                    "Resources": {
                        "GPU": str(gpu_count)
                    },
                    "Timeout": 1800
                }
            }
        }