"""
Main interface for converting Hugging Face models to Lilypad modules.
"""

import os
import shutil
from typing import Dict, Any, Optional
from dataclasses import dataclass
from .analyzer import ModelAnalyzer
from .template_generator import TemplateGenerator

@dataclass
class ConversionResult:
    """Results of model conversion"""
    model_id: str
    module_path: str
    files: Dict[str, str]
    analysis: Dict[str, Any]

class HFToLilypadConverter:
    """Convert Hugging Face models to Lilypad modules"""
    
    def __init__(self):
        self.analyzer = ModelAnalyzer()
        self.generator = TemplateGenerator()
        
    def convert_model(
        self,
        model_id: str,
        output_dir: str,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> ConversionResult:
        """
        Convert a Hugging Face model to a Lilypad module
        
        Args:
            model_id: Hugging Face model ID (e.g., 'bert-base-uncased')
            output_dir: Directory to save module files
            custom_config: Optional custom configuration overrides
            
        Returns:
            ConversionResult object with module details
        """
        # Create clean output directory
        module_dir = os.path.join(output_dir, model_id.replace('/', '-').lower())
        if os.path.exists(module_dir):
            shutil.rmtree(module_dir)
        os.makedirs(module_dir)
        
        try:
            # Analyze model
            analysis = self.analyzer.analyze_model(model_id)
            
            # Apply any custom configuration
            if custom_config:
                analysis = self._apply_custom_config(analysis, custom_config)
            
            # Generate module files
            files = self.generator.generate_files(analysis, module_dir)
            
            # Generate model downloader
            self._generate_downloader(model_id, module_dir, analysis)
            
            return ConversionResult(
                model_id=model_id,
                module_path=module_dir,
                files=files,
                analysis=analysis.__dict__
            )
            
        except Exception as e:
            # Clean up on failure
            if os.path.exists(module_dir):
                shutil.rmtree(module_dir)
            raise Exception(f"Failed to convert model {model_id}: {str(e)}")
            
    def _apply_custom_config(
        self,
        analysis: Any,
        custom_config: Dict[str, Any]
    ) -> Any:
        """Apply custom configuration overrides"""
        # Update relevant fields from custom config
        for key, value in custom_config.items():
            if hasattr(analysis, key):
                setattr(analysis, key, value)
                
        return analysis
        
    def _generate_downloader(
        self,
        model_id: str,
        module_dir: str,
        analysis: Any
    ) -> None:
        """Generate model download script"""
        downloader_path = os.path.join(module_dir, "download_model.py")
        
        # Generate appropriate download code based on model type
        if "StableDiffusion" in analysis.model_loader:
            download_code = f'''
from diffusers import {analysis.model_loader}

def download_model():
    """Download and cache model files"""
    pipe = {analysis.model_loader}.from_pretrained(
        "{model_id}",
        torch_dtype="auto",
        use_safetensors=True
    )
    pipe.save_pretrained("./model")

if __name__ == "__main__":
    download_model()
'''
        else:
            download_code = f'''
from transformers import {analysis.model_loader}, {analysis.processor_type or "AutoProcessor"}

def download_model():
    """Download and cache model files"""
    # Download model
    model = {analysis.model_loader}.from_pretrained("{model_id}")
    model.save_pretrained("./model")
    
    # Download processor if needed
    if "{analysis.processor_type}":
        processor = {analysis.processor_type}.from_pretrained("{model_id}")
        processor.save_pretrained("./model")

if __name__ == "__main__":
    download_model()
'''
        
        with open(downloader_path, 'w') as f:
            f.write(download_code)
            
    def validate_module(
        self,
        module_path: str,
        test_inputs: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Validate a generated module
        
        Args:
            module_path: Path to generated module
            test_inputs: Optional test inputs for validation
            
        Returns:
            True if validation passes
        """
        required_files = [
            'Dockerfile',
            'run_inference.py',
            'lilypad_module.json.tmpl',
            'requirements.txt',
            'download_model.py'
        ]
        
        # Check all required files exist
        for file in required_files:
            if not os.path.exists(os.path.join(module_path, file)):
                raise ValueError(f"Missing required file: {file}")
                
        # TODO: Add more validation:
        # - Try building Docker image
        # - Run local inference test if test_inputs provided
        # - Validate module config format
        
        return True